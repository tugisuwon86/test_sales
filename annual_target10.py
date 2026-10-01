"""A new geographic/analogue/boosting search for the 10% WAPE target.
Models and rules are selected before scoring the previously inspected 2026 cohort.
"""
from pathlib import Path
import sys, os, json, pickle, warnings, hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.annual_packages'))
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from scipy.optimize import linprog
from scipy import sparse
from threadpoolctl import threadpool_limits
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',message='X does not have valid feature names')
OUT=ROOT/'annual_target10_results'; OUT.mkdir(exist_ok=True)
ASOF=pd.Timestamp('2026-09-30')

def metric(y,p):
    y,p=np.asarray(y),np.asarray(p)
    return dict(n=len(y),WAPE=abs(y-p).sum()/y.sum(),MAPE=(abs(y-p)/y).mean(),MAE_dollars=abs(y-p).mean()*1e6,bias=(p-y).sum()/y.sum())

def data():
    d=pd.read_csv(ROOT/'sales_features.csv',dtype={'location_num':str})
    d=d[d.subtype.isin(['FSU','DTO'])].copy()
    d['open_date']=pd.to_datetime(d.open_date); d['anniversary']=d.open_date+pd.DateOffset(years=1)
    d['year']=d.anniversary.dt.year; d['y']=d.avg_monthly_sales*12/1e6
    eligible=(d.open_date<ASOF-pd.DateOffset(years=1)) & d.y.gt(0) & d.location_type.eq('existing')
    excluded={'location_num','location_name','open_date','anniversary','year','y','location_type','location_status','hex_string','avg_monthly_sales','avg_monthly_sales_log'}
    base=[c for c in d if c not in excluded]
    for c in base:
        if pd.api.types.is_numeric_dtype(d[c]): d[c]=d[c].replace([np.inf,-np.inf],np.nan)
        else: d[c]=d[c].fillna('__MISSING__').astype(str)
    d['opening_year']=d.open_date.dt.year
    d['market_type']=d.state+'_'+d.subtype+'_'+d.no_dt_flag.astype(str)
    d['density_income']=np.log1p(d.total_households_per_sq_mile_1.clip(lower=0))*np.log1p(d.med_household_income_1.clip(lower=0))
    d['worker_resident_ratio']=d.worker_population_1/(d.total_population_1+1)
    d['daytime_resident_ratio']=d.daytime_population_1/(d.total_population_1+1)
    d['anchor_competitor_ratio']=(d.bigbox_1+1)/(d.competitor_1+1)
    d['neighbor_annual']=d.avg_monthly_sales_mean_1year*12/1e6
    d['neighbor_spread']=(d.avg_monthly_sales_75th_1year-d.avg_monthly_sales_25th_1year)/(d.avg_monthly_sales_mean_1year+1)
    engineered=base+[c for c in d if c not in excluded and c not in base]
    return d,eligible,base,list(dict.fromkeys(engineered))

def historic_features(d,labeled):
    """Reference targets must be mature before the subject store opens, including training rows."""
    numeric=['total_population_1','worker_population_1','med_household_income_1','total_households_per_sq_mile_1','food_away_from_home_1','median_average_daily_traffic_1']
    features=pd.DataFrame(index=d.index)
    for index,row in d.iterrows():
        h=labeled[labeled.anniversary<row.open_date].copy()
        if len(h)<10: continue
        assert index not in h.index
        latest=h.year.max(); h=h[h.year>=latest-5]
        sales=h.y.to_numpy()*1.035**np.maximum(0,row.year-h.year.to_numpy())
        # Inflation assumption is capped at 3.5% per year, not fitted to future outcomes.
        global_median=float(np.median(sales)); features.loc[index,'hist_global']=global_median
        for key in ['region','state','dma']:
            mask=h[key].eq(row[key]).to_numpy(); n=int(mask.sum())
            local=np.median(sales[mask]) if n else global_median
            features.loc[index,'hist_'+key]=n/(n+15)*local+15/(n+15)*global_median
            features.loc[index,'hist_'+key+'_n']=n
        lat1,lat2=np.radians(row.latitude),np.radians(h.latitude.to_numpy())
        dlat=lat2-lat1; dlon=np.radians(h.longitude.to_numpy()-row.longitude)
        km=6371*2*np.arcsin(np.sqrt(np.clip(np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2,0,1)))
        penalty=np.where(h.subtype.to_numpy()!=row.subtype,200,0)+np.where(h.no_dt_flag.to_numpy()!=row.no_dt_flag,100,0)
        order=np.argsort(km+penalty)
        for k in [3,10,25]:
            ids=order[:min(k,len(order))]; w=1/(20+km[ids]+penalty[ids])
            features.loc[index,f'hist_geo_{k}']=np.average(sales[ids],weights=w)
        features.loc[index,'hist_nearest_km']=np.min(km)
        hv=np.log1p(h[numeric].clip(lower=0)); rv=np.log1p(pd.to_numeric(row[numeric]).clip(lower=0))
        med=hv.median(); scale=hv.std().replace(0,1).fillna(1)
        distance=(((hv.fillna(med)-rv.fillna(med))/scale).to_numpy()**2).mean(axis=1)
        distance+=np.where(h.subtype.to_numpy()!=row.subtype,3,0)+np.where(h.no_dt_flag.to_numpy()!=row.no_dt_flag,3,0)
        ids=np.argsort(distance)[:min(15,len(h))]
        features.loc[index,'hist_demographic_15']=np.average(sales[ids],weights=1/(.2+distance[ids]))
    return features

def specifications():
    specs=[]
    for fset in ['base','history']:
        for loss in ['regression','regression_l1']:
            for leaves in [7,15,31]:
                specs.append(dict(name=f'lgb_{fset}_{loss}_{leaves}',family='lgb',features=fset,loss=loss,leaves=leaves))
        for loss in ['reg:squarederror','reg:absoluteerror']:
            for depth in [3,5]: specs.append(dict(name=f'xgb_{fset}_{loss}_{depth}',family='xgb',features=fset,loss=loss,depth=depth))
    for transform in ['log','ratio']:
        specs.append(dict(name='lgb_history_'+transform,family='lgb',features='history',loss='regression',leaves=15,transform=transform))
    for depth in [4,6]:
        for transform in ['raw','log']:
            specs.append(dict(name=f'cat_history_{depth}_{transform}',family='cat',features='history',depth=depth,transform=transform))
    for group in ['region','state','dma']:
        specs.append(dict(name=f'experts_{group}',family='experts',group=group,features='history',loss='regression_l1',leaves=7))
    for k in [5,15,40,80]:
        for transform in ['raw','ratio']:
            specs.append(dict(name=f'knn_{k}_{transform}',family='knn',k=k,transform=transform,features='compact'))
    for c in [1.,10.,100.]: specs.append(dict(name=f'svr_{c}',family='svr',C=c,features='compact'))
    for feature in ['hist_geo_3','hist_geo_10','hist_geo_25','hist_demographic_15','hist_state','hist_dma']:
        specs.append(dict(name=feature,family='heuristic',feature=feature,features='history'))
    return specs

def fit_predict(spec,tr,va,sets):
    family=spec['family']; cols=sets[spec['features']]
    if family=='heuristic': return va[spec['feature']].fillna(tr.y.median()).to_numpy()
    if family=='experts':
        global_spec={**spec,'family':'lgb'}; prediction=fit_predict(global_spec,tr,va,sets)
        for value,part in va.groupby(spec['group']):
            local=tr[tr[spec['group']]==value]
            if len(local)>=60:
                lp=fit_predict(global_spec,local,part,sets); ix=va.index.get_indexer(part.index)
                weight=len(local)/(len(local)+150); prediction[ix]=(1-weight)*prediction[ix]+weight*lp
        return prediction
    cat=[c for c in cols if not pd.api.types.is_numeric_dtype(tr[c])]; num=[c for c in cols if c not in cat]
    transform=spec.get('transform','raw'); y=tr.y.to_numpy().copy()
    fill=tr.neighbor_annual.median()
    if transform=='log': y=np.log(y)
    if transform=='ratio': y/=tr.neighbor_annual.fillna(fill).clip(lower=.1).to_numpy()
    if family=='cat':
        model=CatBoostRegressor(iterations=280,depth=spec['depth'],learning_rate=.045,l2_leaf_reg=10,verbose=False,thread_count=4,random_seed=42,allow_writing_files=False)
        model.fit(tr[cols],y,cat_features=cat); prediction=model.predict(va[cols])
    else:
        enc=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),num),('cat',OneHotEncoder(handle_unknown='ignore',min_frequency=5,sparse_output=False),cat)])
        X=enc.fit_transform(tr[cols]); V=enc.transform(va[cols])
        if family=='lgb': model=LGBMRegressor(objective=spec['loss'],n_estimators=280,num_leaves=spec['leaves'],min_child_samples=25,learning_rate=.035,reg_lambda=10,colsample_bytree=.85,verbosity=-1,n_jobs=4,random_state=42)
        elif family=='xgb': model=XGBRegressor(objective=spec['loss'],n_estimators=280,max_depth=spec['depth'],min_child_weight=15,learning_rate=.035,reg_lambda=15,subsample=.85,colsample_bytree=.85,n_jobs=4,random_state=42,tree_method='hist')
        elif family=='knn': model=KNeighborsRegressor(n_neighbors=spec['k'],weights='distance',n_jobs=4)
        else: model=SVR(C=spec['C'],epsilon=.2,gamma='scale')
        model.fit(X,y); prediction=model.predict(V)
    if transform=='log': prediction=np.exp(prediction)
    if transform=='ratio': prediction*=va.neighbor_annual.fillna(fill).clip(lower=.1).to_numpy()
    return np.maximum(.05,prediction)

def fit_weights(X,y):
    n,k=X.shape
    A=sparse.vstack([sparse.hstack([X,-sparse.eye(n)]),sparse.hstack([-X,-sparse.eye(n)])])
    r=linprog(np.r_[np.zeros(k),np.ones(n)],A_ub=A,b_ub=np.r_[y,-y],A_eq=sparse.csr_matrix(np.r_[np.ones(k),np.zeros(n)][None,:]),b_eq=[1],bounds=[(0,None)]*(n+k),method='highs')
    if not r.success: raise RuntimeError(r.message)
    return r.x[:k]

def rule(prediction,frame,hist,hpred,method,strength):
    p=np.asarray(prediction).copy(); residual=hist.y.to_numpy()-hpred
    if method=='none': return p
    if method=='scale':
        ratios=hist.y.to_numpy()/hpred; order=np.argsort(ratios)
        k=np.searchsorted(np.cumsum(hpred[order]),hpred.sum()/2)
        return p*(1+strength*(ratios[order[min(k,len(order)-1)]]-1))
    correction=pd.Series(np.median(residual),index=frame.index)
    if method!='global':
        h=hist.assign(residual=residual)
        levels=[['subtype'],['region','subtype']]
        if method=='state': levels.append(['state','subtype'])
        if method=='dma': levels.extend([['state','subtype'],['dma','subtype']])
        for keys in levels:
            stats=h.groupby(keys).residual.agg(['median','count'])
            for key,b in frame.groupby(keys):
                key=key[0] if len(keys)==1 and isinstance(key,tuple) else key
                if key in stats.index:
                    s=stats.loc[key]; w=s['count']/(s['count']+20)
                    correction.loc[b.index]=(1-w)*correction.loc[b.index]+w*s['median']
    return np.maximum(.05,p+strength*np.clip(correction.to_numpy(),-1.5,1.5))

def main():
    d,eligible,base,engineered=data(); labeled=d.loc[eligible]
    key=hashlib.sha256((ROOT/'sales_features.csv').read_bytes()).hexdigest()[:12]
    history_path=OUT/f'history_features_{key}.csv'
    if history_path.exists(): history=pd.read_csv(history_path,index_col=0).reindex(d.index)
    else:
        history=historic_features(d,labeled); history.to_csv(history_path)
    d=d.join(history); labeled=d.loc[eligible]
    sets={'base':engineered,'history':engineered+history.columns.tolist()}
    sets['compact']=[c for c in sets['history'] if c.startswith('hist_') or c in ['latitude','longitude','region','state','subtype','ssam_flag','no_dt_flag','neighbor_annual','neighbor_spread','total_population_1','worker_population_1','total_households_per_sq_mile_1','med_household_income_1','median_average_daily_traffic_1','opening_year']]
    # Only numeric population/traffic logs are needed for distance learners; scales are fit within each fold.
    folds=[2023,2024,2025]; oof=labeled[labeled.year.isin(folds)]; specs=specifications(); lookup={s['name']:s for s in specs}
    matrix=pd.DataFrame(index=oof.index); results=[]
    cache=OUT/'candidate_oof'; cache.mkdir(exist_ok=True)
    for s in specs:
        path=cache/(key+'_'+s['name'].replace(':','_')+'.csv')
        if path.exists(): p=pd.read_csv(path,index_col=0).prediction.reindex(oof.index)
        else:
            p=pd.Series(index=oof.index,dtype=float)
            for year in folds:
                tr=labeled[labeled.year<year]; va=oof[oof.year==year]
                p.loc[va.index]=fit_predict(s,tr,va,sets)
            p.rename('prediction').to_csv(path)
        assert np.isfinite(p).all()
        matrix[s['name']]=p
        for year in folds:
            m=oof.year==year; results.append(dict(candidate=s['name'],year=year,**metric(oof.loc[m,'y'],p[m])))
        print(s['name'],round(metric(oof.y,p)['WAPE'],4),flush=True)
    # Reuse the exact original no-MALL baseline OOF predictions, without refitting or test-label access.
    baseline_path=next((ROOT/'annual_no_mall_results').glob('oof_baseline_log_*.pkl'))
    with open(baseline_path,'rb') as f: bp,_=pickle.load(f)
    matrix['baseline']=bp.reindex(oof.index)/1e6
    matrix.to_csv(OUT/'oof_matrix.csv'); pd.DataFrame(results).to_csv(OUT/'candidate_validation.csv',index=False)
    dev=oof.year<2025; val=oof.year==2025
    ranking=pd.DataFrame([dict(candidate=c,**metric(oof.loc[dev,'y'],matrix.loc[dev,c])) for c in matrix]).sort_values('WAPE')
    ranking.to_csv(OUT/'development_ranking.csv',index=False)
    names=list(dict.fromkeys(['baseline']+ranking.head(8).candidate.tolist()))
    blends={name:{name:1.} for name in names}
    for name in names:
        if name!='baseline':
            for w in [.25,.5,.75]: blends[f'baseline_{name}_{w}']={'baseline':1-w,name:w}
    w=fit_weights(matrix.loc[dev,names].to_numpy(),oof.loc[dev,'y'].to_numpy())
    for shrink in [0.,.5,1.]: blends[f'L1_{shrink}']={c:float(v) for c,v in zip(names,(1-shrink)*w+shrink/len(names)) if v>1e-7}
    configurations=[('none',0)]+[(m,s) for m in ['global','region','state','dma','scale'] for s in [.25,.5,1.]]
    search=[]
    for name,weights in blends.items():
        raw=sum(matrix[c].to_numpy()*w for c,w in weights.items())
        for method,strength in configurations:
            p=rule(raw[val],oof.loc[val],oof.loc[dev],raw[dev],method,strength)
            search.append(dict(blend=name,method=method,strength=strength,**metric(oof.loc[val,'y'],p)))
    search=pd.DataFrame(search).sort_values('WAPE',kind='stable'); search.to_csv(OUT/'rule_selection_2025.csv',index=False)
    choice=search.iloc[0]; weights=blends[choice.blend]
    selected={'choice':choice.to_dict(),'weights':weights,'specifications':specs,'features':sets,'input_hash':key}
    (OUT/'frozen_selection.json').write_text(json.dumps(selected,indent=2))
    print('FROZEN',choice.to_dict(),weights,flush=True)
    tr=labeled[labeled.year<2026]; test=labeled[labeled.year==2026]
    previous=pd.read_csv(ROOT/'annual_no_mall_results/test_predictions.csv',dtype={'location_num':str}).set_index('location_num').loc[test.location_num]
    test_matrix={'baseline':previous.baseline.to_numpy()/1e6}
    for name in weights:
        if name!='baseline': test_matrix[name]=fit_predict(lookup[name],tr,test,sets)
    raw=sum(test_matrix[name]*weight for name,weight in weights.items())
    history_prediction=sum(matrix[name].to_numpy()*weight for name,weight in weights.items())
    prediction=rule(raw,test,oof,history_prediction,choice.method,float(choice.strength))
    # Preserve the requested conservative ceiling in the final forecast; report raw/adjusted separately.
    ceiling=tr.loc[tr.year==2025,'y'].mean()*1.035
    prediction*=min(1.,ceiling/prediction.mean())
    output=test[['location_num','location_name','year','state','dma','region','subtype','ssam_flag','y']].copy()
    output['baseline']=test_matrix['baseline']; output['raw']=raw; output['prediction']=prediction
    output.to_csv(OUT/'test_predictions_millions.csv',index=False)
    metrics=pd.DataFrame([dict(model=name,**metric(test.y,p)) for name,p in [('baseline',test_matrix['baseline']),('new_raw',raw),('new_adjusted',prediction)]])
    metrics.to_csv(OUT/'test_metrics.csv',index=False)
    subgroup=[]
    for column in ['state','region','dma','subtype','ssam_flag']:
        for value,b in output.groupby(column):
            subgroup.append(dict(grouping=column,group=value,**metric(b.y,b.prediction)))
    pd.DataFrame(subgroup).to_csv(OUT/'test_group_metrics.csv',index=False)
    summary={'selection':selected,'test_metrics':metrics.to_dict(orient='records'),'candidate_count':len(specs),'rule_count':len(search),'reached_10pct':bool(metrics.iloc[-1].WAPE<=.1)}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(metrics.to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4): main()

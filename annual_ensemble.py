"""WAPE-focused annual experiments, temporal ensembles and conservative calibration.

Run with the same Python environment as annual_forecast.py. Inputs remain unchanged.
"""
from pathlib import Path
import os, sys, json, pickle, warnings
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.annual_packages'))
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from sklearn.linear_model import Ridge,HuberRegressor
from scipy.optimize import linprog
from scipy import sparse
from threadpoolctl import threadpool_limits
warnings.filterwarnings('ignore',category=FutureWarning)
OUT=ROOT/'annual_ensemble_results'; OUT.mkdir(exist_ok=True)
ASOF=pd.Timestamp('2026-09-29')

def prepare():
    d=pd.read_csv(ROOT/'sales_features.csv',dtype={'location_num':str})
    d['open_date']=pd.to_datetime(d.open_date)
    d['anniversary']=d.open_date+pd.DateOffset(years=1)
    d['year']=d.anniversary.dt.year
    d['y']=d.avg_monthly_sales*12/1e6
    eligible=(d.open_date<ASOF-pd.DateOffset(years=1)) & d.y.gt(0) & d.location_type.eq('existing')
    excluded={'location_num','location_name','open_date','anniversary','hex_string','location_type','location_status','avg_monthly_sales','avg_monthly_sales_log','year','y'}
    original=[c for c in d if c not in excluded]
    for c in original:
        if not pd.api.types.is_numeric_dtype(d[c]): d[c]=d[c].fillna('__MISSING__').astype(str)
        else: d[c]=d[c].replace([np.inf,-np.inf],np.nan)
    d['opening_year']=d.open_date.dt.year
    d['opening_sin']=np.sin(2*np.pi*d.open_date.dt.month/12)
    d['opening_cos']=np.cos(2*np.pi*d.open_date.dt.month/12)
    d['subtype_ssam']=d.subtype+'_'+d.ssam_flag.astype(str)
    d['region_subtype']=d.region+'_'+d.subtype
    d['neighbor_annual']=d.avg_monthly_sales_mean_1year*12/1e6
    d['neighbor_spread']=(d.avg_monthly_sales_75th_1year-d.avg_monthly_sales_25th_1year)/(d.avg_monthly_sales_mean_1year+1)
    d['closest_relative']=d.avg_monthly_sales_closest_1year/(d.avg_monthly_sales_mean_1year+1)
    d['worker_to_population']=d.worker_population_1/(d.total_population_1+1)
    d['daytime_to_population']=d.daytime_population_1/(d.total_population_1+1)
    d['competitors_per_10k']=1e4*d.competitor_count_1/(d.total_population_1+1)
    added=[c for c in d if c not in original and c not in excluded]
    for c in ['close_highway_dist','close_highway_min']:
        d[c+'_missing_sentinel']=(d[c]>=100000).astype(int)
        d[c+'_clean']=d[c].mask(d[c]>=100000)
        added += [c+'_missing_sentinel',c+'_clean']
    for c in ['total_population_1','worker_population_1','daytime_population_1','total_households_per_sq_mile_1','med_household_income_1','food_away_from_home_1','median_average_daily_traffic_1']:
        d['log_'+c]=np.log1p(d[c].clip(lower=0)); added.append('log_'+c)
    engineered=[c for c in original if not c.startswith('growth_rate_')]+added
    engineered=list(dict.fromkeys(engineered))
    compact=[c for c in engineered if c in ['latitude','longitude','subtype','ssam_flag','region','state','no_dt_flag','neighbor_annual','neighbor_spread','closest_relative','opening_year','opening_sin','opening_cos','worker_to_population','daytime_to_population','competitors_per_10k','subtype_ssam','region_subtype'] or c.startswith('log_')]
    for c in engineered:
        if pd.api.types.is_numeric_dtype(d[c]): d[c]=d[c].replace([np.inf,-np.inf],np.nan)
    return d,eligible,{'original':original,'engineered':engineered,'compact':compact}

def metric(y,p):
    y,p=np.asarray(y),np.asarray(p); err=abs(y-p)
    return {'n':len(y),'WAPE':err.sum()/y.sum(),'MAPE':(err/y).mean(),'MAE_dollars':err.mean()*1e6,'bias':(p-y).sum()/y.sum(),'actual_mean_dollars':y.mean()*1e6,'predicted_mean_dollars':p.mean()*1e6}

def specs():
    s=[]
    def add(name,**kw): s.append(dict(name=name,**kw))
    for loss in ['MAE','RMSE']:
        for mode,window,decay in [('all',0,0),('recent3',3,0),('weighted',0,.35)]:
            add(f'cat_{loss}_{mode}',family='cat',loss=loss,window=window,decay=decay,features='engineered',depth=4)
    for depth in [3,6]: add(f'cat_MAE_depth{depth}',family='cat',loss='MAE',decay=.35,features='engineered',depth=depth)
    add('cat_MAE_compact',family='cat',loss='MAE',decay=.35,features='compact',depth=4)
    add('cat_Huber_weighted',family='cat',loss='Huber:delta=1.0',decay=.35,features='engineered',depth=4)
    add('cat_MAE_uplift35',family='cat',loss='MAE',growth=.035,features='engineered',depth=4)
    add('cat_MAE_neighbor_ratio',family='cat',loss='MAE',decay=.35,transform='ratio',features='compact',depth=4)
    add('cat_RMSE_log',family='cat',loss='RMSE',decay=.35,transform='log',features='engineered',depth=4)
    add('cat_MAE_original',family='cat',loss='MAE',decay=.35,features='original',depth=4)
    for loss in ['absolute_error','squared_error']:
        for window in [0,3]: add(f'hist_{loss}_{window}',family='hist',loss=loss,window=window,decay=.35,features='engineered')
    for window in [0,3]: add(f'extra_{window}',family='extra',window=window,features='engineered')
    add('ridge_log_recent',family='ridge',window=4,transform='log',features='compact')
    add('huber_neighbor_ratio',family='huber',window=4,transform='ratio',features='compact')
    add('historical_group_median',family='heuristic',mode='median',window=3)
    add('neighbor_group_ratio',family='heuristic',mode='ratio',window=3)
    add('neighbor_group_ratio_uplift',family='heuristic',mode='ratio',window=5,growth=.035)
    add('original_notebook_baseline',family='cat',loss='RMSE',transform='log',features='original',depth=6,iterations=200,learning_rate=.05,l2=3)
    add('original_raw_leader',family='cat',loss='RMSE',features='original',depth=5,iterations=350,learning_rate=.05,l2=3)
    return s

def hierarchical_values(tr,frame,values,shrink=15):
    # Keep subtype-specific history rather than replacing sparse MALLs with FSU distribution.
    global_med=float(np.median(values)); result=pd.Series(global_med,index=frame.index)
    for keys in [['subtype'],['subtype','ssam_flag'],['region','subtype','ssam_flag']]:
        stats=tr.assign(value=np.asarray(values)).groupby(keys).value.agg(['median','count'])
        for key,b in frame.groupby(keys,dropna=False):
            lookup=key[0] if len(keys)==1 and isinstance(key,tuple) else key
            if lookup in stats.index:
                row=stats.loc[lookup]; weight=row['count']/(row['count']+shrink)
                result.loc[b.index]=(1-weight)*result.loc[b.index]+weight*row['median']
    return result.to_numpy()

def fit(spec,tr,year,sets):
    window=spec.get('window',0)
    if window: tr=tr[tr.year>=year-window]
    tr=tr.copy(); weights=np.exp(-spec.get('decay',0)*(year-1-tr.year.to_numpy()))
    target=tr.y.to_numpy()*(1+spec.get('growth',0))**(year-tr.year.to_numpy())
    fill=float(tr.neighbor_annual.median()); fill=fill if np.isfinite(fill) else float(tr.y.median())
    if spec['family']=='heuristic':
        values=target/(tr.neighbor_annual.fillna(fill).clip(lower=.1).to_numpy()) if spec['mode']=='ratio' else target
        return {'spec':spec,'tr':tr[['region','subtype','ssam_flag']],'values':values,'fill':fill}
    transform=spec.get('transform','raw')
    if transform=='ratio': target=target/tr.neighbor_annual.fillna(fill).clip(lower=.1).to_numpy()
    if transform=='log': target=np.log(target)
    cols=sets[spec['features']]; cat=[c for c in cols if not pd.api.types.is_numeric_dtype(tr[c])]; num=[c for c in cols if c not in cat]
    family=spec['family']
    if family=='cat':
        model=CatBoostRegressor(iterations=spec.get('iterations',260),depth=spec.get('depth',4),learning_rate=spec.get('learning_rate',.055),l2_leaf_reg=spec.get('l2',8),
            loss_function=spec['loss'],verbose=False,thread_count=4,random_seed=42,allow_writing_files=False)
        model.fit(tr[cols],target,cat_features=cat,sample_weight=weights)
    else:
        enc=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False,min_frequency=5),cat)])
        if family=='hist': est=HistGradientBoostingRegressor(loss=spec['loss'],max_iter=180,max_leaf_nodes=12,min_samples_leaf=20,l2_regularization=10,learning_rate=.06,early_stopping=False,random_state=42)
        elif family=='extra': est=ExtraTreesRegressor(n_estimators=180,min_samples_leaf=7,max_features=.7,n_jobs=4,random_state=42)
        elif family=='huber': est=HuberRegressor(epsilon=1.5,alpha=10,max_iter=500)
        else: est=Ridge(alpha=30)
        model=make_pipeline(enc,est)
        model.fit(tr[cols],target,**{model.steps[-1][0]+'__sample_weight':weights})
    return {'model':model,'spec':spec,'cols':cols,'fill':fill}

def predict(artifact,frame):
    spec=artifact['spec']
    if spec['family']=='heuristic':
        p=hierarchical_values(artifact['tr'],frame,artifact['values'])
        if spec['mode']=='ratio': p=p*frame.neighbor_annual.fillna(artifact['fill']).clip(lower=.1).to_numpy()
    else:
        p=artifact['model'].predict(frame[artifact['cols']])
        if spec.get('transform')=='log': p=np.exp(p)
        if spec.get('transform')=='ratio': p=p*frame.neighbor_annual.fillna(artifact['fill']).clip(lower=.1).to_numpy()
    return np.maximum(p,.05)

def cap(p,tr,year,growth=.035):
    last=tr.year.max(); ceiling=tr.loc[tr.year==last,'y'].mean()*(1+growth)**max(0,year-last)
    return np.asarray(p)*min(1.,ceiling/np.mean(p))

def l1_weights(X,y):
    # Convex least-absolute-error blend: nonnegative weights sum to 1.
    n,k=X.shape
    A=sparse.vstack([sparse.hstack([sparse.csr_matrix(X),-sparse.eye(n)]),sparse.hstack([-sparse.csr_matrix(X),-sparse.eye(n)])])
    result=linprog(np.r_[np.zeros(k),np.ones(n)/n],A_ub=A,b_ub=np.r_[y,-y],A_eq=sparse.csr_matrix(np.r_[np.ones(k),np.zeros(n)][None,:]),b_eq=[1],bounds=[(0,None)]*(n+k),method='highs')
    if not result.success: raise RuntimeError(result.message)
    return result.x[:k]

def calibration(frame,p,history,hpred,config):
    method,strength=config
    if method=='none': return np.asarray(p)
    residual=history.y.to_numpy()-np.asarray(hpred)
    if method=='offset': delta=np.repeat(np.median(residual),len(frame))
    elif method=='subtype':
        delta=hierarchical_values(history,frame,residual,shrink=30)
    elif method=='scale':
        # Optimal multiplier for absolute error is a prediction-weighted median of y/p.
        ratio=history.y.to_numpy()/hpred; order=np.argsort(ratio)
        ix=np.searchsorted(np.cumsum(hpred[order]),np.sum(hpred)/2)
        scale=ratio[order[min(ix,len(order)-1)]]
        return np.maximum(.05,np.asarray(p)*(1+strength*(scale-1)))
    elif method=='compress': return np.maximum(.05,np.median(p)+strength*(np.asarray(p)-np.median(p)))
    return np.maximum(.05,np.asarray(p)+strength*delta)

def main():
    d,eligible,sets=prepare(); labeled=d.loc[eligible]; ss=specs(); lookup={s['name']:s for s in ss}
    folds=[2022,2023,2024,2025]; oof=labeled[labeled.year.isin(folds)].copy()
    matrix=pd.DataFrame(index=oof.index); rows=[]
    cache=OUT/'candidate_oof'; cache.mkdir(exist_ok=True)
    for spec in ss:
        file=cache/(spec['name']+'.csv')
        if file.exists(): values=pd.read_csv(file,index_col=0)['prediction'].reindex(oof.index).to_numpy()
        else:
            series=pd.Series(index=oof.index,dtype=float)
            for year in folds:
                tr=labeled[labeled.year<year]; va=oof[oof.year==year]
                model=fit(spec,tr,year,sets); series.loc[va.index]=predict(model,va)
            series.rename('prediction').to_csv(file); values=series.to_numpy()
        assert np.isfinite(values).all()
        matrix[spec['name']]=values
        for year in folds:
            ix=oof.year==year; rows.append(dict(candidate=spec['name'],year=year,**metric(oof.loc[ix,'y'],values[ix])))
        print('Completed',spec['name'],'OOF WAPE',round(metric(oof.y,values)['WAPE'],4),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'candidate_validation.csv',index=False)
    matrix.to_csv(OUT/'oof_prediction_matrix.csv')
    dev=oof.year<2025; tune=oof.year==2025
    rank=pd.DataFrame([dict(candidate=c,**metric(oof.loc[dev,'y'],matrix.loc[dev,c])) for c in matrix]).sort_values('WAPE')
    rank.to_csv(OUT/'development_ranking.csv',index=False)
    top=list(dict.fromkeys(rank.head(8).candidate.tolist()+['original_notebook_baseline','original_raw_leader'])); blends={}
    for c in top: blends['single__'+c]={c:1.}
    for n in [3,5,8]: blends['mean_top'+str(n)]={c:1/n for c in top[:n]}
    for name in top[:5]:
        if name=='original_notebook_baseline': continue
        for share in [.25,.5,.75]:
            blends[f'baseline_plus_{name}_{share}']={'original_notebook_baseline':1-share,name:share}
    weight=l1_weights(matrix.loc[dev,top].to_numpy(),oof.loc[dev,'y'].to_numpy())
    for shrink in [0.,.5,1.]:
        w=(1-shrink)*weight+shrink/len(top)
        blends['l1_blend_shrink'+str(shrink)]={c:float(v) for c,v in zip(top,w) if v>1e-7}
    configs=[('none',0),('offset',.5),('offset',1),('subtype',.5),('subtype',1),('scale',.5),('scale',1),('compress',.75),('compress',.9),('compress',1.1)]
    selection=[]
    for name,weights in blends.items():
        full=sum(matrix[c].to_numpy()*w for c,w in weights.items())
        for cfg in configs:
            for growth in [0.,.0175,.035]:
                p=calibration(oof.loc[tune],full[tune],oof.loc[dev],full[dev],cfg)
                p=cap(p,labeled[labeled.year<2025],2025,growth)
                selection.append(dict(blend=name,method=cfg[0],strength=cfg[1],growth=growth,**metric(oof.loc[tune,'y'],p)))
    selection=pd.DataFrame(selection).sort_values(['WAPE','MAPE']); selection.to_csv(OUT/'selection_2025.csv',index=False)
    best=selection.iloc[0]; weights=blends[best.blend]; cfg=(best.method,float(best.strength)); growth=float(best.growth)
    selected_oof=sum(matrix[c].to_numpy()*w for c,w in weights.items())
    # Freeze weights/config after 2025; 2026 labels are used only for follow-up scoring.
    selection_json={'selection':best.to_dict(),'weights':weights,'feature_sets':sets,'as_of':str(ASOF.date()),'specifications':ss}
    (OUT/'selection.json').write_text(json.dumps(selection_json,indent=2))
    print('FROZEN SELECTION',best.to_dict(),weights,flush=True)
    tr=labeled[labeled.year<2026]; test=labeled[labeled.year==2026]
    models={c:fit(lookup[c],tr,2026,sets) for c in weights}
    raw=sum(predict(models[c],test)*w for c,w in weights.items())
    pred=cap(calibration(test,raw,oof,selected_oof,cfg),tr,2026,growth)
    trainraw=sum(predict(models[c],tr)*w for c,w in weights.items())
    trainpred=cap(calibration(tr,trainraw,oof,selected_oof,cfg),tr,2026,growth)
    prev=pd.read_csv(ROOT/'annual_results/test_2026_predictions.csv',dtype={'location_num':str}).set_index('location_num').loc[test.location_num]
    output=test[['location_num','location_name','open_date','anniversary','year','subtype','ssam_flag','region','y']].copy()
    output['prediction']=pred; output['raw_prediction']=raw
    output['baseline']=prev.baseline.to_numpy()/1e6; output['previous_adjusted']=prev.selected_adjusted.to_numpy()/1e6
    output.to_csv(OUT/'test_predictions_millions.csv',index=False)
    summary=pd.DataFrame([dict(model=name,**metric(test.y,p)) for name,p in [('notebook_baseline',output.baseline),('previous_adjusted',output.previous_adjusted),('new_raw_ensemble',raw),('new_calibrated_ensemble',pred)]])
    summary.to_csv(OUT/'test_metrics.csv',index=False)
    # Earlier rolling predictions for selected blend are retrospective model-selection diagnostics.
    parts=[]
    for year in folds:
        b=oof[oof.year==year].copy(); b['prediction']=selected_oof[oof.year==year]; b['split']='historical_OOF_raw_selected_blend'; parts.append(b)
    fitted=tr.copy(); fitted['prediction']=trainpred; fitted['split']='train_in_sample_optimistic'; parts.append(fitted)
    tested=test.copy(); tested['prediction']=pred; tested['split']='test_2026_followup'; parts.append(tested)
    combined=pd.concat(parts)
    combined[['location_num','year','subtype','ssam_flag','region','y','prediction','split']].to_csv(OUT/'evaluation_predictions_millions.csv',index=False)
    seg=[]
    for split,b in combined.groupby('split'):
        for columns in [['subtype'],['ssam_flag'],['subtype','ssam_flag'],['year']]:
            for key,g in b.groupby(columns,dropna=False):
                key=key if isinstance(key,tuple) else (key,)
                seg.append(dict(split=split,grouping='+'.join(columns),group=' | '.join(map(str,key)),**metric(g.y,g.prediction)))
    pd.DataFrame(seg).to_csv(OUT/'segment_metrics.csv',index=False)
    # Production refit after scoring, on all mature labels.
    final={c:fit(lookup[c],labeled,2026,sets) for c in weights}
    inf=d.loc[~eligible].copy(); inf['forecast_year']=inf.year.clip(lower=2026).fillna(2026).astype(int)
    inf['raw_prediction']=sum(predict(final[c],inf)*w for c,w in weights.items())
    inf['prediction']=np.nan
    for year,b in inf.groupby('forecast_year'):
        inf.loc[b.index,'prediction']=cap(calibration(b,b.raw_prediction.to_numpy(),oof,selected_oof,cfg),labeled,year,growth)
    export=inf[['location_num','location_name','location_type','open_date','anniversary','forecast_year','subtype','ssam_flag','region']].copy()
    export['annual_sales_forecast']=inf.prediction*1e6; export['raw_annual_sales_forecast']=inf.raw_prediction*1e6
    export['reason']=np.where(inf.open_date>=ASOF-pd.DateOffset(years=1),'immature_or_future','missing_target')
    export.to_csv(OUT/'inference_forecasts.csv',index=False)
    export[export.location_type=='pipeline'].to_csv(OUT/'pipeline_forecasts.csv',index=False)
    with open(OUT/'ensemble.pkl','wb') as f: pickle.dump({'models':final,'weights':weights,'calibration':cfg,'growth':growth,'oof':oof,'oof_predictions':selected_oof,'reference':labeled[['year','y']],'as_of':ASOF},f)
    assert set(test.location_num).isdisjoint(export.location_num)
    assert np.isfinite(export.annual_sales_forecast).all() and export.annual_sales_forecast.gt(0).all()
    (OUT/'summary.json').write_text(json.dumps({'selection':best.to_dict(),'weights':weights,'test_metrics':summary.to_dict(orient='records'),'training_n':len(tr),'test_n':len(test),'inference_n':len(inf)},indent=2))
    print(summary.to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4): main()

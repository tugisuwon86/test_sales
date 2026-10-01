"""Reproducible annualized sales experiments. Run from the directory containing the CSV."""
from pathlib import Path
import sys, json, warnings, pickle
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '.annual_packages'))
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', category=FutureWarning)
OUT = ROOT / 'annual_results'
OUT.mkdir(exist_ok=True)
ASOF = pd.Timestamp('2026-09-29')

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    e = p-y
    lo, hi = np.quantile(y, [.2,.8])
    return dict(n=len(y), MAE=abs(e).mean(), RMSE=np.sqrt((e**2).mean()),
                MAPE=(abs(e)/y).mean(), WAPE=abs(e).sum()/y.sum(), R2=r2_score(y,p),
                bias=e.mean()/y.mean(), low_MAPE=(abs(e[y<=lo])/y[y<=lo]).mean(),
                high_MAPE=(abs(e[y>=hi])/y[y>=hi]).mean(), actual_mean=y.mean(), predicted_mean=p.mean())

def fit_model(kind, tr, cols):
    cat=[c for c in cols if not pd.api.types.is_numeric_dtype(tr[c])]
    num=[c for c in cols if c not in cat]
    log = kind.endswith('_log')
    y=np.log(tr.y) if log else tr.y
    if kind.startswith('cat'):
        m=CatBoostRegressor(iterations=350 if kind!='cat_baseline_log' else 200,
             depth=5 if kind!='cat_baseline_log' else 6, learning_rate=.05,
             loss_function='RMSE', verbose=False, thread_count=4, random_seed=42,
             allow_writing_files=False)
        m.fit(tr[cols], y, cat_features=cat)
    else:
        enc=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),num),
             ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
        est=Ridge(alpha=100) if kind.startswith('ridge') else (ExtraTreesRegressor(n_estimators=250,min_samples_leaf=8,max_features=.8,n_jobs=4,random_state=42) if kind.startswith('extra') else HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=15,l2_regularization=10,random_state=42))
        m=make_pipeline(enc,est); m.fit(tr[cols],y)
    return m

def predict(m, kind, frame, cols):
    p=m.predict(frame[cols]); return np.maximum(np.exp(p) if kind.endswith('_log') else p,1)

def reference(tr, batch, year, grouping, growth):
    # Use up to three previous anniversary cohorts; pool sparse groups.
    hist=tr[tr.year>=year-3]
    if len(hist)<30: hist=tr
    ans=[]
    keys=[] if grouping=='global' else (['subtype'] if grouping=='subtype' else ['region','subtype'])
    grouper=batch.groupby(keys,dropna=False,sort=False) if keys else [('all',batch)]
    for key,b in grouper:
        ref=hist
        for c in keys: ref=ref[ref[c]==b[c].iloc[0]]
        if len(ref)<30 and 'subtype' in keys: ref=hist[hist.subtype==b.subtype.iloc[0]]
        if len(ref)<30: ref=hist
        vals=ref.y.to_numpy() * (1+growth)**(year-ref.year.to_numpy())
        ans.append((b.index,vals))
    return ans

def postprocess(p, batch, tr, year, method, strength, grouping, growth):
    result=pd.Series(p,index=batch.index,dtype=float)
    if method=='raw': return result.to_numpy()
    for idx,vals in reference(tr,batch,year,grouping,growth):
        v=result.loc[idx].to_numpy(); center=v.mean()
        if method=='linear': out=center+strength*(v-center)
        elif method=='power': out=center*(v/center)**strength
        elif method=='quantile':
            # Batch rank mapping is intentionally transductive: no batch labels used.
            ranks=pd.Series(v).rank(method='average').to_numpy()
            q=(ranks-.5)/len(v)
            mapped=np.quantile(vals,q)
            out=(1-strength)*v+strength*mapped
        elif method=='mean_anchor': out=v*(1-strength+strength*vals.mean()/center)
        out=np.maximum(out,1)
        # All adjusted candidates respect a historical, mix-aware mean ceiling.
        out*=min(1.,vals.mean()/out.mean())
        result.loc[idx]=out
    # Portfolio guardrail: pooled multi-year references must not imply a jump
    # above the latest observed cohort mean grown at the chosen <=3.5% rate.
    latest_year=tr.year.max()
    ceiling=tr.loc[tr.year==latest_year,'y'].mean()*(1+growth)**max(0,year-latest_year)
    result*=min(1.,ceiling/result.mean())
    return result.to_numpy()

def main():
    d=pd.read_csv(ROOT/'sales_features.csv',dtype={'location_num':str})
    d['open_date']=pd.to_datetime(d['open_date'],errors='coerce')
    d['anniversary']=d.open_date+pd.DateOffset(years=1)
    d['year']=d.anniversary.dt.year
    d['y']=pd.to_numeric(d.avg_monthly_sales,errors='coerce')*12
    mature=d.open_date < ASOF-pd.DateOffset(years=1)
    eligible=mature & d.y.notna() & (d.y>0) & d.location_type.eq('existing')
    d['opening_month']=d.open_date.dt.month
    exclude={'location_num','location_name','location_type','location_status','open_date','anniversary','hex_string','avg_monthly_sales','avg_monthly_sales_log','year','y','month_of_opening'}
    allcols=[c for c in d if c not in exclude]
    sales=[c for c in allcols if c.startswith(('avg_monthly_sales_','growth_rate_'))]
    dynamic=[c for c in allcols if c.startswith(('boundary_','months_'))]
    core=[c for c in allcols if c not in sales+dynamic]
    compact=[c for c in core if c in ['latitude','longitude','subtype','region','state','ssam_flag','no_dt_flag','opening_month','total_population_1','worker_population_1','med_household_income_1','competitor_count_1','total_households_per_sq_mile_1','close_highway_dist','median_average_daily_traffic_1']]
    sets={'compact':compact,'static':core,'static_boundary':core+dynamic,'all':allcols,'no_growth':[c for c in allcols if not c.startswith('growth_rate_')]}
    for c in allcols:
        if not pd.api.types.is_numeric_dtype(d[c]): d[c]=d[c].fillna('__MISSING__').astype(str)
        else: d[c]=d[c].replace([np.inf,-np.inf],np.nan)
    labeled=d.loc[eligible].copy()
    labeled.groupby('year').agg(n=('y','size'),mean_annual_sales=('y','mean')).to_csv(OUT/'cohorts.csv')
    audit={'as_of':str(ASOF.date()),'strict_open_date_cutoff':str((ASOF-pd.DateOffset(years=1)).date()),'rows':len(d),'eligible':len(labeled),'test_2026':int((labeled.year==2026).sum()),'immature_labeled_excluded':int((~mature & d.y.notna()).sum()),'duplicate_locations':int(d.location_num.duplicated().sum()),'features':sets}
    (OUT/'data_audit.json').write_text(json.dumps(audit,indent=2))
    folds=[2023,2024,2025]
    cache={}; rows=[]
    kinds=['cat_baseline_log','cat_raw','ridge_log','extra_raw','hist_raw']
    recalibrate='--calibrate-only' in sys.argv
    previous=pd.read_csv(OUT/'model_ranking.csv',index_col=0).nsmallest(4,'selection_score').index.tolist() if recalibrate else []
    for fs,cols in sets.items():
        for kind in kinds:
            ident=fs+'__'+kind; cache[ident]=[]
            if recalibrate and ident not in previous: continue
            for yr in folds:
                tr=labeled[labeled.year<yr]; va=labeled[labeled.year==yr]
                m=fit_model(kind,tr,cols); p=predict(m,kind,va,cols)
                cache[ident].append((yr,tr,va,p))
                rows.append(dict(candidate=ident,year=yr,**metrics(va.y,p)))
            print('completed',ident,flush=True)
    scores=pd.read_csv(OUT/'model_validation.csv') if recalibrate else pd.DataFrame(rows)
    scores.to_csv(OUT/'model_validation.csv',index=False)
    pd.concat([pd.DataFrame({'candidate':ident,'year':yr,'row_index':va.index,'actual':va.y,'prediction':p}) for ident,entries in cache.items() for yr,tr,va,p in entries]).to_csv(OUT/'validation_predictions.csv',index=False)
    ranks=scores.groupby('candidate')[['WAPE','low_MAPE','high_MAPE']].mean()
    ranks['selection_score']=ranks.WAPE+.25*(ranks.low_MAPE+ranks.high_MAPE)
    ranks.sort_values('selection_score').to_csv(OUT/'model_ranking.csv')
    finalists=ranks.nsmallest(4,'selection_score').index.tolist()
    configs=[('raw',1.,'global',0.)]
    for group in ['global','subtype','region_subtype']:
        for growth in [0.,.0175,.035]:
            for method,strengths in [('linear',[1.1,1.25,1.5]),('power',[1.1,1.25,1.5]),('quantile',[.25,.5,.75,1.]),('mean_anchor',[.5,1.])]:
                configs += [(method,s,group,growth) for s in strengths]
    prows=[]
    for ident in finalists:
        for config in configs:
            mets=[]
            for yr,tr,va,p in cache[ident]: mets.append(metrics(va.y,postprocess(p,va,tr,yr,*config)))
            avg={k:np.mean([x[k] for x in mets]) for k in ['WAPE','MAPE','low_MAPE','high_MAPE','MAE','bias']}
            prows.append(dict(candidate=ident,method=config[0],strength=config[1],grouping=config[2],growth=config[3],selection_score=avg['WAPE']+.25*(avg['low_MAPE']+avg['high_MAPE']),**avg))
    post=pd.DataFrame(prows).sort_values('selection_score'); post.to_csv(OUT/'postprocess_validation.csv',index=False)
    best=post.iloc[0]; ident=best.candidate; fs,kind=ident.split('__'); cols=sets[fs]
    config=(best.method,float(best.strength),best.grouping,float(best.growth))
    tr=labeled[labeled.year<2026]; test=labeled[labeled.year==2026]
    model=fit_model(kind,tr,cols); raw=predict(model,kind,test,cols)
    adjusted=postprocess(raw,test,tr,2026,*config)
    base=fit_model('cat_baseline_log',tr,allcols); bp=predict(base,'cat_baseline_log',test,allcols)
    tests=pd.DataFrame([dict(model=name,**metrics(test.y,p)) for name,p in [('notebook_style_baseline',bp),('selected_raw',raw),('selected_adjusted',adjusted)]])
    tests.to_csv(OUT/'test_2026_metrics.csv',index=False)
    tp=test[['location_num','location_name','open_date','anniversary','subtype','region','y']].copy()
    tp=tp.rename(columns={'y':'actual_annualized_sales'})
    tp['baseline']=bp; tp['selected_raw']=raw; tp['selected_adjusted']=adjusted
    tp.to_csv(OUT/'test_2026_predictions.csv',index=False)
    # Diagnostics only: these 2026 comparisons never feed model selection.
    diag=[]
    for cfg in configs:
        diag.append(dict(method=cfg[0],strength=cfg[1],grouping=cfg[2],growth=cfg[3],**metrics(test.y,postprocess(raw,test,tr,2026,*cfg))))
    pd.DataFrame(diag).to_csv(OUT/'test_2026_postprocess_diagnostics_NOT_FOR_SELECTION.csv',index=False)
    seg=[]
    for group in ['region','subtype']:
        for value,b in test.groupby(group):
            mask=test.index.get_indexer(b.index)
            if len(b)>=2: seg.append(dict(group=group,value=value,**metrics(b.y,adjusted[mask])))
    pd.DataFrame(seg).to_csv(OUT/'test_2026_segments.csv',index=False)
    final=fit_model(kind,labeled,cols)
    inference=d.loc[~eligible].copy(); inference['forecast_year']=inference.year.clip(lower=2026).fillna(2026).astype(int)
    inference['raw_annualized_sales']=predict(final,kind,inference,cols)
    inference['forecast_annualized_sales']=np.nan
    for yr,b in inference.groupby('forecast_year'):
        inference.loc[b.index,'forecast_annualized_sales']=postprocess(b.raw_annualized_sales,b,labeled,yr,*config)
    inference['reason']=np.where(~mature.loc[inference.index],'not_yet_one_year_old','missing_or_ineligible_label')
    inference[['location_num','location_name','location_type','open_date','anniversary','subtype','region','forecast_year','reason','raw_annualized_sales','forecast_annualized_sales']].to_csv(OUT/'inference_forecasts.csv',index=False)
    with open(OUT/'model.pkl','wb') as f: pickle.dump(dict(model=final,kind=kind,features=cols,config=config,reference=labeled[['year','y','region','subtype']],as_of=ASOF),f)
    if hasattr(final,'feature_importances_'):
        pd.Series(final.feature_importances_,index=cols).sort_values(ascending=False).rename('importance').to_csv(OUT/'feature_importance.csv')
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for name,p in [('Baseline',bp),('Selected raw',raw),('Selected adjusted',adjusted)]:
        axes[0].scatter(test.y/1e6,p/1e6,s=15,alpha=.5,label=name)
        q=np.linspace(.01,.99,99); axes[1].plot(q,np.quantile(p,q)/1e6,label=name)
    lim=max(test.y.max(),adjusted.max())/1e6
    axes[0].plot([0,lim],[0,lim],'k--'); axes[0].set(xlabel='Actual ($M)',ylabel='Forecast ($M)',title='Untouched 2026 holdout')
    axes[1].plot(q,np.quantile(test.y,q)/1e6,'k--',label='Actual'); axes[1].set(xlabel='Quantile',ylabel='Annualized sales ($M)',title='Distribution and tails'); axes[1].legend(fontsize=8)
    bins=pd.qcut(test.y,5,labels=False)
    for name,p in [('Baseline',bp),('Selected',adjusted)]: axes[2].plot(range(1,6),[np.mean((p-test.y.to_numpy())[bins==i])/1e6 for i in range(5)],marker='o',label=name)
    axes[2].axhline(0,color='k',ls='--'); axes[2].set(xlabel='Actual sales quintile (diagnostic)',ylabel='Mean forecast error ($M)',title='Tail bias'); axes[2].legend()
    fig.tight_layout(); fig.savefig(OUT/'test_2026_diagnostics.png',dpi=160); plt.close(fig)
    summary={'selected':best.to_dict(),'test_metrics':tests.to_dict(orient='records'),'inference_rows':len(inference)}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()

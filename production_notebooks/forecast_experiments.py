"""Time-aware, subtype/tail-aware experiments for opening-cohort average sales.

Run with Python after installing requirements-forecast.txt. Features must be
available at opening; the supplied files cannot establish their vintage.
"""
from pathlib import Path
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from sales_forecast_config import FEATURE_COLUMNS
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor

TARGET = 'avg_sales'
DATE = 'open_date'

def load_data(path='90_days_features.parquet'):
    d = pd.read_parquet(path).copy()
    d[DATE] = pd.to_datetime(d[DATE])
    return d.sort_values(DATE).reset_index(drop=True)

def features(d, extended=False):
    # Reuse the original explicit allow-list rather than automatically including
    # potentially post-opening sales, status, event or anniversary fields.
    cols = list(FEATURE_COLUMNS)
    if extended:
        cols += ['latitude', 'longitude', 'close_highway_min', 'close_ramp_1',
                 'close_ramp_2', 'interstate_within_2min']
    x = d[[c for c in cols if c in d]].copy()
    x['year'] = d[DATE].dt.year
    x['month'] = d[DATE].dt.month
    x['month_sin'] = np.sin(2*np.pi*x.month/12)
    x['month_cos'] = np.cos(2*np.pi*x.month/12)
    x['time_index'] = (x.year-2017)*12+x.month
    return x

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return dict(count=len(y), MAE=mean_absolute_error(y,p),
                RMSE=np.sqrt(mean_squared_error(y,p)),
                WAPE=100*np.abs(y-p).sum()/max(np.abs(y).sum(),1),
                MAPE=100*np.mean(np.abs(y[y>0]-p[y>0])/y[y>0]) if (y>0).any() else np.nan,
                bias=float(np.mean(p-y)), R2=r2_score(y,p) if len(y)>1 else np.nan)

def thresholds(train):
    return {s: g[TARGET].quantile([.2,.8]).values for s,g in train.groupby('subtype')}

def report(meta, pred, cuts):
    rows = []
    for s in ['ALL'] + sorted(meta.subtype.unique()):
        mask = np.ones(len(meta),dtype=bool) if s=='ALL' else (meta.subtype==s).to_numpy()
        tails = np.array(['low' if y<cuts.get(t,[-np.inf,np.inf])[0] else
                          'high' if y>cuts.get(t,[-np.inf,np.inf])[1] else 'middle'
                          for t,y in zip(meta.subtype,meta[TARGET])])
        for tail in ['all','low','middle','high']:
            m = mask & ((tails==tail) if tail!='all' else True)
            if m.any(): rows.append(dict(subtype=s,tail=tail,**metrics(meta[TARGET].to_numpy()[m],np.asarray(pred)[m])))
    return pd.DataFrame(rows)

def weights(train, mode):
    w = np.ones(len(train))
    if 'balanced' in mode:
        counts = train.subtype.value_counts()
        w *= np.sqrt(len(train)/(len(counts)*train.subtype.map(counts).to_numpy()))
    if 'tail' in mode:
        cuts = thresholds(train)
        w *= np.array([2.5 if y<cuts[s][0] or y>cuts[s][1] else 1 for s,y in zip(train.subtype,train[TARGET])])
    if 'recent' in mode:
        w *= np.exp(-np.log(2)*(train[DATE].max()-train[DATE]).dt.days.to_numpy()/730)
    return w/w.mean()

def fit_predict(train, valid, cfg):
    if cfg['family']=='median':
        if cfg.get('window_days'):
            train=train[train[DATE]>=train[DATE].max()-pd.Timedelta(days=cfg['window_days'])]
        med = train.groupby('subtype')[TARGET].agg(cfg.get('aggregate','median'))
        return valid.subtype.map(med).fillna(train[TARGET].median()).to_numpy()
    a,b = features(train,cfg.get('extended',False)),features(valid,cfg.get('extended',False))
    if cfg.get('time_only'):
        keep=['subtype','year','month_sin','month_cos']
        a,b=a[keep],b[keep]
    cat = [c for c in a if a[c].dtype=='object' or c in ['subtype','state','region','density_classification_1','ssam_flag','no_dt_flag','inline_flag']]
    y = train[TARGET].to_numpy()
    if cfg.get('log'): y=np.log1p(y)
    w=weights(train,cfg.get('weight',''))
    if cfg['family'] in ['ridge','extra']:
        for c in cat:
            a[c]=a[c].fillna('missing').astype(str); b[c]=b[c].fillna('missing').astype(str)
        numeric=[c for c in a if c not in cat]
        prep=ColumnTransformer([
            ('numeric',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),numeric),
            ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
        aa=prep.fit_transform(a);bb=prep.transform(b)
        if cfg['family']=='ridge':model=Ridge(alpha=cfg.get('alpha',100))
        else:model=ExtraTreesRegressor(n_estimators=400,min_samples_leaf=cfg.get('min_child',10),max_features=.8,random_state=42,n_jobs=1)
        model.fit(aa,y,sample_weight=w)
        p=model.predict(bb)
        return np.maximum(np.expm1(p) if cfg.get('log') else p,0)
    if cfg['family']=='cat':
        for c in cat:
            a[c]=a[c].fillna('missing').astype(str); b[c]=b[c].fillna('missing').astype(str)
        for c in a.columns.difference(cat):
            a[c]=a[c].replace([np.inf,-np.inf],np.nan).fillna(0)
            b[c]=b[c].replace([np.inf,-np.inf],np.nan).fillna(0)
        model=CatBoostRegressor(iterations=cfg.get('trees',600),depth=cfg.get('depth',5),
             learning_rate=.035,loss_function=cfg.get('loss','RMSE'),l2_leaf_reg=8,
             random_seed=42,thread_count=1,verbose=False,allow_writing_files=False)
        model.fit(a,y,cat_features=cat,sample_weight=w)
    else:
        enc=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
        a[cat]=enc.fit_transform(a[cat].astype(str)); b[cat]=enc.transform(b[cat].astype(str))
        model=lgb.LGBMRegressor(objective=cfg.get('loss','regression'),
            n_estimators=cfg.get('trees',400),learning_rate=.025,num_leaves=cfg.get('leaves',15),
            max_depth=5,min_child_samples=cfg.get('min_child',20),reg_lambda=5,
            colsample_bytree=.8,subsample=.85,subsample_freq=1,
            random_state=42,n_jobs=1,verbosity=-1)
        model.fit(a,y,sample_weight=w)
    p=model.predict(b)
    return np.maximum(np.expm1(p) if cfg.get('log') else p,0)

def candidates():
    configs=[dict(name='subtype_median',family='median')]
    for window in [365,730,1095]:
        configs.append(dict(name=f'subtype_median_recent_{window}',family='median',window_days=window))
    configs.append(dict(name='subtype_mean_recent_730',family='median',window_days=730,aggregate='mean'))
    for log in [False,True]:
        configs.append(dict(name=f'subtype_time_trend_{"log" if log else "raw"}',family='ridge',time_only=True,log=log,alpha=20))
    for family,losses in [('lgb',['regression','regression_l1']),('cat',['RMSE','MAE'])]:
        for loss in losses:
            for log in [False,True]:
                for weight in ['', 'tail_balanced','recent_balanced']:
                    configs.append(dict(name=f'{family}_{loss}_{"log" if log else "raw"}_{weight or "uniform"}',family=family,loss=loss,log=log,weight=weight))
    configs += [dict(name='cat_extended_tail',family='cat',loss='RMSE',weight='tail_balanced',extended=True),
                dict(name='lgb_small_tail',family='lgb',loss='regression',weight='tail_balanced',leaves=7,min_child=35)]
    for alpha in [10,100,1000]:
        for log in [False,True]:
            configs.append(dict(name=f'ridge_{alpha}_{"log" if log else "raw"}',family='ridge',alpha=alpha,log=log))
    for log in [False,True]:
        for weight in ['', 'tail_balanced']:
            configs.append(dict(name=f'extra_{"log" if log else "raw"}_{weight or "uniform"}',family='extra',log=log,weight=weight))
    return configs

def routed_predict(train, valid, cfg, specialist=False):
    p=fit_predict(train,valid,cfg)
    if specialist:
        for s in valid.subtype.unique():
            a=train[train.subtype==s]; mask=(valid.subtype==s).to_numpy()
            if len(a)>=40:
                q=fit_predict(a,valid.loc[mask],cfg)
                # Shrink sparse specialists toward the pooled model.
                alpha=len(a)/(len(a)+100)
                p[mask]=(1-alpha)*p[mask]+alpha*q
    return p

class SalesForecastWrapper:
    """Reusable selected routing for opening-time rows (target not required).

    Uses deterministic refitting on each predict call; suited to this small
    dataset. Persist training snapshots/configuration for reproducibility.
    """
    def __init__(self, configurations, routes):
        self.configurations = configurations
        self.routes = routes

    def fit(self, training_data):
        self.training_data = training_data.copy()
        self.training_data[DATE] = pd.to_datetime(self.training_data[DATE])
        if self.training_data[TARGET].isna().any() or (self.training_data[TARGET]<0).any():
            raise ValueError('Training targets must be observed and nonnegative')
        return self

    def predict(self, future_locations):
        if not hasattr(self, 'training_data'): raise RuntimeError('Call fit first')
        future_locations = future_locations.copy()
        future_locations[DATE] = pd.to_datetime(future_locations[DATE])
        unknown=set(future_locations.subtype)-set(self.routes)
        if unknown: raise ValueError(f'No configured route for subtypes: {unknown}')
        for name in set(self.routes.values()):
            cfg=self.configurations[name]
            if cfg['family']=='median':continue
            expected=set(features(self.training_data,cfg.get('extended',False)).columns)
            actual=set(features(future_locations,cfg.get('extended',False)).columns)
            if cfg.get('time_only'):expected={'subtype','year','month_sin','month_cos'}
            if expected-actual:raise ValueError(f'Missing forecast features: {sorted(expected-actual)}')
        p=np.zeros(len(future_locations))
        for s,name in self.routes.items():
            m=(future_locations.subtype==s).to_numpy()
            if m.any():
                cfg=self.configurations[name]
                p[m]=routed_predict(self.training_data,future_locations.loc[m],cfg,cfg.get('specialist',False))
        return p

    @classmethod
    def from_results(cls, directory='forecast_results'):
        settings=json.loads((Path(directory)/'selected_configuration.json').read_text())
        return cls(settings['configurations'],settings['routes'])

def run_experiments(output='forecast_results', horizon_days=90, configurations=None,
                    validation_periods=None, test_start='2025-07-01', max_workers=4,
                    tail_importance=1.0):
    if not 0<=tail_importance<=1:raise ValueError('tail_importance must be between zero and one')
    out=Path(output); out.mkdir(exist_ok=True)
    folds=validation_periods or [('2023-01-01','2023-07-01'),('2024-01-01','2024-07-01'),('2024-07-01','2025-01-01')]
    signature=dict(data_sha256=hashlib.sha256(Path('90_days_features.parquet').read_bytes()).hexdigest(),
                   horizon_days=horizon_days,validation_periods=[list(f) for f in folds],test_start=str(test_start),version=1)
    manifest=out/'run_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=signature:
        raise ValueError('Cached results belong to a different dataset/design; choose a new output directory')
    manifest.write_text(json.dumps(signature,indent=2))
    d=load_data()
    audit={'rows':len(d),'duplicate_locations':int(d.location_num.duplicated().sum()),
           'date_min':str(d[DATE].min()),'date_max':str(d[DATE].max()),
           'missing_target':int(d[TARGET].isna().sum()),'negative_target':int((d[TARGET]<0).sum())}
    if d[TARGET].isna().any() or (d[TARGET]<0).any(): raise ValueError('Invalid targets: inspect audit')
    # Fixed untouched test; calibration labels must mature before test start.
    test_start=pd.Timestamp(test_start)
    if any(pd.Timestamp(end)>test_start-pd.Timedelta(days=horizon_days) for _,end in folds):
        raise ValueError('Validation labels must mature before the fixed test starts')
    pre=d[d[DATE]<test_start-pd.Timedelta(days=horizon_days)]
    test=d[d[DATE]>=test_start]
    records=[]; predictions={}; configs=configurations if configurations is not None else candidates()
    # Specialist variants use the same validation folds as pooled alternatives.
    if configurations is None:
        configs += [dict(c,name=c['name']+'_specialist',specialist=True) for c in configs if c['family']=='cat' and c.get('weight') in ['', 'tail_balanced'] and not c.get('log')]
    if len({c['name'] for c in configs})!=len(configs):raise ValueError('Configuration names must be unique')
    def evaluate_config(cfg):
        metrics_path=out/(cfg['name']+'_validation.csv')
        predictions_path=out/(cfg['name']+'_validation_predictions.csv')
        config_path=out/(cfg['name']+'_config.json')
        if config_path.exists() and json.loads(config_path.read_text())!=cfg:
            raise ValueError(f'Changed configuration {cfg["name"]}; choose a new name or output directory')
        if metrics_path.exists() and predictions_path.exists():
            config_path.write_text(json.dumps(cfg,indent=2))
            return cfg['name'],pd.read_csv(metrics_path),pd.read_csv(predictions_path)
        collected=[]
        fold_reports=[]
        for start,end in folds:
            start,end=pd.Timestamp(start),pd.Timestamp(end)
            tr=d[d[DATE]<start-pd.Timedelta(days=horizon_days)]
            va=d[(d[DATE]>=start)&(d[DATE]<end)]
            if not len(va):continue
            p=routed_predict(tr,va,cfg,cfg.get('specialist',False))
            r=report(va,p,thresholds(tr)); r['fold']=str(start.date());r['model']=cfg['name'];fold_reports.append(r)
            q=va[['location_num','subtype',DATE,TARGET]].copy();q['prediction']=p; collected.append(q)
        candidate_predictions=pd.concat(collected)
        candidate_reports=pd.concat(fold_reports)
        candidate_reports.to_csv(out/(cfg['name']+'_validation.csv'),index=False)
        candidate_predictions.to_csv(out/(cfg['name']+'_validation_predictions.csv'),index=False)
        config_path.write_text(json.dumps(cfg,indent=2))
        print('completed',cfg['name'],flush=True)
        return cfg['name'], candidate_reports, candidate_predictions
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for name,r,q in pool.map(evaluate_config,configs):
            records.append(r);predictions[name]=q
    detail=pd.concat(records);detail.to_csv(out/'validation_metrics.csv',index=False)
    # Equal importance to each subtype and each tail; do not let FSU dominate.
    band_scores=detail[(detail.subtype!='ALL')&(detail['tail']!='all')].groupby(['model','subtype','tail']).WAPE.mean().groupby(['model','subtype']).mean()
    overall_scores=detail[(detail.subtype!='ALL')&(detail['tail']=='all')].groupby(['model','subtype']).WAPE.mean()
    selection_scores=tail_importance*band_scores+(1-tail_importance)*overall_scores
    scores=selection_scores.groupby('model').mean().sort_values()
    scores.rename('selection_WAPE').to_csv(out/'leaderboard.csv')
    routes={}
    for s in sorted(d.subtype.unique()):
        tab=selection_scores.xs(s,level='subtype')
        # Sparse subtype selection stays pooled; test results never choose routes.
        if sum((predictions[configs[0]['name']].subtype==s))<20:
            routes[s]=scores.index[0]
        else: routes[s]=tab.idxmin()
    cfgmap={c['name']:c for c in configs}
    (out/'selected_configuration.json').write_text(json.dumps(dict(configurations=cfgmap,routes=routes),indent=2))
    p=SalesForecastWrapper(cfgmap,routes).fit(pre).predict(test)
    result=test[['location_num','location_name','subtype',DATE,TARGET]].copy();result['prediction']=p
    # Intervals from rolling out-of-time residuals; empirical coverage is reported,
    # not a claim of exchangeability under temporal distribution shift.
    global_res=np.concatenate([np.abs(q[TARGET]-q.prediction)/np.maximum(q.prediction,1) for name,q in predictions.items() if name==scores.index[0]])
    for s,name in routes.items():
        q=predictions[name];q=q[q.subtype==s]
        res=np.abs(q[TARGET]-q.prediction)/np.maximum(q.prediction,1)
        if len(res)<30:res=global_res
        level=min(1,np.ceil((len(res)+1)*.9)/len(res))
        radius=float(np.quantile(res,level,method='higher'))
        m=result.subtype==s
        result.loc[m,'lower_90']=(result.loc[m,'prediction']*(1-radius)).clip(lower=0)
        result.loc[m,'upper_90']=result.loc[m,'prediction']*(1+radius)
    result['covered_90']=(result[TARGET]>=result.lower_90)&(result[TARGET]<=result.upper_90)
    result.to_csv(out/'test_predictions.csv',index=False)
    final=report(test,p,thresholds(pre));final.to_csv(out/'test_metrics.csv',index=False)
    original_path=out/'original_predictions.csv'
    if original_path.exists():
        original=pd.read_csv(original_path,dtype={'location_num':str})
        if original.location_num.tolist()==test.location_num.astype(str).tolist():
            baseline=report(original,original.prediction.to_numpy(),thresholds(pre))
            comparison=baseline.merge(final,on=['subtype','tail'],suffixes=('_original','_selected'))
            comparison['WAPE_change_points']=comparison.WAPE_selected-comparison.WAPE_original
            comparison.to_csv(out/'comparison.csv',index=False)
    audit.update(routes=routes,horizon_days=horizon_days,interval_coverage=float(result.covered_90.mean()),
                 mean_interval_width=float((result.upper_90-result.lower_90).mean()),training_rows=len(pre),test_rows=len(test),tail_importance=tail_importance)
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    print(final[final['tail']=='all'].to_string(index=False));print(json.dumps(audit,indent=2))
    return detail,final,result

if __name__=='__main__':run_experiments()

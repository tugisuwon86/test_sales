"""Opening-time feature engineering and fitted models for the second search.

No actual sales of the forecast location are used as predictors. All learned
preprocessing/selection is fitted on the training cohort supplied to fit().
"""
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, LogisticRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from forecast_experiments import TARGET, DATE, thresholds
from sales_forecast_config import FEATURE_COLUMNS, LOCATION_ATTRIBUTE_COLUMNS

BLOCKED = {'avg_sales', 'avg_monthly_sales', 'avg_monthly_sales_log',
           'location_num', 'location_status', 'location_type', 'anniversary',
           'hex_string', 'Unnamed: 0', 'total_sales', 'trans_count', 'total_trans'}
CATEGORIES = {'subtype', 'state', 'region', 'dma', 'submarket',
              'density_classification_1', 'ssam_flag', 'no_dt_flag', 'inline_flag',
              'prototype','playground','is_relocated','has_outside_seating','has_cross_parking',
              'has_bypass_lane','has_drivethru_door'}
ORIGINAL = FEATURE_COLUMNS + LOCATION_ATTRIBUTE_COLUMNS
ACCESS = ['latitude', 'longitude', 'close_highway_min', 'close_ramp_1',
          'close_ramp_2', 'interstate_within_2min']

def engineered_features(data, group='rich'):
    d=data.copy(); dates=pd.to_datetime(d[DATE]); f=d[[c for c in ORIGINAL+ACCESS+['dma','submarket'] if c in d]].copy()
    def col(name):return pd.to_numeric(d[name],errors='coerce') if name in d else pd.Series(np.nan,index=d.index)
    def ratio(a,b):return a/b.where(b.abs()>1)
    f['year']=dates.dt.year; f['month']=dates.dt.month
    f['month_sin']=np.sin(2*np.pi*dates.dt.month/12); f['month_cos']=np.cos(2*np.pi*dates.dt.month/12)
    f['day_of_year_sin']=np.sin(2*np.pi*dates.dt.dayofyear/365.25)
    f['day_of_year_cos']=np.cos(2*np.pi*dates.dt.dayofyear/365.25)
    f['opening_weekday']=dates.dt.dayofweek
    names=d.location_name.fillna('').astype(str) if 'location_name' in d else pd.Series('',index=d.index)
    for label,pattern in {'name_no_dt':r'no[\s\-]*d[tT]|no[\s\-]*drive',
                          'name_inline':r'in[\s\-]*line', 'name_airport':r'airport|terminal',
                          'name_campus':r'university|college|campus',
                          'name_mall':r'\bmall\b|shopping center',
                          'name_highway':r'\bhwy\b|highway|interstate'}.items():
        f[label]=names.str.contains(pattern,case=False,regex=True).astype(int)
    mean=col('avg_monthly_sales_mean_1year'); closest=col('avg_monthly_sales_closest_1year')
    lower=col('avg_monthly_sales_25th_1year');upper=col('avg_monthly_sales_75th_1year')
    f['peer_closest_vs_mean']=ratio(closest,mean)
    f['peer_spread_relative']=ratio(upper-lower,mean)
    f['peer_lower_vs_mean']=ratio(lower,mean);f['peer_upper_vs_mean']=ratio(upper,mean)
    f['peer_mean_daily_26']=mean/26
    f['peer_closest_daily_26']=closest/26
    for label,base in [('closest',closest),('mean',mean),('25th',lower),('75th',upper)]:
        f['peer_growth_relative_'+label]=ratio(col('growth_rate_'+label),base)
    pop=col('total_population_1'); households=col('total_households_1')
    f['worker_share']=ratio(col('worker_population_1'),pop)
    f['daytime_resident_ratio']=ratio(col('daytime_population_1'),pop)
    f['people_per_household']=ratio(pop,households)
    f['disposable_income_share']=ratio(col('med_disposable_income_1'),col('med_household_income_1'))
    f['home_income_ratio']=ratio(col('med_home_value_1'),col('med_household_income_1'))
    f['spending_per_household']=ratio(col('food_away_from_home_1'),households)
    f['budget_per_person']=ratio(col('annual_budget_expenditure_1'),pop)
    f['competitors_per_1000']=1000*ratio(col('competitor_count_1'),pop)
    for radius in [10,15,20,25,30]:
        f[f'nearby_ring_total_{radius}']=sum(col(f'boundary_{r}') for r in [5,10,15,20,25,30] if r<=radius)
    for i in [1,2,3]:
        median=col(f'median_average_daily_traffic_{i}');maximum=col(f'max_average_daily_traffic_{i}')
        f[f'traffic_peak_ratio_{i}']=ratio(maximum,median)
    f['traffic_nearest_vs_outer']=ratio(col('median_average_daily_traffic_1'),col('median_average_daily_traffic_3'))
    f['retail_competitor_ratio']=ratio(col('bigbox_1')+1,col('competitor_1')+1)
    f['traffic_per_population']=ratio(col('median_average_daily_traffic_1'),pop)
    f['peer_sales_per_population']=ratio(mean,pop)
    f['peer_income_interaction']=mean*col('med_household_income_1')/1e5
    # Cannibalization and regional-pricing proxies: nearby-store productivity
    # relative to the number of stores in each trade-area ring.
    for radius in [5,10,15,20,25,30]:
        f[f'peer_sales_per_store_{radius}']=ratio(mean,col(f'boundary_{radius}')+1)
        f[f'closest_sales_per_store_{radius}']=ratio(closest,col(f'boundary_{radius}')+1)
    # Physical capacity and format metadata. Missingness flags are explicit
    # because some fields were introduced later in the historical system.
    for c in LOCATION_ATTRIBUTE_COLUMNS:
        if c in f: f[c+'_missing']=f[c].isna().astype(int)
    f['site_to_building_ratio']=ratio(col('total_site_square_footage'),col('building_square_footage'))
    f['occupancy_per_1000sqft']=1000*ratio(col('maximum_occupancy'),col('building_square_footage'))
    f['order_points_per_lane']=ratio(col('total_order_points'),col('total_drivethru_lanes')+1)
    # A value of 100000 is an unavailable-access sentinel in these columns.
    for c in ['close_highway_dist','close_highway_min']:
        if c in f:
            f[c+'_unavailable']=(f[c]>=99999).astype(int)
            f.loc[f[c]>=99999,c]=np.nan
    time=['year','month_sin','month_cos','day_of_year_sin','day_of_year_cos','opening_weekday']
    corecat=['subtype','state','region','density_classification_1','ssam_flag','no_dt_flag','inline_flag']
    peer=[c for c in f if c.startswith(('avg_monthly_sales_','growth_rate_','peer_'))]
    demographic=[c for c in f if any(w in c for w in ['population','income','household','home_value','spending','budget','worker_share','daytime_resident','people_per_household'])]
    access=[c for c in f if any(w in c for w in ['traffic','highway','ramp','interstate','bigbox','competitor','boundary','nearby_ring','months_'])]
    name=[c for c in f if c.startswith('name_')]
    attributes=[c for c in f if c in LOCATION_ATTRIBUTE_COLUMNS or c.endswith('_missing') or c in
                ['site_to_building_ratio','occupancy_per_1000sqft','order_points_per_lane']]
    geography=['latitude','longitude','dma','submarket']
    groups={'original':ORIGINAL+time,
            'rich':list(f),
            'core':corecat+time+peer+['latitude','longitude','median_average_daily_traffic_1','bigbox_2','boundary_5']+name,
            'peers':corecat+time+peer,
            'market':corecat+time+peer+geography+name,
            'demographics':corecat+time+demographic+name,
            'access':corecat+time+access+geography+name,
            'static':corecat+time+demographic+access+geography+name,
            'attributes':corecat+time+attributes+name,
            'static_attributes':corecat+time+demographic+access+geography+attributes+name,
            'no_time':[c for c in f if c not in time+['month']],
            'events_exploratory':list(f)+['num_events','total_predicthq_attendance']}
    if group=='events_exploratory':
        f['num_events']=col('num_events');f['total_predicthq_attendance']=col('total_predicthq_attendance')
    if group not in groups:raise ValueError(f'Unknown feature group {group}')
    columns=list(dict.fromkeys(c for c in groups[group] if c in f and c not in BLOCKED))
    return f[columns].replace([np.inf,-np.inf],np.nan)

class FeatureBuilder:
    def __init__(self,cfg):self.cfg=cfg
    def fit_transform(self,train,y):
        x=engineered_features(train,self.cfg.get('features','rich'))
        self.raw_columns=list(x.columns)
        self.cat=[c for c in x if c in CATEGORIES]
        self.numeric=[c for c in x if c not in self.cat]
        self.encoder=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
        if self.cat:x[self.cat]=self.encoder.fit_transform(x[self.cat].fillna('missing').astype(str))
        if self.cfg.get('numeric_log'):
            x[self.numeric]=np.sign(x[self.numeric])*np.log1p(x[self.numeric].abs())
        self.bounds={}
        if self.cfg.get('clip'):
            for c in self.numeric:
                self.bounds[c]=x[c].quantile([.01,.99]).to_numpy()
                x[c]=x[c].clip(*self.bounds[c])
        self.medians=x.median().fillna(0)
        # Selection is learned separately inside every training fold.
        keep=[c for c in x if x[c].nunique(dropna=False)>1]
        numeric=[c for c in self.numeric if c in keep]
        if self.cfg.get('deduplicate',True) and len(numeric)>1:
            corr=x[numeric].corr().abs(); dropped=set()
            for i,c in enumerate(numeric):
                if c in dropped:continue
                for c2 in numeric[i+1:]:
                    if corr.loc[c,c2]>.995:dropped.add(c2)
            keep=[c for c in keep if c not in dropped]
        selector=self.cfg.get('selector')
        if selector:
            if selector=='importance':
                ranker=lgb.LGBMRegressor(n_estimators=120,num_leaves=7,max_depth=3,
                   min_child_samples=25,reg_lambda=10,verbosity=-1,n_jobs=1,random_state=19)
                ranker.fit(x[keep],y,categorical_feature=[c for c in self.cat if c in keep])
                ranking=pd.Series(ranker.booster_.feature_importance(importance_type='gain'),index=keep).sort_values(ascending=False)
            elif selector=='correlation':
                ranking=x[keep].corrwith(pd.Series(y,index=x.index),method='spearman').abs().fillna(0).sort_values(ascending=False)
            else:raise ValueError(f'Unknown selector {selector}')
            mandatory=[c for c in ['subtype','year','inline_flag','name_no_dt'] if c in keep]
            keep=list(dict.fromkeys(mandatory+ranking.head(self.cfg.get('top_k',20)).index.tolist()))
        self.columns=keep
        self.cat=[c for c in self.cat if c in keep]
        return x[keep]
    def transform(self,data):
        x=engineered_features(data,self.cfg.get('features','rich'))
        missing=set(self.raw_columns)-set(x)
        if missing:raise ValueError(f'Missing engineered inputs: {sorted(missing)}')
        rawcat=[c for c in self.raw_columns if c in CATEGORIES]
        if rawcat:x[rawcat]=self.encoder.transform(x[rawcat].fillna('missing').astype(str))
        if self.cfg.get('numeric_log'):
            x[self.numeric]=np.sign(x[self.numeric])*np.log1p(x[self.numeric].abs())
        for c,bounds in self.bounds.items():x[c]=x[c].clip(*bounds)
        return x[self.columns]

def sample_weights(train,cfg):
    y=train[TARGET].to_numpy();w=np.ones(len(train)); cuts=thresholds(train)
    power=cfg.get('relative_power',0)
    if power:w*=np.power(np.median(y)/np.maximum(y,1),power)
    w*=np.array([cfg.get('low_weight',1) if v<cuts[s][0] else
                 cfg.get('high_weight',1) if v>cuts[s][1] else 1
                 for s,v in zip(train.subtype,y)])
    if cfg.get('half_life'):
        age=(train[DATE].max()-train[DATE]).dt.days.to_numpy()
        w*=np.exp(-np.log(2)*age/cfg['half_life'])
    if cfg.get('balance'):
        counts=train.subtype.value_counts()
        w*=np.sqrt(len(train)/(len(counts)*train.subtype.map(counts).to_numpy()))
    return np.clip(w/w.mean(),.05,20)

def dense_pipeline(x,cat):
    numeric=[c for c in x if c not in cat]
    return ColumnTransformer([
      ('numeric',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),numeric),
      ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])

class OpeningSalesModel:
    """Fitted estimator; predict never accesses the forecast row's target."""
    def __init__(self,cfg):self.cfg=cfg
    def anchor(self,d):
        c='avg_monthly_sales_mean_1year'
        p=pd.to_numeric(d[c],errors='coerce').fillna(self.peer_median).to_numpy()/26
        return np.maximum(p,100)
    def fit(self,train):
        train=train.copy();train[DATE]=pd.to_datetime(train[DATE])
        self.peer_median=float(train.avg_monthly_sales_mean_1year.median())
        transform=self.cfg.get('transform','raw');y=train[TARGET].to_numpy()/10000
        if transform=='log':y=np.log(np.maximum(train[TARGET].to_numpy(),1)/10000)
        elif transform=='peer_ratio':y=train[TARGET].to_numpy()/self.anchor(train)
        elif transform=='peer_log':y=np.log(np.maximum(train[TARGET].to_numpy(),1)/self.anchor(train))
        self.builder=FeatureBuilder(self.cfg);x=self.builder.fit_transform(train,y);cat=self.builder.cat
        w=sample_weights(train,self.cfg);family=self.cfg.get('family','lgb')
        if family=='lgb':
            self.model=lgb.LGBMRegressor(objective=self.cfg.get('loss','regression_l1'),
               alpha=self.cfg.get('alpha',.5),n_estimators=self.cfg.get('trees',400),
               learning_rate=self.cfg.get('lr',.035),num_leaves=self.cfg.get('leaves',15),
               max_depth=self.cfg.get('depth',5),min_child_samples=self.cfg.get('min_child',20),
               reg_alpha=self.cfg.get('reg_alpha',0),reg_lambda=self.cfg.get('reg_lambda',5),
               colsample_bytree=self.cfg.get('colsample',.85),subsample=.85,subsample_freq=1,
               min_data_per_group=10,cat_smooth=20,cat_l2=10,
               random_state=self.cfg.get('seed',42),verbosity=-1,n_jobs=1,
               linear_tree=self.cfg.get('linear_tree',False))
            self.model.fit(x,y,sample_weight=w,categorical_feature=cat if self.cfg.get('native_cat',True) else [])
        elif family=='cat':
            for c in cat:x[c]=x[c].astype(int).astype(str)
            self.model=CatBoostRegressor(iterations=self.cfg.get('trees',250),
                 depth=self.cfg.get('depth',4),learning_rate=self.cfg.get('lr',.04),
                 loss_function=self.cfg.get('loss','MAE'),l2_leaf_reg=self.cfg.get('reg_lambda',8),
                 random_seed=42,thread_count=1,verbose=False,allow_writing_files=False)
            self.model.fit(x.fillna(0),y,cat_features=cat,sample_weight=w)
        else:
            self.prep=dense_pipeline(x,cat);a=self.prep.fit_transform(x)
            if family=='ridge':self.model=Ridge(alpha=self.cfg.get('alpha',100))
            elif family=='elastic':self.model=ElasticNet(alpha=self.cfg.get('alpha',.01),l1_ratio=self.cfg.get('l1_ratio',.5),max_iter=5000)
            elif family=='huber':self.model=HuberRegressor(alpha=self.cfg.get('alpha',1),epsilon=1.35,max_iter=400)
            elif family=='svr':self.model=SVR(C=self.cfg.get('C',3),epsilon=self.cfg.get('epsilon',.15),gamma=self.cfg.get('gamma','scale'))
            elif family=='knn':self.model=KNeighborsRegressor(n_neighbors=self.cfg.get('neighbors',30),weights='distance')
            elif family in ['extra','forest']:
                klass=ExtraTreesRegressor if family=='extra' else RandomForestRegressor
                self.model=klass(n_estimators=250,min_samples_leaf=self.cfg.get('min_child',8),
                  max_depth=self.cfg.get('depth',None),max_features=.7,random_state=42,n_jobs=1)
            elif family=='hist':self.model=HistGradientBoostingRegressor(loss=self.cfg.get('loss','absolute_error'),max_iter=250,
                   max_leaf_nodes=15,max_depth=4,min_samples_leaf=25,l2_regularization=10,early_stopping=False,random_state=42)
            elif family=='mlp':self.model=MLPRegressor(hidden_layer_sizes=(32,16),alpha=self.cfg.get('alpha',10),
                   solver='lbfgs',max_iter=400,random_state=42)
            else:raise ValueError(f'Unknown family {family}')
            if family=='knn':self.model.fit(a,y)
            else:self.model.fit(a,y,sample_weight=w)
        return self
    def predict(self,data):
        x=self.builder.transform(data);family=self.cfg.get('family','lgb')
        if family=='cat':
            for c in self.builder.cat:x[c]=x[c].astype(int).astype(str)
            p=self.model.predict(x.fillna(0))
        elif family=='lgb':p=self.model.predict(x)
        else:p=self.model.predict(self.prep.transform(x))
        transform=self.cfg.get('transform','raw')
        if transform=='log':p=np.exp(np.clip(p,-8,6))*10000
        elif transform=='peer_ratio':p=p*self.anchor(data)
        elif transform=='peer_log':p=np.exp(np.clip(p,-8,6))*self.anchor(data)
        else:p=p*10000
        return np.maximum(p,0)

class TailMixtureModel:
    """Opening features predict tail membership; actual tail is training-only."""
    def __init__(self,cfg):self.cfg=cfg
    def fit(self,train):
        cuts=thresholds(train)
        labels=np.array([0 if y<cuts[s][0] else 2 if y>cuts[s][1] else 1 for s,y in zip(train.subtype,train[TARGET])])
        self.builder=FeatureBuilder(self.cfg);x=self.builder.fit_transform(train,train[TARGET].to_numpy()/10000)
        w=sample_weights(train,{k:v for k,v in self.cfg.items() if k not in ['low_weight','high_weight']})
        if self.cfg.get('gate','lgb')=='logistic':
            self.prep=dense_pipeline(x,self.builder.cat);a=self.prep.fit_transform(x)
            self.gate=LogisticRegression(C=self.cfg.get('C',.1),max_iter=1000,
              class_weight='balanced' if self.cfg.get('balanced_gate') else None)
            self.gate.fit(a,labels,sample_weight=w)
        else:
            self.gate=lgb.LGBMClassifier(n_estimators=220,num_leaves=7,max_depth=3,
               min_child_samples=30,learning_rate=.035,reg_lambda=20,verbosity=-1,n_jobs=1,
               random_state=42,class_weight='balanced' if self.cfg.get('balanced_gate') else None)
            self.gate.fit(x,labels,categorical_feature=self.builder.cat,sample_weight=w)
        self.class_counts=np.bincount(labels,minlength=3)
        self.experts=[]
        for label in range(3):
            a=train.iloc[np.where(labels==label)[0]]
            cfg=dict(self.cfg,family='lgb',loss='regression_l1',transform='log',trees=200,
                     leaves=7,min_child=20,reg_lambda=10,low_weight=1,high_weight=1)
            self.experts.append(OpeningSalesModel(cfg).fit(a))
        return self
    def probabilities(self,data):
        x=self.builder.transform(data)
        if self.cfg.get('gate','lgb')=='logistic':x=self.prep.transform(x)
        p=self.gate.predict_proba(x)
        if self.cfg.get('balanced_gate') and self.cfg.get('correct_prior',True):
            p=p*self.class_counts/self.class_counts.sum()
        p=np.power(np.maximum(p,1e-6),1/self.cfg.get('temperature',1))
        return p/p.sum(axis=1,keepdims=True)
    def predict(self,data):
        p=self.probabilities(data);experts=np.column_stack([e.predict(data) for e in self.experts])
        if self.cfg.get('hard_gate'):return experts[np.arange(len(data)),p.argmax(axis=1)]
        return (p*experts).sum(axis=1)

def fit_model(train,cfg):
    klass=TailMixtureModel if cfg.get('family')=='mixture' else OpeningSalesModel
    return klass(cfg).fit(train)

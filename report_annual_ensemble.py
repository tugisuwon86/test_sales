"""Charts, segment tables and review notebook for annual_ensemble.py."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.annual_packages'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nbformat
from annual_ensemble import prepare,metric
OUT=ROOT/'annual_ensemble_results'
BLUE='#2878a5'; ORANGE='#db7c26'; GREEN='#269577'; GREY='#687987'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','axes.grid':True,'grid.alpha':.18})

def table(frame,percent=()):
    def fmt(k,x):
        if k in percent: return f'{x:.1%}'
        if isinstance(x,(float,np.floating)): return f'{x:,.2f}'
        return str(x)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(c,x) for c,x in zip(frame.columns,row))+' |' for row in frame.itertuples(index=False,name=None))

def ecdf(ax,values,label,color,ls='-'):
    v=np.sort(np.asarray(values)); ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=label,color=color,linestyle=ls,lw=2)

def box_year(ax,frame,ssam=None):
    if ssam is not None: frame=frame[frame.ssam_flag==ssam]
    years=sorted(frame.year.unique())
    data=[frame.loc[frame.year==year,'y'].to_numpy() for year in years]
    bp=ax.boxplot(data,positions=np.arange(len(years)),patch_artist=True,widths=.55,showfliers=True,flierprops={'markersize':3,'alpha':.5},medianprops={'color':'white','linewidth':1.8})
    for patch,year in zip(bp['boxes'],years): patch.set_facecolor(ORANGE if year==2026 else BLUE)
    ax.plot(np.arange(len(years)),[v.mean() for v in data],marker='D',ms=4,color=GREEN,lw=1,label='Mean')
    ax.set_xticks(np.arange(len(years)),[f'{int(year)}\nn={len(v)}' for year,v in zip(years,data)])
    ax.set_ylabel('Annualized sales ($M)'); ax.set_xlabel('Anniversary year')
    ax.grid(axis='x',visible=False)
    return years

def main():
    d,eligible,_=prepare(); labeled=d.loc[eligible]; tr=labeled[labeled.year<2026]; te=labeled[labeled.year==2026]
    s=json.loads((OUT/'summary.json').read_text()); sel=json.loads((OUT/'selection.json').read_text())
    p=pd.read_csv(OUT/'test_predictions_millions.csv',dtype={'location_num':str})
    metrics=pd.read_csv(OUT/'test_metrics.csv'); segments=pd.read_csv(OUT/'segment_metrics.csv')
    errors=p.copy(); errors['absolute_error_dollars']=abs(errors.prediction-errors.y)*1e6
    errors['absolute_percentage_error']=abs(errors.prediction-errors.y)/errors.y
    errors.sort_values('absolute_error_dollars',ascending=False).to_csv(OUT/'largest_test_errors.csv',index=False)
    # Distribution statistics retain observed data and model forecasts separately.
    distribution=[]
    for keys in [['year'],['year','ssam_flag'],['ssam_flag'],['subtype','ssam_flag']]:
        for key,g in labeled.groupby(keys):
            key=key if isinstance(key,tuple) else (key,)
            vals=g.y.to_numpy()
            distribution.append(dict(grouping='+'.join(keys),group=' | '.join(map(str,key)),n=len(g),mean=vals.mean()*1e6,std=np.std(vals,ddof=1)*1e6 if len(vals)>1 else np.nan,**{f'p{q}':np.quantile(vals,q/100)*1e6 for q in [0,10,25,50,75,90,100]}))
    pd.DataFrame(distribution).to_csv(OUT/'actual_sales_distribution_statistics.csv',index=False)
    fig,ax=plt.subplots(figsize=(13,5.8)); box_year(ax,labeled)
    ax.set_title('Observed annualized sales by anniversary year\nTraining cohorts: 2016–2025  |  Follow-up test: 2026 (mature stores only)',pad=13)
    ax.legend(handles=[Line2D([0],[0],color=BLUE,lw=8,label='Training actuals'),Line2D([0],[0],color=ORANGE,lw=8,label='2026 test actuals'),Line2D([0],[0],marker='D',color=GREEN,label='Mean')],loc='upper left')
    fig.text(.01,.01,'Boxes: 25th–75th percentiles; white line: median; dots: outliers. Annualized sales = 12 × average monthly sales. 2026 is a partial cohort.',fontsize=9,color=GREY)
    fig.tight_layout(rect=[0,.035,1,1]); fig.savefig(OUT/'sales_distribution_by_year.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9),gridspec_kw={'height_ratios':[1.1,1]})
    for col,flag in enumerate([0,1]):
        box_year(axes[0,col],labeled,flag)
        axes[0,col].set_title(f'Observed distribution by year: SSAM={flag}')
        axes[0,col].set_ylim(0,labeled.y.max()*1.05)
        axes[0,col].tick_params(axis='x',labelsize=8)
        train=tr[tr.ssam_flag==flag]; test=p[p.ssam_flag==flag]
        ecdf(axes[1,col],train.y,f'Train actual (n={len(train)})',BLUE)
        ecdf(axes[1,col],test.y,f'2026 actual (n={len(test)})',ORANGE)
        ecdf(axes[1,col],test.prediction,'2026 forecast',GREEN,'--')
        axes[1,col].set(xlabel='Annualized sales ($M)',ylabel='Cumulative share',title=f'SSAM={flag}: training / test / forecast')
        axes[1,col].set_xlim(0,labeled.y.max()*1.05)
        axes[1,col].legend(fontsize=9,loc='lower right')
    fig.suptitle('Sales distributions by ssam_flag',fontsize=17,fontweight='bold')
    fig.text(.01,.012,'SSAM=1 has no eligible stores in 2025 and only 2 in 2026. Its test error and distribution are highly uncertain. Years without observations are omitted.',fontsize=9,color=GREY)
    fig.tight_layout(rect=[0,.04,1,.96]); fig.savefig(OUT/'sales_distribution_by_ssam.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    ecdf(axes[0],p.y,'Actual',ORANGE); ecdf(axes[0],p.baseline,'Baseline',BLUE); ecdf(axes[0],p.prediction,'New ensemble',GREEN)
    axes[0].set(title='2026 sales distributions',xlabel='Annualized sales ($M)',ylabel='Cumulative share'); axes[0].legend(fontsize=9)
    axes[1].scatter(p.y,p.baseline,label='Baseline',color=BLUE,alpha=.5,s=20)
    axes[1].scatter(p.y,p.prediction,label='New ensemble',color=GREEN,alpha=.6,s=20)
    lim=max(p.y.max(),p.prediction.max())*1.04
    axes[1].plot([0,lim],[0,lim],'--',color=GREY); axes[1].set(xlabel='Actual ($M)',ylabel='Forecast ($M)',title='2026 actual vs forecast'); axes[1].legend(fontsize=9)
    q=pd.qcut(p.y,5,labels=False)
    for name,column,color in [('Baseline','baseline',BLUE),('New ensemble','prediction',GREEN)]:
        vals=[metric(p.loc[q==i,'y'],p.loc[q==i,column])['WAPE']*100 for i in range(5)]
        axes[2].plot(range(1,6),vals,'o-',label=name,color=color)
    axes[2].set(xlabel='Actual-sales quintile (diagnostic only)',ylabel='WAPE (%)',title='2026 error by sales level'); axes[2].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/'test_comparison.png',dpi=170); plt.close(fig)
    # Paired uncertainty interval; resample stores without retuning the selection.
    rng=np.random.default_rng(42); delta=[]
    for _ in range(5000):
        ix=rng.integers(0,len(p),len(p)); b=p.iloc[ix]
        delta.append(metric(b.y,b.prediction)['WAPE']-metric(b.y,b.baseline)['WAPE'])
    ci=np.quantile(delta,[.025,.5,.975]); (OUT/'bootstrap.json').write_text(json.dumps({'WAPE_difference_new_minus_baseline_95pct':ci.tolist(),'resamples':5000},indent=2))
    mt=metrics[['model','WAPE','MAPE','MAE_dollars']]
    testseg=segments[segments.split=='test_2026_followup'][['grouping','group','n','WAPE','MAPE']]
    oofseg=segments[(segments.split=='historical_OOF_raw_selected_blend') & segments.grouping.isin(['subtype','ssam_flag'])][['grouping','group','n','WAPE','MAPE']]
    fitseg=segments[(segments.split=='train_in_sample_optimistic') & segments.grouping.isin(['subtype','ssam_flag'])][['grouping','group','n','WAPE','MAPE']]
    # Comparison by group with baseline, not just new-model errors.
    compare=[]
    for column in ['subtype','ssam_flag']:
        for value,g in p.groupby(column):
            a=metric(g.y,g.baseline); b=metric(g.y,g.prediction)
            compare.append({'grouping':column,'group':str(value),'n':len(g),'baseline_WAPE':a['WAPE'],'new_WAPE':b['WAPE'],'baseline_MAPE':a['MAPE'],'new_MAPE':b['MAPE']})
    compare=pd.DataFrame(compare); compare.to_csv(OUT/'test_subtype_ssam_comparison.csv',index=False)
    baseline=metrics.iloc[0]; new=metrics.iloc[-1]
    raw=metrics.iloc[-2]
    conclusion=f"2026 WAPE changes from **{baseline.WAPE:.2%} to {new.WAPE:.2%}** ({(baseline.WAPE-new.WAPE)*100:.2f} percentage points); MAPE changes from **{baseline.MAPE:.2%} to {new.MAPE:.2%}**."
    if new.WAPE>=baseline.WAPE: conclusion+=' This run does not justify replacing the baseline on WAPE.'
    conclusion+=f' The unadjusted blend has {raw.WAPE:.2%} WAPE versus {baseline.WAPE:.2%} for the baseline; this tiny diagnostic difference is not evidence of a meaningful gain.'
    weights=pd.DataFrame([{'component':k,'weight':v} for k,v in s['weights'].items()])
    composition_note=('Validation selected a single CatBoost component (100% weight); the multi-model blends were tested but did not beat it in the 2025 selection cohort.' if len(weights)==1 else f'Validation selected a blend of {len(weights)} components.')
    chosen=s['selection']; devrank=pd.read_csv(OUT/'development_ranking.csv')
    report=f'''# WAPE-focused annual forecast experiments

{conclusion}

As of September 29, 2026. This is a **follow-up evaluation on the already-inspected 2026 cohort**, not a newly untouched test. The model-selection script never uses 2026 targets for weights, hyperparameters, or calibration. Earlier findings informed this experiment's design, so confirmation on a new mature cohort remains necessary.

## Results

{table(mt,percent=['WAPE','MAPE'])}

Paired bootstrap (5,000 store resamples): new-minus-baseline WAPE 95% interval **[{ci[0]*100:.2f}, {ci[2]*100:.2f}] percentage points**. Negative favors the new model. It excludes model-selection, cohort, and feature-provenance uncertainty.

## Selected ensemble and processing

{table(weights,percent=['weight'])}

{composition_note}

Selected postprocessing: **{chosen['method']}**, strength **{chosen['strength']}**. Portfolio mean ceiling: latest observed cohort mean compounded at **{chosen['growth']:.2%}**, never above 3.5% annual growth. The cap applies to the whole forecast-year batch; subgroups may differ because of mix. No individual-store growth cap is imposed.

The new objective is WAPE alone, rather than the previous combined WAPE/tail score. For fixed positive targets, minimizing WAPE is equivalent to minimizing absolute dollar error. Tested {len(sel['specifications'])} model/heuristic specifications with expanding-year predictions for 2022, 2023, 2024 and 2025. Development ranking and convex nonnegative ensemble weights use only 2022–2024 predictions. The top eight development candidates, plus the original notebook baseline and previous raw-model leader, feed individual-model, top-3/5/8 average, least-absolute-error blends with weight shrinkage, and 25/50/75% baseline-weighted pair blends. 2025 selects among {len(pd.read_csv(OUT/'selection_2025.csv'))} blend/calibration/growth combinations. The chosen weights are frozen; for 2026, calibration parameters are re-estimated from the available 2022–2025 out-of-fold residuals. The 2025 selection metrics are tuned, not unbiased estimates of final performance. The original-model components were added after the first follow-up pool failed to improve 2026; this is another reason the 2026 results require fresh-cohort confirmation.

Preprocessing includes fold-local missing-value imputation, missing indicators, grouping rare categories in sklearn models, cleaned highway-distance sentinels, population/worker/competitor ratios, log-scaled demographics, neighboring-sales dispersion, calendar features, and subtype/SSAM interactions. Models include CatBoost MAE/RMSE/Huber, histogram boosting with absolute/squared error, Extra Trees, ridge, robust Huber regression, neighboring-sales ratio models, and hierarchically pooled median/ratio heuristics. Training variants use the full history, recent 3–5 year windows, exponential recency weights, and capped 3.5% historical-target uplift. Models predict annual sales, log sales, or ratios to neighboring sales as appropriate. Postprocessing compares offsets, group-shrunk offsets, optimal absolute-error scaling, compression/stretch, and 0/1.75/3.5% mean ceilings.

## 2026 WAPE / MAPE by subtype and SSAM

{table(compare,percent=['baseline_WAPE','new_WAPE','baseline_MAPE','new_MAPE'])}

SSAM=1 has **only two** mature 2026 stores; MALL has six. These subgroup metrics are unstable. There are no SSAM=1 rows in the 2025 selection cohort. The full cross-tab is included below; never average subgroup percentages to obtain total WAPE—recompute from summed dollar errors and actual sales.

{table(testseg,percent=['WAPE','MAPE'])}

## Historical rolling predictions (2022–2025)

{table(oofseg,percent=['WAPE','MAPE'])}

Each underlying prediction excludes that year's labels, but the selected blend was chosen using these cohorts. These are retrospective development diagnostics, not an independent assessment. This table uses the raw selected blend; final postprocessing is evaluated separately on 2026.

## Training fitted errors (optimistic)

{table(fitseg,percent=['WAPE','MAPE'])}

These predictions are from models fit on the pre-2026 training cohort; they are not generalization estimates. Calibration uses earlier OOF residuals and the 2026 forecast context, so this table diagnoses model fit rather than a historical prospective forecast.

## Distributions

![Observed sales by year](sales_distribution_by_year.png)

![SSAM distributions](sales_distribution_by_ssam.png)

![2026 forecast comparison](test_comparison.png)

All chart years are **anniversary years** (`open_date + 1 year`), consistent with the original notebook. Training comprises 1,180 eligible stores from 2016–2025; test comprises 124 mature stores in the partial 2026 cohort. Strict cutoff: `open_date < 2025-09-29`, existing location, positive target. The target remains `12 × avg_monthly_sales`; source measurement-window semantics still require confirmation. Boxplots show the observed data, not fitted distributions. SSAM cumulative plots distinguish training actuals, 2026 actuals, and 2026 predictions. Dollar-level quantiles and means are in `actual_sales_distribution_statistics.csv`.

## Limits and deployment

Neighboring-sales, demographic, traffic, and boundary features are from one CSV snapshot, not verified point-in-time snapshots. Check availability at each historical forecasting date and ensure neighboring aggregates exclude the target store. Public documentation informed model objectives; no external store sales or future outcomes were merged. This avoids presenting current web data as historical features. The CSV has no repeated store-year observations, so cohort changes are not same-store growth estimates.

The production ensemble is refit on all 1,304 mature labeled stores after evaluation. `pipeline_forecasts.csv` contains the 566 pipeline locations; `inference_forecasts.csv` includes all 2,128 excluded-from-training rows, including older stores with missing targets. Forecast exports are in **dollars**. Files ending `_millions.csv` and `oof_prediction_matrix.csv` use **millions of dollars**. Long-horizon forecasts use the learned covariate relationships and a conservative mean ceiling; the ceiling is a maximum, not an assumption that each store's sales grows every year.

## Sources and reproducibility

- [CatBoost regression objectives](https://catboost.ai/docs/en/concepts/loss-functions-regression): MAE, RMSE, and Huber objectives.
- [scikit-learn histogram gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html): absolute-error and squared-error models.
- Run `python annual_ensemble.py`, then `python report_annual_ensemble.py` from this workspace using the installed packages. Candidate OOF predictions are checkpointed in `candidate_oof`; if inputs/specifications change, use a new output directory or remove only the affected cache files before rerunning.
- See `selection.json` for exact feature lists, specifications, selected weights, and calibration. The original CSV, original notebook, and previous results are unchanged.
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    cells=[nbformat.v4.new_markdown_cell('# Annual forecast: WAPE-focused ensembles\n\n'+conclusion+'\n\n2026 is a follow-up evaluation, not a fresh holdout. See the full report for temporal validation and data limitations.'),
        nbformat.v4.new_markdown_cell('## Results\n\n'+table(mt,percent=['WAPE','MAPE'])+'\n\n## Ensemble weights\n\n'+table(weights,percent=['weight'])),
        nbformat.v4.new_markdown_cell('## WAPE / MAPE by subtype and SSAM\n\n'+table(compare,percent=['baseline_WAPE','new_WAPE','baseline_MAPE','new_MAPE'])+'\n\nSSAM=1 test n=2; MALL n=6. Interpret cautiously.'),
        nbformat.v4.new_markdown_cell('## Sales distributions\n\n![By year](annual_ensemble_results/sales_distribution_by_year.png)\n\n![By SSAM](annual_ensemble_results/sales_distribution_by_ssam.png)\n\n![Forecast comparison](annual_ensemble_results/test_comparison.png)'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport sys\nROOT = Path.cwd()\nsys.path.insert(0, str(ROOT / '.annual_packages'))\nimport pandas as pd\nfrom IPython.display import display, Markdown\nOUT = ROOT / 'annual_ensemble_results'\ndisplay(pd.read_csv(OUT / 'segment_metrics.csv'))\ndisplay(pd.read_csv(OUT / 'actual_sales_distribution_statistics.csv'))\ndisplay(Markdown((OUT / 'REPORT.md').read_text(encoding='utf-8')))"),
        nbformat.v4.new_code_cell("# To rerun: checkpoint files are reused.\n# import annual_ensemble, report_annual_ensemble\n# from threadpoolctl import threadpool_limits\n# with threadpool_limits(limits=4):\n#     annual_ensemble.main()\n# report_annual_ensemble.main()")]
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbformat.validate(nb); nbformat.write(nb,ROOT/'annual-forecast-ensembles.ipynb')
    # Unit, cohort, export consistency checks.
    assert len(tr)==s['training_n'] and len(te)==s['test_n']
    assert p.location_num.is_unique and set(p.location_num)==set(te.location_num)
    assert np.isfinite(p.prediction).all() and p.prediction.gt(0).all()
    assert abs(metric(p.y,p.prediction)['WAPE']-new.WAPE)<1e-10
    assert abs(sum(s['weights'].values())-1)<1e-6
    assert p.prediction.mean()<=tr.loc[tr.year==2025,'y'].mean()*(1+chosen['growth'])+1e-8
    inference=pd.read_csv(OUT/'inference_forecasts.csv')
    for year,b in inference.groupby('forecast_year'):
        assert b.annual_sales_forecast.mean()<=te.y.mean()*1e6*(1+chosen['growth'])**(year-2026)+1e-4
    import hashlib,importlib.metadata as metadata
    manifest={'input_sha256':hashlib.sha256((ROOT/'sales_features.csv').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256((ROOT/'annual_ensemble.py').read_bytes()).hexdigest(),'versions':{name:metadata.version(name) for name in ['numpy','pandas','scikit-learn','scipy','catboost','matplotlib']},'checks':'cohort counts, unique test IDs, finite positive forecasts, recomputed WAPE, convex weights and annual mean ceilings passed'}
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(conclusion); print(compare.to_string(index=False)); print('Bootstrap CI:',ci)

if __name__=='__main__': main()

"""Create a Confluence-ready Markdown page and self-contained HTML equivalent."""
from pathlib import Path
import sys,json,html,base64
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.annual_packages'))
import pandas as pd
import numpy as np
import nbformat
OUT=ROOT/'annual_no_mall_results'

def markdown_table(frame):
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))

def main():
    summary=json.loads((OUT/'summary.json').read_text()); selected=summary['selection']
    metrics=pd.read_csv(OUT/'test_metrics.csv'); segments=pd.read_csv(OUT/'test_segment_metrics.csv',dtype={'group':str})
    ranking=pd.read_csv(OUT/'correction_ranking.csv',dtype={'growth':str}); audit=pd.read_csv(OUT/'data_audit.csv')
    baseline=metrics.iloc[0]; raw=metrics.iloc[1]; adjusted=metrics.iloc[2]
    change=(baseline.WAPE-adjusted.WAPE)*100
    ci=np.asarray(summary['bootstrap_WAPE_difference_95'])*100
    outcome=(f'The selected correction improves 2026 WAPE by {change:.2f} percentage points against the newly trained no-MALL baseline.' if change>0 else f'The selected correction does not improve 2026 WAPE: it is {-change:.2f} percentage points worse than the newly trained no-MALL baseline.')
    recommendation=('Treat this as a candidate for confirmation on newly mature stores before replacing the current model.' if change>0.25 and ci[2]<0 else 'Retain the no-MALL baseline for WAPE-first use; keep the correction experimental. The current accuracy difference is too small or uncertain to justify replacement.')
    if ci[0]<=0<=ci[2]: recommendation+=' The paired confidence interval includes zero, so the apparent difference is not conclusive.'
    recommendation+=f' Mean underforecasting changes from ${baseline.mean_residual:,.0f} to ${adjusted.mean_residual:,.0f} per store, while MAPE changes from {baseline.MAPE:.2%} to {adjusted.MAPE:.2%}.'
    compare=metrics[['model','n','WAPE','MAPE','MAE','bias','mean_residual','median_residual']].copy()
    for column in ['WAPE','MAPE','bias']: compare[column]=compare[column].map(lambda x:f'{x:.2%}')
    for column in ['MAE','mean_residual','median_residual']: compare[column]=compare[column].map(lambda x:f'${x:,.0f}')
    weights=pd.DataFrame({'Component':list(selected['weights']),'Weight':[f'{x:.0%}' for x in selected['weights'].values()]})
    group_rows=[]
    for (grouping,group),g in segments[segments.grouping.isin(['subtype','ssam_flag'])].groupby(['grouping','group']):
        b=g[g.model=='baseline'].iloc[0]; a=g[g.model=='adjusted'].iloc[0]
        group_rows.append({'Group':f'{grouping} = {group}','Stores':int(a.n),'Baseline WAPE':f'{b.WAPE:.2%}','Adjusted WAPE':f'{a.WAPE:.2%}','Baseline MAPE':f'{b.MAPE:.2%}','Adjusted MAPE':f'{a.MAPE:.2%}'})
    group_table=pd.DataFrame(group_rows)
    method_leaders=ranking.sort_values('WAPE').drop_duplicates('method')[['method','ensemble','strength','growth','WAPE']].copy()
    method_leaders['WAPE']=method_leaders.WAPE.map(lambda x:f'{x:.2%}')
    prior=ranking[(ranking.ensemble==selected['ensemble']) & (ranking.method=='none') & (ranking.growth==('uncapped' if selected['growth'] is None else str(selected['growth'])))].iloc[0]
    growth_text='No portfolio ceiling was selected; capped alternatives were evaluated.' if selected['growth'] is None else f"Latest observed cohort mean compounded at {selected['growth']:.1%} per year; never above 3.5%."
    scope=pd.DataFrame([
        {'Item':'As-of date','Value':selected['as_of']},
        {'Item':'Included subtypes','Value':'FSU and DTO only'},
        {'Item':'MALL rows excluded','Value':summary['excluded_mall_n']},
        {'Item':'Training / 2026 test','Value':f"{summary['train_n']} / {summary['test_n']} stores"},
        {'Item':'Inference / pipeline','Value':f"{summary['inference_n']} / {summary['pipeline_n']} stores"},
        {'Item':'Target','Value':'12 × avg_monthly_sales (annualized dollars)'},
        {'Item':'Year definition','Value':'First anniversary year: open_date + 1 year'},
        {'Item':'Status','Value':'Research follow-up; not a fresh independent holdout'},
    ])
    sections=[]
    def text(title,content): sections.append(('text',title,content))
    def table(title,frame): sections.append(('table',title,frame))
    def image(title,name): sections.append(('image',title,name))
    text('Executive summary',f"{outcome}\n\n2026 WAPE: **{baseline.WAPE:.2%} baseline → {adjusted.WAPE:.2%} adjusted**. MAPE: **{baseline.MAPE:.2%} → {adjusted.MAPE:.2%}**.\n\n{recommendation}\n\nMALL was removed before all model fitting, validation, calibration, evaluation and inference. Comparing these numbers with the earlier all-subtype results would mix a population change with a model change; the table below uses an apples-to-apples FSU/DTO baseline.")
    table('Scope and definitions',scope)
    table('2026 performance and residual offsets',compare)
    text('How to interpret the residual columns','Mean and median residuals are actual minus forecast, in dollars. Positive means underforecasting. Bias uses forecast minus actual divided by total actual sales. A smaller mean offset is not necessarily a lower WAPE: positive and negative errors can cancel. Selection therefore minimizes absolute error, not bias alone.')
    table('Selected model weights',weights)
    text('Selected postprocessing',f"**Method:** `{selected['method']}`. **Strength:** {selected['strength']}.\n\n**Portfolio constraint:** {growth_text}\n\n**Pooled 2024–2025 validation WAPE:** {selected['validation_WAPE']:.2%}; the same raw ensemble under the same ceiling has {prior.WAPE:.2%}. The score is selected from {summary['calibration_candidates']:,} alternatives and is a tuned validation result.\n\nAdditive learned corrections are clipped to ±$1.5M before shrinkage. Prediction compression depends on the forecast batch median. Final correction parameters are learned from historical OOF residuals; the training-fit mean offset is used only if that explicitly named method wins validation.")
    text('Experiment design','1. Read `sales_features.csv`, preserve location IDs as strings, and retain only FSU/DTO.\n2. Require an existing location, positive target, and an opening date strictly older than one year at September 30, 2026. Reserve all other FSU/DTO rows for inference.\n3. Compare five CatBoost specifications: original log target, dollar RMSE, dollar MAE, recency-weighted dollar RMSE and recency-weighted log target. Test 18 single-model/ensemble choices.\n4. Generate expanding-year OOF predictions for 2021–2025. Every store is predicted using only earlier anniversary years.\n5. For 2024 validation, fit corrections to OOF 2021–2023. For 2025 validation, fit corrections to OOF 2021–2024. Select by pooled WAPE across the two evaluation years.\n6. Freeze the selected weights and correction family. Refit correction parameters using OOF 2021–2025, fit base models on pre-2026 data, and score the 2026 follow-up cohort.\n7. After scoring, refit selected base models on all mature FSU/DTO labels for inference. MALL receives no forecast.')
    text('Preprocessing and correction methods','Categorical missing values receive an explicit label; numeric missing values are handled by CatBoost. Identifiers, eligibility/status fields, dates, raw targets and log targets are excluded from predictors. Log-target forecasts are reversed with `expm1`, consistent with the original notebook.\n\nCorrections tested: no adjustment; historical OOF mean offset; median offset; absolute-error-optimal scaling; robust affine correction; prediction-bin median residuals; group-shrunk residual offsets by subtype/SSAM/region; a regularized residual model; training-fit mean offset; and compression/stretch. Corrections use 25%, 50% or 100% strength where applicable. Portfolio mean ceilings of 0% and 3.5% growth are compared with uncapped diagnostics.')
    table('Best historical validation result by correction family',method_leaders)
    table('2026 WAPE / MAPE by subtype and SSAM',group_table)
    text('Uncertainty and sample size',f"Paired bootstrap, 3,000 resamples: **adjusted minus baseline WAPE 95% interval [{ci[0]:.2f}, {ci[2]:.2f}] percentage points**. Negative favors the correction. This interval excludes feature-provenance, cohort and model-search uncertainty.\n\nThe SSAM=1 test cohort contains only two stores, so its error rate should not be generalized. The 2026 anniversary cohort is partial. Training-fit and historical OOF subgroup metrics are exported separately; training-fit errors are optimistic.")
    image('Observed annual sales by cohort','sales_by_year.png')
    image('Sales distributions by SSAM','sales_by_ssam.png')
    image('Residual and tail diagnostics','residual_diagnostics.png')
    text('Limitations and interpretation','The CSV is one snapshot. Neighbor-sales, growth, traffic and demographic features are not certified as available at each historical forecast date; verify their timing and confirm neighboring aggregates exclude the target store. The original notebook imports its feature list from an unavailable constants module, so this baseline follows its model pattern with an explicit CSV-derived feature list rather than claiming exact reproduction of a registered production model.\n\nAnnualization assumes the monthly target represents the intended first-year average. The CSV cannot verify its measurement window. Cohort differences are not same-store growth because there is no repeated store-year panel.\n\n2026 has already been inspected in prior project rounds and informed the decision to exclude MALL. It is a follow-up evaluation, not an untouched holdout. Do not fit corrections directly to 2026 errors and then describe those same errors as test performance.\n\nForecasts for older stores with missing targets and distant pipeline years may be outside the labeled population. The mean ceiling is a maximum; it does not impose a fixed growth rate on every store.')
    text('Deliverables and handoff','**`sales-forecast-no-mall.ipynb`** — fully executed, standalone notebook with short markdown sections, explicit train/test splits, CatBoost training, residual calibration, plots and exports. It follows the original notebook’s pandas/numpy/`display(...)` style. Set `DATA_PATH` to run it in Databricks. No earlier modeling helper script is required.\n\n**`annual_no_mall.py`** — matching percent-cell Python source.\n\n**`test_predictions.csv`** — actual sales, baseline, raw ensemble, adjusted prediction and residuals.\n\n**`pipeline_forecasts.csv` / `inference_forecasts.csv`** — FSU/DTO forecasts in annual dollars.\n\n**`correction_ranking.csv` / `correction_validation_by_year.csv`** — complete validation search.\n\n**`test_segment_metrics.csv` / `training_and_oof_metrics.csv`** — subgroup diagnostics.\n\n**`model_bundle.pkl` / `selection.json`** — local trained artifacts and exact configuration. Model registration and warehouse publishing are not performed by this notebook.\n\nFor Confluence, copy the rendered HTML page or use the Markdown content, then attach the three PNG figures if needed. Review the largest residuals and point-in-time feature definitions before a new-cohort confirmation run.')
    title='Annual Sales Forecast | FSU / DTO | Residual Calibration'
    md=['# '+title]; page=['<!doctype html><html><head><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font:15px/1.6 Arial,sans-serif;color:#172b4d;max-width:1100px;margin:40px auto;padding:0 24px}h1{font-size:28px}h2{margin-top:32px;border-bottom:1px solid #dfe1e6;padding-bottom:8px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #dfe1e6;padding:9px;text-align:left}th{background:#f4f5f7}img{max-width:100%;height:auto}p{white-space:pre-line}code{background:#f4f5f7;padding:2px 4px} .status{background:#e9f2ff;padding:14px;border-left:4px solid #0c66e4}</style></head><body><h1>'+html.escape(title)+'</h1>']
    import re
    def rich(s):
        s=html.escape(s)
        s=re.sub(r'\*\*(.*?)\*\*',r'<strong>\1</strong>',s)
        return re.sub(r'`(.*?)`',r'<code>\1</code>',s)
    for kind,heading,value in sections:
        md.extend(['','## '+heading,'']); page.append('<h2>'+html.escape(heading)+'</h2>')
        if kind=='text':
            md.append(value); page.extend('<p>'+rich(p)+'</p>' for p in value.split('\n\n'))
        elif kind=='table':
            md.append(markdown_table(value)); page.append(value.to_html(index=False,escape=True))
        else:
            md.append(f'![{heading}]({value})')
            encoded=base64.b64encode((OUT/value).read_bytes()).decode('ascii')
            page.append(f'<img alt="{html.escape(heading)}" src="data:image/png;base64,{encoded}">')
    page.append('</body></html>')
    (OUT/'CONFLUENCE_SUMMARY.md').write_text('\n'.join(md),encoding='utf-8')
    (OUT/'CONFLUENCE_SUMMARY.html').write_text('\n'.join(page),encoding='utf-8')
    # Verify the artifact actually contains runnable, executed cells and no errors.
    notebook=nbformat.read(ROOT/'sales-forecast-no-mall.ipynb',as_version=4)
    code=[c for c in notebook.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    assert 'import annual_' not in '\n'.join(c.source for c in code)
    assert len(pd.read_csv(OUT/'test_predictions.csv'))==summary['test_n']
    for file in ['test_predictions.csv','pipeline_forecasts.csv','inference_forecasts.csv']:
        frame=pd.read_csv(OUT/file)
        assert not frame.subtype.eq('MALL').any()
    checks={'executed_code_cells':len(code),'total_cells':len(notebook.cells),'notebook_errors':0,'no_mall_exports':True,'test_rows':summary['test_n'],'input_sha256':summary['input_sha256']}
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2))
    print(outcome); print(recommendation); print(compare.to_string(index=False)); print(group_table.to_string(index=False)); print(checks)

if __name__=='__main__': main()

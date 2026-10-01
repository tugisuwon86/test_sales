"""Prepare the extra-data request and document the outcome of the 10% search."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.annual_packages'))
import pandas as pd
import numpy as np
OUT=ROOT/'annual_target10_results'; OUT.mkdir(exist_ok=True)

def table(frame):
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))

def templates():
    d=pd.read_csv(ROOT/'sales_features.csv',dtype={'location_num':str})
    d=d[d.subtype.isin(['FSU','DTO'])]
    keys=['location_num','location_name','open_date','subtype','ssam_flag','state','dma','region','latitude','longitude']
    fields={
        'store_operations_request.csv':['forecast_as_of_date','feature_as_of_date','opening_event_type','relocation_source_location_num','target_window_start','target_window_end','target_month_count','closed_days_in_target','store_sqft','kitchen_sqft','drive_thru_lanes','order_points','pickup_windows','seating_capacity','weekly_operating_hours','parking_spaces','drive_thru_queue_capacity','first_full_operating_date','city','postal_code','census_tract_geoid'],
        'nearby_places_request.csv':['feature_as_of_date','nearest_same_brand_location_num','nearest_same_brand_drive_minutes','nearest_same_brand_annual_sales_as_of_date','nearest_same_brand_annual_sales','same_brand_count_5min','same_brand_count_10min','same_brand_openings_12m','same_brand_closures_12m','nearest_bigbox_brand','nearest_bigbox_drive_minutes','bigbox_count_5min','nearest_qsr_brand','nearest_qsr_drive_minutes','qsr_count_5min','qsr_count_10min','retail_center_type','frontage_road_aadt','highway_exit_drive_minutes','college_enrollment_10min','tourism_or_event_anchor','estimated_daytime_visits_5min'],
    }
    for filename,columns in fields.items():
        frame=d[keys].copy()
        for c in columns: frame[c]=''
        frame.to_csv(OUT/filename,index=False)
    pd.DataFrame(columns=['location_num','sales_month','net_sales','transactions','average_ticket','operating_days','closed_days','data_available_date']).to_csv(OUT/'monthly_sales_request_template.csv',index=False)
    prior=pd.read_csv(ROOT/'annual_no_mall_results/test_predictions.csv',dtype={'location_num':str})
    prior['baseline_absolute_error']=(prior.annual_sales-prior.baseline).abs()
    prior['baseline_APE']=prior.baseline_absolute_error/prior.annual_sales
    prior.sort_values('baseline_absolute_error',ascending=False).to_csv(OUT/'priority_label_audit_2026.csv',index=False)
    actual=prior.annual_sales.sum(); error=prior.baseline_absolute_error.sum()
    ordered=prior.sort_values('baseline_absolute_error',ascending=False)
    gap={'actual_total_dollars':actual,'baseline_absolute_error_dollars':error,'target_error_budget_dollars':actual*.1,
         'required_error_reduction_dollars':error-.1*actual,'required_error_reduction_fraction':1-.1*actual/error,
         'diagnostic_WAPE_if_top10_errors_were_perfect':(error-ordered.baseline_absolute_error.head(10).sum())/actual}
    (OUT/'target_gap.json').write_text(json.dumps(gap,indent=2))
    request='''# Data needed to pursue 10% store-level WAPE

Please provide FSU/DTO data for **all available historical and inference locations**, not only the 2026 misses. Keep `location_num` as a string, including leading zeroes. The two location-level templates contain the 3,190 supplied FSU/DTO locations and their known identifiers; new fields are intentionally blank.

## Priority 1 — establish the target and event type

- Define `avg_monthly_sales`: first 12 full months, trailing 12 months, or another window. Provide target window start/end, number of valid months, temporary closure days, and sales available dates.
- Define `ssam_flag`. Identify new opening, relocation, rebuild/reopening, and other events; give the source store ID for relocations. A store reopening after an established business may have a different demand baseline from a new market entry.
- Provide monthly sales, transactions and operating days, ideally covering the complete first 24 months and any predecessor-store history. A blank header-only template is provided. Monthly target-period outcomes are labels, not pre-opening predictors.
- Give the intended forecast issue date. Early 30/90-day sales could support a different post-opening forecast, but must not be used to claim a pre-opening result.

## Priority 2 — demand and physical capacity

- Store/kitchen square footage; drive-through lanes, order points and pickup windows; queue capacity; seating; operating hours and parking.
- Same-brand locations within 5/10 minutes, their sales known before forecast issuance, opening/closure dates, relocation source and transferred business.
- These fields must reflect the version available at the intended forecast date, not the current remodeled store.

## Priority 3 — external demand generators and access

- Nearest big-box and QSR **brand/name plus drive time**, and counts by 5/10-minute catchment. Nearest-name-only is weaker than identity, size, access and distance together.
- Retail-center type/anchor, frontage-road traffic, highway-exit access, colleges, event/tourism anchors and dated visitation estimates where available.
- Validated city, postal code and Census tract GEOID. The current file has state and DMA, but no reliable city field; location-name fragments should not be treated as verified city names.
- Store data can then be joined to lagged tract-level education, employment mix, vehicle access and commuting statistics.

## Requested files

1. `store_operations_request.csv`: fill the added operational, event and target-audit columns.
2. `nearby_places_request.csv`: fill the dated supply/competition/anchor columns.
3. `monthly_sales_request_template.csv`: return one row per location/month.

`priority_label_audit_2026.csv` identifies the largest current errors for source verification. It is **not** a list of observations to remove or a permission to fit special-case rules to their test outcomes. Pull comparable information for the training cohort as well.

## Public-data access status

- Census ACS API returned a Missing Key page. A downloaded, appropriately dated ACS extract with tract/county identifiers can be used; do not paste API keys into chat.
- The BEA direct ZIP endpoint returned an HTML landing page, BLS direct download returned HTTP 403, and FRED CSV download returned HTTP 404 in this environment.
- No public numerical data were successfully ingested in this run. Sources were researched; missing values were not invented or silently replaced with current data.
- Sources: [Census geography/API guide](https://www.census.gov/data/developers/guidance/api-user-guide.html), [Census LODES employment data](https://lehd.ces.census.gov/data/lehd-code-samples/sections/lodes.html), [BEA downloadable regional accounts](https://apps.bea.gov/regional/downloadzip.htm), [FRED example BEA state-income series](https://fred.stlouisfed.org/series/CAPCPI).
'''
    (OUT/'DATA_REQUEST.md').write_text(request,encoding='utf-8')
    bad=OUT/'public_data'/'SAINC1.zip'
    if bad.exists() and bad.read_bytes()[:2]!=b'PK':
        target=bad.with_name('BEA_unusable_landing_page.html')
        if not target.exists(): bad.rename(target)

def report():
    s=json.loads((OUT/'summary.json').read_text()); gap=json.loads((OUT/'target_gap.json').read_text())
    metrics=pd.read_csv(OUT/'test_metrics.csv'); p=pd.read_csv(OUT/'test_predictions_millions.csv')
    base=metrics.iloc[0]; adjusted=metrics.iloc[-1]
    comparison=metrics.copy()
    for c in ['WAPE','MAPE','bias']: comparison[c]=comparison[c].map(lambda x:f'{x:.2%}')
    comparison['MAE_dollars']=comparison.MAE_dollars.map(lambda x:f'${x:,.0f}')
    weights=pd.DataFrame([{'Component':c,'Weight':f'{v:.1%}'} for c,v in s['selection']['weights'].items()])
    rng=np.random.default_rng(42); diffs=[]
    for _ in range(3000):
        ix=rng.integers(0,len(p),len(p)); b=p.iloc[ix]
        diffs.append((abs(b.y-b.prediction).sum()-abs(b.y-b.baseline).sum())/b.y.sum())
    ci=np.quantile(diffs,[.025,.975])
    decision='The 10% target was reached on this follow-up evaluation, but still needs fresh-cohort confirmation.' if adjusted.WAPE<=.1 else f'The 10% target was **not reached**. The frozen candidate scores **{adjusted.WAPE:.2%} WAPE**, versus **{base.WAPE:.2%}** for the no-MALL baseline.'
    md=f'''# Pursuing 10% annual store-level WAPE

{decision}

## Follow-up results

{table(comparison)}

Paired store-bootstrap 95% interval, adjusted minus baseline WAPE: [{ci[0]*100:.2f}, {ci[1]*100:.2f}] percentage points. Negative favors the candidate. This does not include model-search or feature-provenance uncertainty. 2026 has been examined repeatedly, so it is not an independent holdout.

## What was tried in this round

{s['candidate_count']} new specifications: LightGBM with multiple leaf sizes and absolute/squared losses; XGBoost with multiple depths and absolute/squared losses; log and neighbor-ratio targets; CatBoost with additional history features; regional/state/DMA experts shrunk toward a national model; nearest-neighbor models; support-vector regression; geographic and demographic comparable-store rules.

New history features use only reference stores whose anniversary precedes the subject opening date. They include nearby 3/10/25-store estimates, demographic analogues, and region/state/DMA historical medians with sample-size shrinkage. Reference sales are grown at no more than 3.5% annually. The current CSV target window still needs confirmation before those values can be certified as historical first-year outcomes.

Model ranking and nonnegative blend weights use 2023–2024 rolling predictions. {s['rule_count']} blend/postprocessing configurations are selected on 2025. Rules include global, region, state and DMA offsets and absolute-error-optimal scale adjustments. Correction strengths are shrunk. The chosen weights and rule are frozen before scoring 2026; final portfolio mean is capped at the 2025 cohort mean plus 3.5%.

No store-name exceptions, exclusions based on error size, or rules fitted to 2026 actuals were used. City rules require a validated city field; the existing DMA is not called a city.

## Selected weights and rule

{table(weights)}

Rule: {s['selection']['choice']['method']}, strength {s['selection']['choice']['strength']}.

## Size of the remaining problem

The baseline has **${gap['baseline_absolute_error_dollars']/1e6:.1f}M** of absolute error across 118 stores. Ten-percent WAPE permits **${gap['target_error_budget_dollars']/1e6:.1f}M**. That requires a **{gap['required_error_reduction_fraction']:.1%}** reduction in the baseline error.

Even a diagnostic scenario that magically fixed the ten largest errors would leave **{gap['diagnostic_WAPE_if_top10_errors_were_perfect']:.2%} WAPE** across the full cohort. This is not a forecast result; it demonstrates why a few location-specific exceptions cannot meet the target.

These experiments do not prove 10% is impossible or establish an irreducible error floor. They do show that reaching it has not been supported by the supplied feature set and tested methods. The next material step is target/event verification and additional pre-forecast operating-capacity and local-demand information, rather than reporting fitted test corrections as improved test accuracy.

## Public sources

Official Census ACS/LODES, BEA, BLS and FRED sources were researched. Numerical downloads were not successfully available through the attempted endpoints; no public numerical features were ingested. See `DATA_REQUEST.md` for source links and exact access results. Do not interpret public-data research as a successful enrichment.

## Next data pull

Use `store_operations_request.csv`, `nearby_places_request.csv`, and `monthly_sales_request_template.csv`. The two location-level files are prefilled with all 3,190 FSU/DTO identifiers. Priority: establish target windows and event types; then operating capacity and same-brand predecessor/nearby-store history; then dated QSR/anchor/access information. Enrich training and inference consistently, not just the 2026 misses.

Review `priority_label_audit_2026.csv` to verify the largest errors at their source. Full method scores are in `candidate_validation.csv`, `development_ranking.csv` and `rule_selection_2025.csv`; state/region/DMA/subtype/SSAM test metrics are in `test_group_metrics.csv`.
'''
    (OUT/'REPORT.md').write_text(md,encoding='utf-8')
    (OUT/'bootstrap.json').write_text(json.dumps({'new_minus_baseline_95':ci.tolist()},indent=2))
    # Create a readable companion notebook with the implementation broken into sections.
    import ast,nbformat
    source=(ROOT/'annual_target10.py').read_text(encoding='utf-8'); tree=ast.parse(source)
    funcs=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    first=funcs[0].lineno; lines=source.splitlines()
    cells=[nbformat.v4.new_markdown_cell('# Annual forecast: geography and 10% WAPE target\n\n'+decision+'\n\nMALL remains excluded. This follow-up evaluation does not establish an untouched holdout. See the report and data-request files.'),
           nbformat.v4.new_markdown_cell('### Results\n\n'+table(comparison)),
           nbformat.v4.new_markdown_cell('### Setup\nRun from the workspace containing sales_features.csv. This notebook contains the modeling code; cached OOF predictions are reused.'),
           nbformat.v4.new_code_cell('\n'.join(lines[:first-1]).replace('ROOT=Path(__file__).resolve().parent','ROOT=Path.cwd()'))]
    descriptions={'metric':'Evaluation metrics','data':'Load FSU/DTO and transform features','historic_features':'Build chronological geographic and demographic comparables','specifications':'Define the new model grid','fit_predict':'Fit one model or geographic expert','fit_weights':'Fit nonnegative absolute-error blend weights','rule':'Fit and apply historical regional rules'}
    for f in funcs:
        if f.name=='main': continue
        cells += [nbformat.v4.new_markdown_cell('### '+descriptions.get(f.name,f.name)),nbformat.v4.new_code_cell(ast.get_source_segment(source,f))]
    main=next(f for f in funcs if f.name=='main')
    body='\n'.join(line[4:] if line.startswith('    ') else line for line in lines[main.lineno:main.end_lineno])
    # Split main at natural execution boundaries without hiding the work in a single helper call.
    markers=['    dev=oof.year<2025','    search=[]','    choice=search.iloc[0]','    tr=labeled[labeled.year<2026]','    output=test[[']
    chunks=[]; buffer=[]
    for line in body.splitlines():
        if any(line.startswith(m.strip()) for m in markers) and buffer:
            chunks.append('\n'.join(buffer)); buffer=[]
        buffer.append(line)
    if buffer: chunks.append('\n'.join(buffer))
    labels=['Generate historical features and rolling model predictions','Rank candidates and define blends','Validate region/state/DMA rules on 2025','Freeze selection','Score the 2026 follow-up cohort','Export diagnostics']
    for label,chunk in zip(labels,chunks):
        cells += [nbformat.v4.new_markdown_cell('### '+label),nbformat.v4.new_code_cell(chunk)]
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    for c in notebook.cells:
        if c.cell_type=='code': compile(c.source,'notebook','exec')
    nbformat.validate(notebook); nbformat.write(notebook,ROOT/'sales-forecast-target10-geography.ipynb')
    print(decision); print(metrics.to_string(index=False)); print('Prepared data request templates and notebook.')

if __name__=='__main__':
    templates()
    if '--templates-only' not in sys.argv: report()

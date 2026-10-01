"""Build the review notebook, report, and paired holdout uncertainty estimates."""
from pathlib import Path
import sys, json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.annual_packages'))
import pandas as pd
import numpy as np
OUT=ROOT/'annual_results'

def table(frame):
    def fmt(x): return f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))

def main():
    s=json.loads((OUT/'summary.json').read_text()); a=json.loads((OUT/'data_audit.json').read_text())
    p=pd.read_csv(OUT/'test_2026_predictions.csv'); m=pd.read_csv(OUT/'test_2026_metrics.csv')
    display_metrics=m[['model','MAE','WAPE','low_MAPE','high_MAPE','bias']]
    r=pd.read_csv(OUT/'model_ranking.csv'); pp=pd.read_csv(OUT/'postprocess_validation.csv')
    rng=np.random.default_rng(42); y=p.actual_annualized_sales.to_numpy()
    delta=[]
    for _ in range(3000):
        ix=rng.integers(0,len(p),len(p)); yy=y[ix]
        delta.append((abs(p.selected_adjusted.to_numpy()[ix]-yy).sum()-abs(p.baseline.to_numpy()[ix]-yy).sum())/yy.sum())
    ci=np.quantile(delta,[.025,.5,.975]); pd.Series(ci,index=['lower_95','median','upper_95'],name='selected_minus_baseline_WAPE').to_csv(OUT/'paired_bootstrap.csv')
    b=s['selected']
    baseline=m.iloc[0]; selected=m.iloc[2]
    conclusion=(f"On the 2026 holdout, WAPE changes from {baseline.WAPE:.1%} to {selected.WAPE:.1%}; "
                f"low-tail MAPE from {baseline.low_MAPE:.1%} to {selected.low_MAPE:.1%}; "
                f"high-tail MAPE from {baseline.high_MAPE:.1%} to {selected.high_MAPE:.1%}. ")
    conclusion += ('The selected candidate improves overall WAPE on this holdout. ' if selected.WAPE<baseline.WAPE else 'The selected candidate does not improve overall WAPE on this holdout; do not replace the baseline on this evidence alone. ')
    if selected.low_MAPE>=baseline.low_MAPE or selected.high_MAPE>=baseline.high_MAPE:
        conclusion += 'It does not improve both tails, so the tail problem is not fully resolved.'
    inf=pd.read_csv(OUT/'inference_forecasts.csv',dtype={'location_num':str})
    inf[inf.location_type=='pipeline'].to_csv(OUT/'pipeline_forecasts.csv',index=False)
    assert len(p)==a['test_2026'] and p.location_num.is_unique
    assert len(inf)==s['inference_rows'] and inf.location_num.is_unique
    assert np.isfinite(p[['baseline','selected_raw','selected_adjusted']].to_numpy()).all()
    assert np.isfinite(inf[['raw_annualized_sales','forecast_annualized_sales']].to_numpy()).all()
    assert (inf.forecast_annualized_sales>0).all()
    cohorts=pd.read_csv(OUT/'cohorts.csv')
    if b['method']!='raw':
        prior=cohorts[cohorts.year==2025].mean_annual_sales.iloc[0]
        assert p.selected_adjusted.mean()<=prior*(1+b['growth'])+1e-5
        latest=cohorts.iloc[-1]
        for year,part in inf.groupby('forecast_year'):
            ceiling=latest.mean_annual_sales*(1+b['growth'])**max(0,year-latest.year)
            assert part.forecast_annualized_sales.mean()<=ceiling+1e-5
    inf.groupby(['location_type','forecast_year']).forecast_annualized_sales.agg(['count','mean','min','median','max']).to_csv(OUT/'inference_distribution_summary.csv')
    import importlib.metadata as metadata
    versions={name:metadata.version(name) for name in ['numpy','pandas','scikit-learn','scipy','catboost','matplotlib','nbformat']}
    (OUT/'package_versions.json').write_text(json.dumps(versions,indent=2))
    report=f'''# Annual sales forecast experiment

As of **{a['as_of']}**, using the supplied CSV and notebook. Original inputs are unchanged.

## Validation-selected experimental candidate

Validation selected **{b['candidate']}**, with **{b['method']}**, strength **{b['strength']}**, grouping **{b['grouping']}**, and **{100*b['growth']:.2f}%** assumed annual growth. This selection was frozen before scoring 2026. Do not select alternatives using the separately supplied 2026 diagnostic grid.

{conclusion}

{table(display_metrics)}

Sales and dollar errors above are annualized dollars. MAPE, WAPE, bias, and tail MAPE are fractions (0.10 = 10%). Low and high tails are the bottom and top 20% of **actual** holdout sales; these are evaluation groups, never prediction inputs. The selection score is WAPE + 0.25 × (low-tail MAPE + high-tail MAPE), averaged equally across 2023–2025 folds.

Paired store bootstrap, 3,000 resamples: selected-minus-baseline WAPE 95% interval **[{ci[0]*100:.2f}, {ci[2]*100:.2f}] percentage points**. Negative favors the selected candidate. This reflects store sampling uncertainty only, not feature leakage, repeated model selection, or cohort uncertainty.

## Data and evaluation

- Target: `12 * avg_monthly_sales`. This assumes the source average is the relevant first-year monthly average. The CSV cannot verify the underlying measurement window or establish actual calendar-year sales.
- Strict maturity rule: `open_date < 2025-09-29`, positive observed target, and `location_type == existing`.
- {a['eligible']} training/evaluation-eligible stores; {a['test_2026']} holdout stores with anniversary year 2026. {a['immature_labeled_excluded']} immature labeled rows excluded.
- Expanding validation: train on anniversary years before 2023, test 2023; then before 2024, test 2024; then before 2025, test 2025. Final holdout fit uses years before 2026 only.
- The 2026 cohort is partial through September 28. Earlier validation cohorts cover full years; opening-season mix can affect comparability.
- Missing values are retained and imputed within each sklearn training fold, or handled natively by CatBoost. No target imputation. IDs, names, status, raw dates, anniversary, and target/log-target columns are excluded from predictors.
- Five specifications: notebook-style CatBoost on log sales, CatBoost on dollar sales, ridge on log sales, Extra Trees, and histogram gradient boosting. Five feature sets: compact ({len(a['features']['compact'])}), static, static plus boundary/nearby-opening features, all eligible inputs, and all except growth features. Exact lists are in `data_audit.json`.
- The notebook imports its original feature list from an unavailable `constants` module. The baseline is a reconstruction using all eligible CSV inputs, 200 iterations/depth 6/learning rate .05, not a claim to reproduce the registered production model exactly.

## Distribution adjustments

The four best raw candidates on validation were each tested with raw predictions plus 108 adjustment combinations: linear stretch, power stretch, 25/50/75/100% rank-based historical quantile matching, and partial/full mean anchoring. Each uses global, subtype, or region × subtype grouping and 0%, 1.75%, or 3.5% growth.

Historical references use the preceding three available anniversary cohorts. Each reference observation is grown to the forecast year at the chosen rate. Fewer than 30 reference stores triggers subtype then global pooling; if recent global history has fewer than 30, all earlier training history is used. Adjusted predictions are positive, and their group mean is capped at the grown reference mean. A second portfolio ceiling caps the batch mean at the latest observed cohort mean compounded to the forecast year at the chosen rate (at most 3.5%). This prevents pooling from implying a larger year-over-year mean jump. These are cohort-level constraints, not individual-store growth limits. Raw candidates have no mean constraint.

Quantile mapping uses prediction ranks within the forecast batch, never test labels. Consequently forecasts can change when the batch composition changes, especially for small groups. Better matching a marginal distribution does not establish better store-level ranking. Stretching can amplify errors as well as recover tails.

The holdout chart shows that several of the highest-sales stores still receive middle-range forecasts. Stretching cannot reliably fix those ranking errors. MALL has just six holdout stores and remains weak (approximately 40.5% MAPE); pooling sparse subtype history into the global distribution can overpredict this subtype. A follow-up should test longer subtype-specific history or hierarchical shrinkage using a fresh temporal holdout, rather than tuning these decisions on 2026. Treat the supplied inference forecasts as experimental.

## Feature timing and scope limitations

The CSV is a single snapshot. Neighbor sales, growth, boundary counts, demographics, and traffic are not certified as point-in-time features. Comparisons that exclude sales/growth features are included, but even static-looking inputs need historical availability checks before treating these as prospective backtest estimates. Confirm that neighboring-sales aggregates exclude each target store. There is no store-year panel here, so cohort mean differences are not estimates of same-store annual growth.

Many older existing stores have missing targets. These are scored separately from the evaluated population and may not be comparable to recent labeled openings. Inference includes all {s['inference_rows']} rows outside training eligibility, with a reason column; it does not treat their available immature labels as annual ground truth. Future forecast years use anniversary year, floored at 2026. Long-range forecasts and sparse subtype forecasts need additional review. Production fitting includes the now-observed mature 2026 rows only after holdout evaluation.

## Validation leaders

{table(r.head(10))}

## Deliverables and rerun

- `annual-forecast-experiments.ipynb`: review notebook with metrics, plots, feature lists, and a rerun cell.
- `annual_forecast.py`: full reproducible modeling and calibration implementation.
- `test_2026_predictions.csv`: untouched holdout actuals, baseline, raw, adjusted predictions.
- `inference_forecasts.csv`: production refit forecasts for rows excluded from labeled training.
- `pipeline_forecasts.csv`: pipeline-only subset for operational review.
- `model_validation.csv`, `model_ranking.csv`, `postprocess_validation.csv`: complete validation comparisons.
- `test_2026_postprocess_diagnostics_NOT_FOR_SELECTION.csv`: exploratory holdout adjustment comparisons only.
- `model.pkl`: local fitted model, feature list, calibration configuration, and historical reference. Load only this trusted local artifact.

Install `requirements-annual.txt` into a Python environment and run `python annual_forecast.py`, then `python summarize_annual.py`. The scripts also recognize workspace-local `.annual_packages`. ASOF is deliberately fixed in the script for reproducibility; update it explicitly for future runs.
'''
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
    cells=[]
    def md(x): cells.append(dict(cell_type='markdown',metadata={},source=x.splitlines(True)))
    def code(x, output=None):
        cells.append(dict(cell_type='code',execution_count=None,metadata={},source=x.splitlines(True),outputs=output or []))
    md('# Annual sales forecasting: models, features, and conservative calibration\n\nReview completed experiments below. The original notebook and CSV are preserved. See `annual_results/REPORT.md` for assumptions and limits. **2026 is a held-out anniversary cohort, not 2026 openings.**')
    code("from pathlib import Path\nimport sys\nROOT = Path.cwd()\nif not (ROOT / 'sales_features.csv').exists():\n    raise RuntimeError('Open this notebook from the sales workspace.')\nsys.path.insert(0, str(ROOT / '.annual_packages'))\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nOUT = ROOT / 'annual_results'\n")
    md('## Protocol\n\nUse open dates strictly before September 29, 2025 and positive observed sales. Annualize monthly averages by 12. Select using expanding anniversary-year folds 2023–2025, then evaluate once on the 124 mature 2026 stores. Tune the four validation-leading models with 109 calibration configurations each. No holdout labels enter fitting or calibration. All CSV feature provenance still requires a point-in-time audit.')
    code("display(pd.read_csv(OUT / 'cohorts.csv'))\ndisplay(pd.read_csv(OUT / 'model_ranking.csv'))")
    md('## Selected model and holdout result\n\n'+f"Selected: **{b['candidate']}**, **{b['method']}**, strength {b['strength']}, {b['grouping']}, {100*b['growth']:.2f}% growth.\n\n"+conclusion+'\n\n'+table(display_metrics)+'\n\n![Holdout diagnostics](annual_results/test_2026_diagnostics.png)')
    code("display(pd.read_csv(OUT / 'test_2026_metrics.csv'))\ndisplay(Image(filename=str(OUT / 'test_2026_diagnostics.png')))\ndisplay(pd.read_csv(OUT / 'test_2026_segments.csv'))")
    md('## Review calibration, tails, and inference\n\nPostprocessing is selected on earlier years. The 2026 diagnostic sweep is exploratory and must not be used to advertise an untouched holdout improvement. Rank mapping is batch dependent. Forecast CSV values are annualized dollars.')
    code("display(pd.read_csv(OUT / 'postprocess_validation.csv').head(20))\ndisplay(pd.read_csv(OUT / 'test_2026_predictions.csv').head())\ndisplay(pd.read_csv(OUT / 'inference_forecasts.csv').head())\ndisplay(Markdown((OUT / 'REPORT.md').read_text(encoding='utf-8')))")
    md('## Rerun all experiments\n\nThis cell refits every candidate and regenerates outputs; it may take several minutes. It never changes the input CSV or original notebook.')
    code("# Uncomment to rerun.\n# import annual_forecast, summarize_annual\n# annual_forecast.main()\n# summarize_annual.main()")
    for i,cell in enumerate(cells): cell['id']=f'annual-{i:02d}'
    nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':sys.version.split()[0]}},nbformat=4,nbformat_minor=5)
    import nbformat
    notebook=nbformat.from_dict(nb)
    nbformat.write(notebook,ROOT/'annual-forecast-experiments.ipynb')
    print('Report and notebook created. Bootstrap interval:',ci)

if __name__=='__main__': main()

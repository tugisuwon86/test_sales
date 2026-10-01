"""Model training, walk-forward backtest, residual correction, save/load."""

import pickle
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from . import constants as C
from .features import FeatureSelector, fit_trend, outcome_features
from .metrics import summarize

warnings.filterwarnings("ignore", message="X does not have valid feature names")

ANCHOR_MIN_COHORT_SIZE = 20  # smaller inference cohorts use the pooled anchor shift


class BaseEnsemble:
    """Geometric mean of the models in constants.BASE_MODELS. Predicts log sales."""

    def fit(self, train: pd.DataFrame):
        self.selector_ = FeatureSelector().fit(train)
        self.trend_ = fit_trend(train)
        self.max_years_ = train["years_since_2015"].max() + C.MAX_TREND_EXTRAPOLATION_YEARS
        self.train_ = train  # needed to build time-safe outcome features at scoring time
        train_extra, _ = outcome_features(train, train.iloc[:1], self.trend_, self.max_years_)
        x_train = pd.concat([self.selector_.transform(train), train_extra], axis=1)
        self.columns_ = x_train.columns.tolist()

        self.models_ = {}
        for name, spec in C.BASE_MODELS.items():
            y_train = train[C.log_target].to_numpy()
            if spec["target"] == "residual":
                y_train = y_train - x_train["trend"].to_numpy()
            if spec["family"] == "lgb":
                model = LGBMRegressor(objective=spec["objective"], **C.LGB_PARAMS)
            else:
                model = CatBoostRegressor(depth=spec["depth"], **C.CAT_PARAMS)
            model.fit(x_train, y_train)
            self.models_[name] = model
        return self

    def predict_log(self, score: pd.DataFrame) -> pd.DataFrame:
        """Per-model and ensemble log predictions for `score` rows."""
        _, score_extra = outcome_features(self.train_, score, self.trend_, self.max_years_)
        x_score = pd.concat([self.selector_.transform(score), score_extra], axis=1)[self.columns_]
        out = pd.DataFrame(index=score.index)
        for name, model in self.models_.items():
            prediction = model.predict(x_score)
            if C.BASE_MODELS[name]["target"] == "residual":
                prediction = prediction + x_score["trend"].to_numpy()
            out[name] = prediction
        out["base"] = out[list(self.models_)].mean(axis=1)
        return out


def walk_forward_base(labeled: pd.DataFrame, cohorts) -> pd.DataFrame:
    """Out-of-fold base predictions: cohort Y is predicted by a model trained on cohorts < Y."""
    parts = []
    for cohort in cohorts:
        train = labeled[labeled["cohort"] < cohort]
        test = labeled[labeled["cohort"] == cohort]
        prediction = BaseEnsemble().fit(train).predict_log(test)
        parts.append(prediction)
        print(f"{cohort=} {len(train)=} {len(test)=} base WAPE={summarize(test[C.target], np.exp(prediction['base']))['WAPE']:.4f}")
    return pd.concat(parts)


class ResidualCorrector:
    """Learns persistent biases from out-of-fold residuals of earlier cohorts.

    1. Factor offsets: ridge regression of the cohort-centred log residual on
       constants.RESIDUAL_FACTORS (relocated, SSAM, DTO, opening quarter).
    2. Level anchor: moves each cohort's median forecast part of the way toward the
       latest observed cohort's actual median (plus an assumed growth rate).
    """

    def fit(self, history: pd.DataFrame, log_base: pd.Series):
        residual = history[C.log_target] - log_base
        centred = residual - residual.groupby(history["cohort"]).transform("median")
        self.ridge_ = Ridge(alpha=C.RESIDUAL_RIDGE_ALPHA).fit(
            history[C.RESIDUAL_FACTORS].astype(float), centred
        )
        self.latest_cohort_ = int(history["cohort"].max())
        latest = history[history["cohort"] == self.latest_cohort_]
        self.latest_median_ = float(latest[C.log_target].median())
        return self

    def factor_offsets(self) -> pd.Series:
        return pd.Series(self.ridge_.coef_, index=C.RESIDUAL_FACTORS)

    def apply(self, frame: pd.DataFrame, log_pred: pd.Series) -> pd.Series:
        offsets = self.ridge_.predict(frame[C.RESIDUAL_FACTORS].astype(float)) - self.ridge_.intercept_
        corrected = log_pred + offsets
        target = self.latest_median_ + C.LEVEL_ANNUAL_GROWTH * (frame["cohort"] - self.latest_cohort_)
        cohort_size = frame.groupby("cohort")["cohort"].transform("size")
        cohort_shift = target - corrected.groupby(frame["cohort"]).transform("median")
        large = cohort_size >= ANCHOR_MIN_COHORT_SIZE
        # Cohorts too small for a stable median borrow the store-weighted shift of the large cohorts.
        pooled_shift = cohort_shift[large].mean() if large.any() else (target - corrected).median()
        shift = np.where(large, cohort_shift, pooled_shift)
        return corrected + C.LEVEL_ANCHOR_STRENGTH * shift


def walk_forward_backtest(labeled: pd.DataFrame, oof: pd.DataFrame, cohorts) -> pd.DataFrame:
    """Score cohorts with corrections learned only from OOF residuals of earlier cohorts."""
    frame = labeled.loc[oof.index].copy()
    frame["log_base"] = oof["base"]
    frame["log_corrected"] = np.nan
    rows = []
    for cohort in cohorts:
        history = frame[frame["cohort"] < cohort]
        test = frame[frame["cohort"] == cohort]
        corrector = ResidualCorrector().fit(history, history["log_base"])
        corrected = corrector.apply(test, test["log_base"])
        frame.loc[test.index, "log_corrected"] = corrected
        for name, values in [("base", test["log_base"]), ("corrected", corrected)]:
            rows.append(dict(cohort=cohort, model=name, **summarize(test[C.target], np.exp(values))))
    return pd.DataFrame(rows), frame


class SalesForecaster:
    """Final model: base ensemble trained on all labeled stores + residual corrector."""

    def fit(self, labeled: pd.DataFrame, oof: pd.DataFrame, artifacts: dict = None):
        """Fit on all labeled stores.

        artifacts: optional outputs of this run (OOF predictions, backtest tables, clusters)
        stored with the model so the analysis notebook documents exactly this run.
        """
        self.base_ = BaseEnsemble().fit(labeled)
        history = labeled.loc[oof.index]
        self.corrector_ = ResidualCorrector().fit(history, oof["base"])
        self.trained_cohorts_ = sorted(labeled["cohort"].unique().tolist())
        self.as_of_date_ = C.AS_OF_DATE
        self.artifacts_ = artifacts or {}
        return self

    def feature_importance(self) -> pd.DataFrame:
        """LightGBM split-gain importance per base model, normalized to 100%."""
        out = {}
        for name, model in self.base_.models_.items():
            if hasattr(model, "booster_"):
                gain = model.booster_.feature_importance(importance_type="gain")
                out[name] = gain / gain.sum() * 100
        table = pd.DataFrame(out, index=self.base_.columns_)
        table["mean"] = table.mean(axis=1)
        return table.sort_values("mean", ascending=False)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        log_pred = self.base_.predict_log(frame)
        corrected = self.corrector_.apply(frame, log_pred["base"])
        out = frame[["location_num", "location_name", "location_type", "subtype", "state", "dma", "region", "open_date", "cohort", "is_relocated"]].copy()
        out["forecast_base_annual_sales"] = np.exp(log_pred["base"]) * 1e6
        out["forecast_annual_sales"] = np.exp(corrected) * 1e6
        out["forecast_avg_monthly_sales"] = out["forecast_annual_sales"] / 12
        # Open dates far in the future look like placeholders; check them before use.
        out["open_date_placeholder"] = out["open_date"].dt.year > C.AS_OF_DATE.year + C.PLACEHOLDER_OPEN_YEARS
        return out

    def save(self, path=C.MODEL_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path=C.MODEL_PATH) -> "SalesForecaster":
        with open(path, "rb") as f:
            return pickle.load(f)

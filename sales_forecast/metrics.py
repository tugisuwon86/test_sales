"""Accuracy metrics. All take actual and predicted values on the same scale."""

import numpy as np


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: total absolute error / total actual."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.abs(y_true - y_pred).sum() / y_true.sum()


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred) / y_true)


def median_ape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.median(np.abs((y_true - y_pred) / y_true))


def summarize(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return dict(
        n=len(y_true),
        WAPE=wape(y_true, y_pred),
        MAPE=mape(y_true, y_pred),
        median_APE=median_ape(y_true, y_pred),
        bias=(y_pred.sum() - y_true.sum()) / y_true.sum(),
    )

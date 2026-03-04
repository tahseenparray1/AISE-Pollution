import numpy as np


def rmse(actual, pred, dim=(1,2)):
    error = actual - pred
    return np.sqrt(np.nanmean(error**2, axis=dim))


def mfb(actual, pred, dim=(1,2)):
    error = (pred - actual)/(pred + actual)
    error = np.where(np.isfinite(error), error, np.nan)
    return 2 * np.nanmean(error,axis=dim)


def smape(actual, pred, dim=(1,2)):
    error = np.abs((actual - pred) / (actual+pred))
    error = np.where(np.isfinite(error), error, np.nan) 
    return 200 * np.nanmean(error, axis=dim)



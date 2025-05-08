import copy
from logging import getLogger

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from chat2plot.schema import PlotConfig, TimeUnit, XAxis

_logger = getLogger(__name__)


def transform(df: pd.DataFrame, config: PlotConfig) -> tuple[pd.DataFrame, PlotConfig]:
    config = copy.deepcopy(config)

    if config.x and (config.x.bin_size or config.x.time_unit):
        x_trans = _transform_x(df, config.x)
        df[x_trans.name] = x_trans
        config.x.column = x_trans.name

    return df, config


def _transform_x(df: pd.DataFrame, ax: XAxis) -> pd.Series:
    dst = df[ax.column].copy()

    try:
        if ax.bin_size:
            dst = binning(dst, ax.bin_size)

        if ax.time_unit:
            dst = round_datetime(dst, ax.time_unit)
    except Exception:
        _logger.warning("transforming x has an error.")

    return pd.Series(dst.values, name=ax.transformed_name())


def binning(series: pd.Series, interval: int) -> pd.Series:
    if not is_numeric_dtype(series.dtype):
        _logger.warning(
            f"binning on column {series.name} has been skipped because it is not a numerical type"
        )
        return series

    start_point = np.floor(series.min() / interval) * interval
    end_point = np.ceil((series.max() + 1) / interval) * interval
    bins = pd.interval_range(
        start=start_point,
        end=end_point,
        freq=interval,
        closed="both" if is_integer_dtype(series) else "left",
    )
    binned_series = pd.cut(series, bins=bins)
    return binned_series.astype(str)


def round_datetime(series: pd.Series, period: TimeUnit) -> pd.Series:
    if is_integer_dtype(series) and period == TimeUnit.YEAR:
        # assuming that it is year column, so no transform is needed
        return series

    try:
        series = pd.to_datetime(series)
    except Exception:
        try:
            series = pd.to_datetime(series, dayfirst=True)
        except Exception:
            _logger.warning(f"Skip to round datetime {series.name}")
            return series

    period_map = {
        TimeUnit.DAY: "D",
        TimeUnit.WEEK: "W",
        TimeUnit.MONTH: "M",
        TimeUnit.QUARTER: "Q",
        TimeUnit.YEAR: "Y",
    }
    return series.dt.to_period(period_map.get(period, period)).dt.to_timestamp()

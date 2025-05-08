import copy
from typing import Any

import altair as alt
import pandas as pd
import plotly.express as px
import vegafusion as vf
from altair.utils.data import to_values
from pandas.api.types import is_numeric_dtype
from pandas.core.groupby.generic import DataFrameGroupBy
from plotly.graph_objs import Figure

from chat2plot.schema import (
    AggregationType,
    BarMode,
    ChartType,
    Filter,
    PlotConfig,
    SortingCriteria,
    SortOrder,
)
from chat2plot.transform import transform


def _ax_config(config: PlotConfig, x: str, y: str) -> dict[str, str | dict[str, str]]:
    ax: dict[str, str | dict[str, str]] = {"x": x, "y": y}
    labels: dict[str, str] = {}

    if config.x and config.x.label:
        labels[x] = config.x.label
    if config.y.label:
        labels[y] = config.y.label

    if labels:
        ax["labels"] = labels

    return ax


def draw_plotly(df: pd.DataFrame, config: PlotConfig, show: bool = True) -> Figure:
    df_filtered = filter_data(df, config.filters).copy()
    df_filtered, config = transform(df_filtered, config)

    chart_type = config.chart_type

    if chart_type in [ChartType.BAR, ChartType.SCALAR]:
        agg = groupby_agg(df_filtered, config)
        x = agg.columns[0]
        y = agg.columns[-1]
        orientation = "v"
        bar_mode = "group" if config.bar_mode == BarMode.GROUP else "relative"

        if config.horizontal:
            x, y = y, x
            orientation = "h"

        fig = px.bar(
            agg,
            color=config.color or None,
            orientation=orientation,
            barmode=bar_mode,
            **_ax_config(config, x, y),
        )
    elif chart_type == ChartType.SCATTER:
        assert config.x is not None
        fig = px.scatter(
            df_filtered,
            color=config.color or None,
            **_ax_config(config, config.x.column, config.y.column),
        )
    elif chart_type == ChartType.PIE:
        agg = groupby_agg(df_filtered, config)
        fig = px.pie(agg, names=agg.columns[0], values=agg.columns[-1])
    elif chart_type in [ChartType.LINE, ChartType.AREA]:
        func_table = {ChartType.LINE: px.line, ChartType.AREA: px.area}

        if is_aggregation(config):
            agg = groupby_agg(df_filtered, config)
            fig = func_table[chart_type](
                agg,
                color=config.color or None,
                **_ax_config(config, agg.columns[0], y=agg.columns[-1]),
            )
        else:
            assert config.x is not None
            fig = func_table[chart_type](
                df_filtered,
                color=config.color or None,
                **_ax_config(config, config.x.column, config.y.column),
            )
    else:
        raise ValueError(f"Unknown chart_type: {chart_type}")

    if show:
        fig.show()

    return fig


def draw_altair(
    df: pd.DataFrame,
    config: dict[str, Any],
    show: bool = True,
    use_vega_fusion: bool = True,
) -> alt.Chart:
    if use_vega_fusion:
        vf.enable()
    spec = copy.deepcopy(config)
    spec["data"] = to_values(df)
    chart = alt.Chart.from_dict(spec)
    if show:
        chart.show()

    return chart


def _is_datetime_like_column(s: pd.Series) -> bool:
    if is_numeric_dtype(s):
        return False
    try:
        pd.to_datetime(s)
        return True
    except Exception:
        try:
            pd.to_datetime(s, dayfirst=True)
            return True
        except Exception:
            return False


def groupby_agg(df: pd.DataFrame, config: PlotConfig) -> pd.DataFrame:
    group_by = [config.x.column] if config.x is not None else []

    if config.color and (not config.x or (config.color != config.x.column)):
        group_by.append(config.color)

    agg_method = {
        AggregationType.AVG: "mean",
        AggregationType.SUM: "sum",
        AggregationType.COUNT: "count",
        AggregationType.DISTINCT_COUNT: "nunique",
        AggregationType.MIN: "min",
        AggregationType.MAX: "max",
    }

    y = config.y
    aggregation = y.aggregation or AggregationType.AVG

    if config.x and config.x.column and _is_datetime_like_column(df[config.x.column]):
        df = df.copy()
        df[config.x.column] = pd.to_datetime(df[config.x.column])

    def _apply_agg(
        aggregation: AggregationType, df: pd.DataFrame | DataFrameGroupBy
    ) -> Any:
        if aggregation == AggregationType.COUNTROWS:
            return len(df) if isinstance(df, pd.DataFrame) else df.size()
        else:
            return df[y.column].agg(agg_method[aggregation])

    if not group_by:
        return pd.DataFrame({y.transformed_name(): [_apply_agg(aggregation, df)]})
    else:
        agg = _apply_agg(aggregation, df.groupby(group_by, dropna=False)).rename(y.transformed_name())  # type: ignore
        ascending = config.sort_order == SortOrder.ASC

        if config.sort_criteria == SortingCriteria.VALUE:
            agg = agg.sort_values(ascending=ascending)
        else:
            agg = agg.sort_index(ascending=ascending, level=0)

        if config.limit:
            agg = agg.iloc[: config.limit]

        return agg.reset_index()


def is_aggregation(config: PlotConfig) -> bool:
    return config.y.aggregation is not None


def filter_data(df: pd.DataFrame, filters: list[str]) -> pd.DataFrame:
    if not filters:
        return df

    def _filter_data(
        df: pd.DataFrame, filters: list[str], with_escape: bool
    ) -> pd.DataFrame:
        if with_escape:
            return df.query(
                " and ".join([Filter.parse_from_llm(f).escaped() for f in filters])
            )
        else:
            return df.query(" and ".join(filters))

    # 1. LLM sometimes forgets to escape column names when necessary.
    #    In this case, adding escaping will handle it correctly.
    # 2. LLM sometimes writes multiple OR conditions in one filter.
    #    In this case, adding escapes leads to errors.
    # Since both cases exist, add escapes and retry only when an error occurs.
    try:
        return _filter_data(df, filters, False)
    except Exception:
        return _filter_data(df, filters, True)

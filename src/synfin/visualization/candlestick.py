"""Candlestick chart generation using mplfinance."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import mplfinance as mpf
    _HAS_MPF = True
except ImportError:
    _HAS_MPF = False
    logger.warning("mplfinance not installed. Candlestick charts unavailable.")


def plot_candlestick(
    df: pd.DataFrame,
    title: str = "Synthetic OHLCV",
    style: str = "charles",
    savepath: Optional[Union[str, Path]] = None,
    show: bool = True,
    volume: bool = True,
) -> None:
    """Generate a candlestick chart from OHLCV DataFrame.

    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns and DatetimeIndex.
        title: Chart title.
        style: mplfinance style (e.g., "charles", "yahoo", "nightclouds").
        savepath: Optional path to save the chart image.
        show: Whether to display the chart.
        volume: Whether to include volume panel.
    """
    if not _HAS_MPF:
        logger.error("mplfinance required for candlestick charts. Install: pip install mplfinance")
        return

    kwargs = dict(
        type="candle",
        style=style,
        title=title,
        volume=volume,
        warn_too_much_data=1000,
    )
    if savepath:
        kwargs["savefig"] = str(savepath)
    if not show:
        kwargs["show_nontrading"] = False

    mpf.plot(df, **kwargs)
    if savepath:
        logger.info("Saved candlestick chart to %s", savepath)

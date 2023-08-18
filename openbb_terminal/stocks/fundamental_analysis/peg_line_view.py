""" PEG Line View """
__docformat__ = "numpy"

import logging
import os
from typing import Optional, Union
import pandas as pd

from openbb_terminal import OpenBBFigure, theme
from openbb_terminal.decorators import log_start_end
from openbb_terminal.helper_funcs import (
    export_data,
    print_rich_table,
)
from openbb_terminal.rich_config import console

from openbb_terminal.stocks import stocks_helper
from openbb_terminal.forex import forex_helper
from openbb_terminal.forex import sdk_helpers
from openbb_terminal.common.technical_analysis import ta_helpers

from openbb_terminal.stocks.fundamental_analysis import av_model
from openbb_terminal.stocks.fundamental_analysis import business_insider_model
from openbb_terminal.stocks.fundamental_analysis import finnhub_model
from openbb_terminal.stocks.fundamental_analysis import fmp_model
from openbb_terminal.stocks.fundamental_analysis import seeking_alpha_model
from openbb_terminal.stocks.fundamental_analysis import stock_analysis_model
from openbb_terminal.stocks.fundamental_analysis import yahoo_finance_model

logger = logging.getLogger(__name__)


@log_start_end(log=logger)
def display_peg_line(
    symbol: str,
    data: Optional[pd.DataFrame] = None,
    currency: Optional[str] = None,
    start_date: Optional[str] = None,
    source_earnings: str = "AlphaVantage",
    source_estimates: str = "SeekingAlpha",
    raw: bool = False,
    export: str = "",
    sheet_name: Optional[str] = None
) -> Union[OpenBBFigure, None]:
    """Display the price to earnings growth line for a given stock. [Source: Multiple]

    Parameters
    ----------
    symbol: str
        Stock ticker symbol
    data: pd.DataFrame
        Stock dataframe
    currency : str
        Optionally specify the currency to return results in.
    start_date : Optional[str]
        Start date of the stock data, format YYYY-MM-DD
    source_earnings : str
        The selected data provider for historical earnings.
    source_estimates: str
        The selected data provider for future estimates.
    raw: bool
        Display raw data only
    sheet_name: str
        Optionally specify the name of the sheet the data is exported to.
    export: str
        Export dataframe data to csv,json,xlsx file
    external_axes: bool, optional
        Whether to return the figure object or not, by default False

    Examples
    --------
    >>> from openbb_terminal.sdk import openbb
    >>> openbb.stocks.fa.peg_chart(symbol="AAPL")

    Notes
    -----
    The model offers the following data source provider options:
    - Historical EPS [Source: AlphaVantage, FinancialModelingPrep, Finnhub]
    - Estimated EPS [Source: SeekingAlpha, BusinessInsider, StockAnalysis]
    - Forex Rates [Source: YahooFinance]
    """
    date_col = "Date"
    eps_col = "Actual EPS"
    est_col = "Estimated EPS"
    est_low_col = "Estimated Low"
    est_high_col = "Estimated High"
    ttm_col = "TTM EPS"
    growth_col = "Growth Rate"
    peg_col = "Price at P/E=G"
    peg_est_col = "Forecast at P/E=G"
    peg_low_col = "Low at P/E=G"
    peg_high_col = "High at P/E=G"

    fig = OpenBBFigure(yaxis_title="Price").set_title(
        f"{symbol} (Time Series) and PEG Line Forecast"
    )

    reported_currency = yahoo_finance_model.get_currency(symbol)
    if len(reported_currency) != 3:
        console.print(f"Warning: Unable to identify the currency unit. Assuming USD.")
        reported_currency = "USD"
    requested_currency = currency if currency else reported_currency

    overview = av_model.get_overview(symbol)
    div_yield = overview.loc["DividendYield"][0]
    year_end = overview.loc["FiscalYearEnd"][0]
    temp_date = pd.to_datetime(year_end, format="%B")
    month_end = temp_date.month
    day_end = (temp_date + pd.offsets.MonthEnd(0)).day
    last_qtr = overview.loc["LatestQuarter"][0]

    # prep historical eps
    eps_df = pd.DataFrame()
    if source_earnings == "AlphaVantage":
        eps_df = av_model.get_earnings(symbol, quarterly=True)
        renames = {
            "Fiscal Date Ending": date_col,
            "Reported EPS": eps_col
        }
    elif source_earnings == "FinancialModelingPrep":
        eps_df = fmp_model.get_earnings_surprises(symbol)
        renames = {
            "date": date_col,
            "actualEarningResult": eps_col,
            "estimatedEarning": est_col
        }
    elif source_earnings == "Finnhub":
        eps_df = finnhub_model.get_earnings_surprises(symbol)
        renames = {
            "period": date_col,
            "actual": eps_col,
            "estimate": est_col
        }
    if eps_df.empty:
        console.print(f"Earnings history not found for symbol: {symbol}")
        return None
    else:
        eps_df = eps_df.rename(columns=renames)
        # NOTE: fixes date alignment (first row is latest and updates backward)
        # TODO: identify freq for the rare cases eps is not reported quarterly
        eps_df[date_col] = pd.date_range(last_qtr, periods=eps_df.shape[0], freq="-3M")
        eps_df[date_col] = eps_df[date_col].astype("str")
        eps_df = eps_df.loc[::-1, [date_col, eps_col, est_col]]
        eps_df[eps_col] = pd.to_numeric(eps_df[eps_col], errors="coerce")
        eps_df[est_col] = pd.to_numeric(eps_df[est_col], errors="coerce")
        eps_df[ttm_col] = eps_df[eps_col].rolling(4).sum()
        eps_df = eps_df.loc[:, [date_col, ttm_col]]

    # convert historical eps currency
    if requested_currency != reported_currency:
        start_date = eps_df[date_col].min()
        rates_df = forex_helper.load(
            to_symbol=requested_currency,
            from_symbol=reported_currency,
            start_date=start_date,
            interval="1mo"
        )
        rates_df.index.name = date_col
        rates_df.index = rates_df.index + pd.offsets.MonthEnd(0)
        rates_df = rates_df.drop_duplicates().loc[:, "Close"]
        eps_df = eps_df.merge(rates_df, how="left", on="Date")
        eps_df[eps_col] = eps_df[eps_col] * eps_df["Close"]
        eps_df[est_col] = eps_df[est_col] * eps_df["Close"]
        eps_df[ttm_col] = eps_df[ttm_col] * eps_df["Close"]
        eps_df.drop(columns="Close", inplace=True)

    # prep estimated eps
    est_df = pd.DataFrame()
    if source_estimates == "BusinessInsider":
        est_df = business_insider_model.get_estimates(symbol)[0]
        est_df.index.name = None
        est_df = est_df.transpose().reset_index()
        renames = {
            "index": date_col,
            "EPS": est_col
        }
    elif source_estimates == "SeekingAlpha":
        est_df = seeking_alpha_model.get_estimates_eps(symbol)
        renames = {
            "fiscalyear": date_col,
            "actual": eps_col,
            "consensus_mean": est_col,
            "consensus_low": est_high_col,
            "consensus_high": est_low_col
        }
    elif source_estimates == "StockAnalysis":
        est_df = stock_analysis_model.get_estimates(symbol)
        est_df = est_df.transpose().drop("EPS", axis=0).reset_index()
        renames = {
            "index": date_col,
            0: est_high_col,
            1: est_col,
            2: est_low_col
        }
    if est_df.empty:
        console.print(f"Earnings estimates not found for symbol: {symbol}")
        return fig.show(external=False)
    else:
        if source_estimates == "BusinessInsider":
            est_df[est_col] = est_df[est_col].str.replace("[^.0-9]", "", regex=True) 
            est_df = est_df[est_df[date_col].notna()]
            est_df[est_low_col] = pd.to_numeric(None)
            est_df[est_high_col] = pd.to_numeric(None)
        if source_estimates == "SeekingAlpha":
            est_df = est_df[est_df["actual"] == 0].head(5)
        est_df = est_df.rename(columns=renames)
        est_df[date_col] = est_df[date_col].astype("str") + f"-{month_end}-{day_end}"
        est_df = est_df.loc[:, [date_col, est_col, est_low_col, est_high_col]]

        # convert estimated eps currency
        # NOTE: these estimates are always provided in USD
        if source_estimates in ["BusinessInsider", "Seeking Alpha"]:
            reported_currency = "USD"
        if requested_currency != reported_currency:
            est_df["Close"] = sdk_helpers.quote(f"USD{requested_currency}").loc["Quote", 0]
            est_df[est_col] = est_df[est_col] * est_df["Close"]
            est_df[est_low_col] = est_df[est_low_col] * est_df["Close"]
            est_df[est_high_col] = est_df[est_high_col] * est_df["Close"]
            est_df.drop(columns="Close", inplace=True)

    # join earnings and estimates
    peg_df = pd.concat([eps_df, est_df], ignore_index=True)
    peg_df.drop_duplicates(subset="Date", inplace=True)
    peg_df[date_col] = pd.to_datetime(peg_df[date_col])
    peg_df.set_index("Date", inplace=True)

    # TODO : calculate historical growth rates

    # calculate future growth rate
    # TODO: consider using complex growth rate (next-last)/last.abs
    growth_df = peg_df.loc[(peg_df.index.month == month_end) & (peg_df.index.day == day_end), :]
    growth_df = growth_df.tail(est_df.shape[0] + 1)
    start = growth_df[ttm_col][0]
    end = growth_df[est_col][-1]
    # NOTE: a P/E of 15 is considered fair even when growth is slower
    rate = max(100 * (pow(end / start, 1 / est_df.shape[0]) - 1), 15)

    peg_df[growth_col] = rate
    peg_df[peg_col] = peg_df[ttm_col] * peg_df[growth_col]
    peg_df[peg_est_col] = peg_df[est_col] * peg_df[growth_col]
    peg_df[peg_low_col] = peg_df[est_low_col] * peg_df[growth_col]
    peg_df[peg_high_col] = peg_df[est_high_col] * peg_df[growth_col]
    i = peg_df.index[eps_df.shape[0] - 1]
    peg_df.at[i, peg_est_col] = peg_df.at[i, peg_col]
    peg_df.at[i, peg_low_col] = peg_df.at[i, peg_col]
    peg_df.at[i, peg_high_col] = peg_df.at[i, peg_col]

    if not data:
        if not start_date:
            start_date = eps_df[date_col].min()
        data = stocks_helper.load(symbol=symbol, start_date=start_date)
        close_col = ta_helpers.check_columns(data, False, False, True)
    data.index.name = date_col

    if start_date:
        data = data[start_date:]
        peg_df = peg_df[start_date:]

    fig.add_scatter(
        x=data.index,
        y=data[close_col].values,
        name="Close",
        line_width=2,
    )

    fig.add_scatter(
        x=peg_df.index,
        y=peg_df[peg_col].values,
        name=peg_col,
        line_width=2,
        fill="tozeroy",
        opacity=0.95
    )

    fig.add_scatter(
        x=peg_df.index,
        y=peg_df[peg_est_col].values,
        name=peg_est_col,
        line_width=2,
        #fill="tonexty",
        #opacity=0.95
    )

    fig.add_scatter(
        x=peg_df.index,
        y=peg_df[peg_low_col].values,
        name=peg_low_col,
        line_width=2,
        fill="tonext",
        opacity=0.95
    )

    fig.add_scatter(
        x=peg_df.index,
        y=peg_df[peg_high_col].values,
        name=peg_high_col,
        line_width=2,
        fill="tonext",
        opacity=0.95
    )

    export_data(
        export,
        os.path.dirname(os.path.abspath(__file__)),
        "peg",
        peg_df,
        sheet_name,
        fig,
    )

    if raw:
        peg_df.index = peg_df.index.date
        return print_rich_table(
            peg_df,
            title=f"{symbol} (Time Series) and PEG Line Forecast",
            headers=list(peg_df.columns),
            show_index=True,
            export=bool(export)
        )

    return fig.show(external=False)

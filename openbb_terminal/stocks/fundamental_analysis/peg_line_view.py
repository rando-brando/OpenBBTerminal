""" PEG Line View """
__docformat__ = "numpy"

import logging
import os
from typing import Optional, Union
import pandas as pd
import yfinance as yf

from openbb_terminal import OpenBBFigure, theme
from openbb_terminal.rich_config import console
from openbb_terminal.decorators import log_start_end
from openbb_terminal.helper_funcs import (
    export_data,
    print_rich_table,
)

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

logger = logging.getLogger(__name__)


@log_start_end(log=logger)
def display_peg_line(
    symbol: str,
    data: Optional[pd.DataFrame] = None,
    start_date: Optional[str] = None,
    growth_rate: Optional[str] = None,
    source_earnings: str = "AlphaVantage",
    source_estimates: str = "SeekingAlpha",
    raw: bool = False,
    export: str = "",
    sheet_name: Optional[str] = None
) -> Union[OpenBBFigure, None]:
    """Display the price/earnings growth line for a given stock. [Source: Multiple]

    Parameters
    ----------
    symbol: str
        Stock ticker symbol
    data: pd.DataFrame
        Stock dataframe
    start_date : Optional[str]
        Start date of the stock data, format YYYY-MM-DD
    growth_rate : Optional[str]
        Optionally specify a custom growth rate as a percent.
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
    >>> openbb.stocks.fa.peg_chart(symbol="AAPL", growth_rate=22)

    Notes
    -----
    The model offers the following data source provider options:
    - Historical EPS [Source: AlphaVantage, FinancialModelingPrep, Finnhub]
    - Estimated EPS [Source: SeekingAlpha, BusinessInsider, Finnhub, StockAnalysis]
    - Forex Rates [Source: YahooFinance]
    """
    title = f"{symbol} Price/Earnings Growth Forecast"
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
    ror_col = "Rate of Return"

    # Get required inputs from free provider to avoid wasteful API calls.
    # NOTE: These inputs ensure that the data are aligned correctly.
    try:
        stock = yf.Ticker(symbol)
        currency = stock.fast_info["currency"]
        reported_currency = stock.info["financialCurrency"]
        dates = stock.get_income_stmt(freq="quarterly").columns
        last_qtr = dates.max()
        year_end = dates[dates.month == dates.month.max()].max()
        month_end = year_end.month
        day_end = year_end.day
    # Fallback on premium endpoints if free provider stops working.
    except Exception:
        overview = av_model.get_overview(symbol)
        last_qtr = overview.loc["LatestQuarter"][0]
        temp_date = pd.to_datetime(overview.loc["FiscalYearEnd"][0], format="%B")
        month_end = temp_date.month
        day_end = (temp_date + pd.offsets.MonthEnd(0)).day
        if source_earnings == "AlphaVantage":
            income = av_model.get_income_statements(symbol, quarterly=True)
            currency = overview.loc["Currency"][0]
            reported_currency = income.loc["reportedCurrency"][0]
        elif source_earnings == "FinancialModelingPrep":
            overview = fmp_model.get_profile(symbol)
            income = fmp_model.get_income(symbol)
            currency = overview.loc["currency"][0]
            reported_currency = income.loc["Reported currency"][0]
        elif source_earnings == "Finnhub":
            overview = finnhub_model.get_profile2(symbol)
            currency = overview.loc["Currency"][0]
            reported_currency = overview.loc["estimateCurrency"][0]

    # prep historical eps
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
        eps_df = finnhub_model.get_earnings(symbol, quarterly=True)
        renames = {
            "period": date_col,
            "v": eps_col
        }
    if eps_df.empty:
        return None
    else:
        eps_df = eps_df.rename(columns=renames)
        # NOTE: fixes date alignment (first row is latest and updates backward)
        # TODO: override freq for the rare cases eps is not reported quarterly
        eps_df[date_col] = pd.date_range(last_qtr, periods=eps_df.shape[0], freq="-3M")
        eps_df[date_col] = eps_df[date_col].astype("str")
        eps_df = eps_df.loc[::-1, [date_col, eps_col]]
        eps_df[eps_col] = pd.to_numeric(eps_df[eps_col], errors="coerce")
        eps_df[ttm_col] = eps_df[eps_col].rolling(4).sum()
        eps_df = eps_df[eps_df[ttm_col].notna()]
        eps_df = eps_df.loc[:, [date_col, ttm_col]]

    # convert historical eps currency
    if currency != reported_currency:
        if not start_date:
            start_date = eps_df[date_col].min()
        rates_df = forex_helper.load(
            to_symbol=currency,
            from_symbol=reported_currency,
            start_date=start_date,
            interval="1mo"
        )
        rates_df.index.name = date_col
        rates_df.index = rates_df.index + pd.offsets.MonthEnd(0)
        rates_df.index = rates_df.index.astype("str")
        rates_df = rates_df.drop_duplicates().loc[:, "Close"]
        eps_df = eps_df.merge(rates_df, how="left", on="Date")
        eps_df[ttm_col] = eps_df[ttm_col] * eps_df["Close"]
        eps_df.drop(columns="Close", inplace=True)

    # calculate historical growth rates
    prior = eps_df[ttm_col].shift(1)
    eps_df[growth_col] = 100 * (eps_df[ttm_col] - prior) / prior.abs()
    eps_df[growth_col] = eps_df[growth_col].apply(lambda x: max(x, 15)).round(2)

    # prep estimated eps
    if source_estimates == "BusinessInsider":
        est_df = business_insider_model.get_estimates(symbol)[0]
        est_df.index.name = None
        est_df = est_df.transpose().reset_index()
        renames = {
            "index": date_col,
            "EPS": est_col
        }
    elif source_estimates == "Finnhub":
        est_df = finnhub_model.get_estimates(symbol, quarterly=True)
        renames = {
            "period": date_col,
            "epsAvg": est_col,
            "epsLow": est_low_col,
            "epsHigh": est_high_col
        }
    elif source_estimates == "SeekingAlpha":
        est_df = seeking_alpha_model.get_estimates_eps(symbol)
        renames = {
            "fiscalyear": date_col,
            "actual": eps_col,
            "consensus_mean": est_col,
            "consensus_low": est_low_col,
            "consensus_high": est_high_col
        }
    elif source_estimates == "StockAnalysis":
        est_df = stock_analysis_model.get_eps_forecast(symbol)
        est_df = est_df.reset_index()
        renames = {
            "index": date_col,
            "High": est_high_col,
            "Avg": est_col,
            "Low": est_low_col
        }
    if est_df.empty:
        return None
    else:
        est_df = est_df.rename(columns=renames)
        est_df = est_df.loc[est_df[date_col].notna()]
        if source_estimates != "Finnhub":
            est_df[date_col] = est_df[date_col].astype("str") + f"-{month_end}-{day_end}"
        if eps_col in est_df.columns:
            est_df = est_df[(est_df[eps_col] == 0) | est_df[eps_col].isna()]
        for col in [est_col, est_low_col, est_high_col]:
            if col not in est_df.columns:
                est_df[col] = pd.to_numeric(None)
            if est_df[col].dtype != "float":
                est_df[col] = est_df[col].str.replace("[^.0-9]", "", regex=True)
                est_df[col] = pd.to_numeric(est_df[col], errors="coerce")
        est_df = est_df.loc[:, [date_col, est_col, est_low_col, est_high_col]].head(5)

    # convert estimated eps currency
    # NOTE: these estimates are always provided in USD
    if source_estimates in ["BusinessInsider", "Seeking Alpha"]:
        reported_currency = "USD"
    if currency != reported_currency:
        est_df["Close"] = sdk_helpers.quote(f"{reported_currency}USD").loc["Quote", 0]
        est_df[est_col] = est_df[est_col] * est_df["Close"]
        est_df[est_low_col] = est_df[est_low_col] * est_df["Close"]
        est_df[est_high_col] = est_df[est_high_col] * est_df["Close"]
        est_df.drop(columns="Close", inplace=True)

    # calculate future growth rate
    num_years = est_df.shape[0]
    if not growth_rate:
        dates = pd.to_datetime(eps_df[date_col])
        year_end = (dates.dt.month == month_end) & (dates.dt.day == day_end)
        start = eps_df.loc[year_end, ttm_col].iloc[-1]
        end = est_df[est_col].iloc[-1]
        growth_rate = 100 * (pow(end / start, 1 / num_years) - 1)
    growth_rate = round(max(growth_rate, 15), 2)
    est_df[growth_col] = growth_rate

    # calculate PEG line
    peg_df = pd.concat([eps_df, est_df], ignore_index=True)
    peg_df.drop_duplicates(subset="Date", inplace=True)
    peg_df[date_col] = pd.to_datetime(peg_df[date_col])
    peg_df.set_index("Date", inplace=True)
    peg_df[peg_col] = peg_df[ttm_col].apply(lambda x: max(x, 0)) * peg_df[growth_col]
    peg_df[peg_est_col] = peg_df[est_col].apply(lambda x: max(x, 0)) * peg_df[growth_col]
    peg_df[peg_low_col] = peg_df[est_low_col].apply(lambda x: max(x, 0)) * peg_df[growth_col]
    peg_df[peg_high_col] = peg_df[est_high_col].apply(lambda x: max(x, 0)) * peg_df[growth_col]
    i = peg_df.index[eps_df.shape[0] - 1]
    peg_df.at[i, peg_est_col] = peg_df.at[i, peg_col]
    peg_df.at[i, peg_low_col] = peg_df.at[i, peg_col]
    peg_df.at[i, peg_high_col] = peg_df.at[i, peg_col]

    # load price data
    if not start_date:
        start_date = peg_df.index.min()
    if not data:
        data = stocks_helper.load(symbol=symbol, start_date=start_date)
    close_col = ta_helpers.check_columns(data, False, False, True)
    data = data[start_date:]
    peg_df = peg_df[start_date:]

    # create rate of return line
    ror_df = pd.DataFrame(data.tail(1)[close_col]).rename(columns={close_col: ror_col})
    ror_df.at[peg_df.index[-1], ror_col] = peg_df[peg_est_col][-1]
    ror = round(100 * (pow(ror_df[ror_col][1] / ror_df[ror_col][0], 1 / num_years) - 1), 2)

    fig = OpenBBFigure(title=title, yaxis_title=f"Price ({reported_currency})")

    fig.add_scatter(
        x=peg_df.index,
        y=peg_df[peg_col].values,
        name=peg_col,
        line_width=2,
        line_color="#ef7d00",
        fill="tozeroy"
    )

    fig.add_scatter(
        x=peg_df.index,
        y=peg_df[peg_est_col].values,
        name=peg_est_col,
        line_width=2,
        line_color="#9b30d9",
        line_dash="dash",
        fill="tozeroy"
    )    

    fig.add_scatter(
        x=peg_df.index,
        y=peg_df[peg_high_col].values,
        name=peg_high_col,
        line_width=2,
        line_color="#5f00af",
        line_dash="dash",
        fill="tonexty"
    )

    fig.add_scatter(
        x=peg_df.index,
        y=peg_df[peg_low_col].values,
        name=peg_low_col,
        line_width=2,
        line_color="#af87ff",
        line_dash="dash"
    )

    fig.add_scatter(
        x=data.index,
        y=data[close_col].values,
        name="Close",
        line_width=2,
        line_color="#ffed00"
    )

    fig.add_scatter(
        x=ror_df.index,
        y=ror_df[ror_col].values,
        name=f"Annual {ror_col}: {str(ror)}%",
        line_width=2,
        line_color="white",
        line_dash="dash"
    )

    fig.add_annotation(
        text=f"Estimated Annual Growth (G): {growth_rate}%",
        font_color="white",
        showarrow=False,
        xanchor="center",
        yanchor="top",
        xref="paper",
        yref="paper",
        y=1,
        bordercolor="white",
        borderwidth=1
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
            title=title,
            headers=list(peg_df.columns),
            show_index=True,
            export=bool(export)
        )

    return fig.show(external=False)

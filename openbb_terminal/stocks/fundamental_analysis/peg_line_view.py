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
    dividend_col = "Dividends"
    yield_col = "Dividend Yield"
    ttm_col = "TTM EPS"
    growth_col = "Growth Rate"
    pe_col = "P/E Ratio"
    norm_col = "Price at 5YMA P/E"
    peg_col = "Price at P/E=G"
    peg_est_col = "Forecast at P/E=G"
    peg_low_col = "Low at P/E=G"
    peg_high_col = "High at P/E=G"
    ror_col = "Rate of Return"

    # prep price data
    today = pd.to_datetime("today")
    if not start_date:
        start_date = today.replace(year=today.year-30).strftime("%Y-%m-01")
    if not data:
        data = stocks_helper.load(symbol=symbol, start_date=start_date)
    if dividend_col not in data.columns:
        data[dividend_col] = 0
    close_col = ta_helpers.check_columns(data, high=False, low=False, close=True)
    data.index.name = date_col

    # NOTE: These inputs ensure that the data are aligned correctly.
    # Get required inputs from free provider to avoid wasteful API calls.
    try:
        stock = yf.Ticker(symbol)
        currency = stock.fast_info["currency"]
        reported_currency = stock.info["financialCurrency"]
        income = stock.get_income_stmt(freq="quarterly")
        dates = income.columns
        last_qtr = dates.max()
        year_end = dates[dates.month == dates.month.max()].max()
        month_end = year_end.month
        day_end = year_end.day
    # Fallback on premium endpoints if free provider stops working.
    except Exception:
        overview = av_model.get_overview(symbol)
        if overview.empty:
            return None
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
        # TODO: override freq for the rare cases eps is not reported quarterly
        eps_df[date_col] = pd.date_range(last_qtr, periods=eps_df.shape[0], freq="-3M")
        eps_df = eps_df.loc[::-1, [date_col, eps_col]]
        eps_df[eps_col] = pd.to_numeric(eps_df[eps_col], errors="coerce")
        eps_df[ttm_col] = eps_df[eps_col].rolling(4).sum()
        eps_df = eps_df[eps_df[ttm_col].notna()]
        eps_df[ttm_col] = eps_df[ttm_col].mask(eps_df[ttm_col].abs() < 0.01, pd.NA)
        eps_df = eps_df[[date_col, ttm_col]]

    # NOTE: these earnings are always provided in USD
    if source_earnings in ["FinancialModelingPrep"]:
        reported_currency = "USD"
    # convert historical eps currency
    if currency != reported_currency:
        rates_df = forex_helper.load(
            to_symbol=currency,
            from_symbol=reported_currency,
            start_date=eps_df[date_col].min().strftime("%Y-%m-01"),
            interval="1mo"
        )
        rates_df.index.name = date_col
        rates_df.index = rates_df.index + pd.offsets.MonthEnd(0)
        rates_df = rates_df.drop_duplicates().loc[:, "Close"]
        eps_df = eps_df.merge(rates_df, how="left", on=date_col)
        eps_df[ttm_col] = eps_df[ttm_col] * eps_df["Close"]
        eps_df.drop(columns="Close", inplace=True)

    # calculate historical growth rates
    num_years = int(eps_df.shape[0] / 4)
    rates_df = eps_df.copy()
    for y in range(min(5, num_years), min(15, num_years) + 1):
        initial = rates_df[ttm_col].shift(y * 4)
        final = rates_df[ttm_col]
        ratio = (final - initial + initial.abs()) / initial.abs()
        rates_df[f"{y}YCAGR"] = ratio.pow(1/y).bfill() - 1
    rates_df = rates_df.filter(regex="CAGR").iloc[:, ::-1]
    eps_df[growth_col] = rates_df.ewm(span=rates_df.shape[1], axis=1).mean().iloc[:, -1]
    eps_df[growth_col] = eps_df[growth_col].rolling(4).mean().shift(-1).fillna(eps_df[growth_col])
    eps_df[growth_col] = eps_df[growth_col].apply(lambda g: min(0.15, 0.085 + 2 * g) if g < 0.15 else g)
    eps_df[growth_col] = eps_df[growth_col].mask(eps_df[growth_col] < 0, 0)
    eps_df[growth_col] = 100 * eps_df[growth_col].round(4)

    # calculate historical p/e ratios
    pe_df = data.merge(eps_df, how="outer", on=date_col, sort=True)
    pe_df.loc[pe_df[ttm_col].isna(), date_col] = pd.to_datetime("")
    pe_df[date_col] = pe_df[date_col].bfill()
    aggs = {close_col: "mean", ttm_col: "last", dividend_col: "sum"}
    pe_df = pe_df.groupby(date_col).agg(aggs)
    pe_df[pe_col] = pe_df[close_col] / pe_df[ttm_col]
    pe_df[pe_col] = pe_df[pe_col].mask(pe_df[pe_col] < 0, 0)
    pe_df[yield_col] = pe_df[dividend_col] / pe_df[close_col]
    pe_df = pe_df[[pe_col, dividend_col, yield_col]]

    # calculate historical peg ratios
    eps_df = eps_df.merge(pe_df, how="left", on=date_col, sort=True)
    abs_ttm = eps_df[ttm_col].mask(eps_df[ttm_col] < 0, 0)
    eps_df[peg_col] = abs_ttm * eps_df[growth_col]
    eps_df[norm_col] = abs_ttm * eps_df[pe_col].ewm(span=20).mean()
    eps_df[norm_col] = eps_df[norm_col].round(2)
    eps_df = eps_df.set_index(date_col).loc[::-1]

    # create plot
    fig = OpenBBFigure(title=title, yaxis_title=f"Price ({currency})")

    # plot historical peg line
    fig.add_scatter(
        x=eps_df.index,
        y=eps_df[peg_col].values,
        name=peg_col,
        line_width=2,
        line_color="#ef7d00",
        fill="tozeroy"
    )

    # plot historical price line
    fig.add_scatter(
        x=data.index,
        y=data[close_col].values,
        name="Close",
        line_width=2,
        line_color="#ffed00"
    )

    # plot normal peg line
    fig.add_scatter(
        x=eps_df.index,
        y=eps_df[norm_col].values,
        name=norm_col,
        line_width=2,
        line_color="#00aaff",
        line_dash="dash"
    )

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
        return fig.show(external=False)
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
            est_df[col] = est_df[col].mask(est_df[col] < 0.01, 0.01)
        est_df = est_df.loc[:, [date_col, est_col, est_low_col, est_high_col]].head(5)

    # NOTE: these estimates are always provided in USD
    if source_estimates in ["BusinessInsider", "SeekingAlpha"]:
        reported_currency = "USD"
    # convert estimated eps currency
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
        estimates = eps_df.loc[year_end, ttm_col].append(est_df[est_col])
        estimates = (estimates - estimates.shift(1)) / estimates.shift(1).abs()
        growth_rate = round(100 * estimates[::-1].ewm(span=5).mean().iloc[-1], 2)
    fair_rate = max(growth_rate, 15)
    est_df[growth_col] = fair_rate

    # calculate PEG line
    peg_df = pd.concat([eps_df, est_df], ignore_index=True)
    peg_df.drop_duplicates(subset="Date", inplace=True)
    peg_df[date_col] = pd.to_datetime(peg_df[date_col])
    peg_df.set_index("Date", inplace=True)
    peg_df[peg_col] = peg_df[ttm_col] * peg_df[growth_col]
    peg_df[peg_est_col] = peg_df[est_col] * peg_df[growth_col]
    peg_df[peg_low_col] = peg_df[est_low_col] * peg_df[growth_col]
    peg_df[peg_high_col] = peg_df[est_high_col] * peg_df[growth_col]
    i = peg_df.index[eps_df.shape[0] - 1]
    peg_df.at[i, peg_est_col] = peg_df.at[i, peg_col]
    peg_df.at[i, peg_low_col] = peg_df.at[i, peg_col]
    peg_df.at[i, peg_high_col] = peg_df.at[i, peg_col]

    # create rate of return line
    ror_df = pd.DataFrame(data.tail(1)[close_col]).rename(columns={close_col: ror_col})
    ror_df.at[peg_df.index[-1], ror_col] = peg_df[peg_est_col][-1]
    ror = round(100 * (pow(ror_df[ror_col][1] / ror_df[ror_col][0], 1 / num_years) - 1), 2)

    # plot estimated peg line (avg)
    fig.add_scatter(
        x=peg_df.index,
        y=peg_df[peg_est_col].values,
        name=peg_est_col,
        line_width=2,
        line_color="#9b30d9",
        line_dash="dash",
        fill="tozeroy"
    )    

    # plot estimated peg line (high)
    fig.add_scatter(
        x=peg_df.index,
        y=peg_df[peg_high_col].values,
        name=peg_high_col,
        line_width=2,
        line_color="#5f00af",
        line_dash="dash",
        fill="tonexty"
    )

    # plot estimated peg line (low)
    fig.add_scatter(
        x=peg_df.index,
        y=peg_df[peg_low_col].values,
        name=peg_low_col,
        line_width=2,
        line_color="#af87ff",
        line_dash="dash"
    )

    # plot rate of return line
    fig.add_scatter(
        x=ror_df.index,
        y=ror_df[ror_col].values,
        name=f"Annual {ror_col}: {str(ror)}%",
        line_width=2,
        line_color="white",
        line_dash="dash"
    )

    # add growth rate annotation
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

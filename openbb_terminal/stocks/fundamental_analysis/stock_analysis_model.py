""" Stock Analysis Model """
__docformat__ = "numpy"

import logging

import pandas as pd
from bs4 import BeautifulSoup

from openbb_terminal.decorators import log_start_end
from openbb_terminal.helper_funcs import request, get_user_agent
from openbb_terminal.rich_config import console

logger = logging.getLogger(__name__)


@log_start_end(log=logger)
def _get_forecast_table(symbol: str, table: str) -> pd.DataFrame:
    """Get the selected forecast table

    Parameters
    ----------
    symbol : str
        Stock ticker symbol
    table : str
        The name of the table to return

    Returns
    -------
    pd.DataFrame
        Dataframe of the selected forecast
    """
    df = pd.DataFrame()
    response = request(
        f"https://stockanalysis.com/stocks/{symbol.lower()}/forecast/",
        headers={"User-Agent": get_user_agent()}
    )
    if response.status_code == 429:
        console.print("Too many requests, please try again later")
        return df

    if response.status_code == 404 or "404 - Page Not Found" in response.text:
        console.print(f"Page not found for symbol: {symbol}")
        return df

    if response.status_code != 200:
        console.print(f"HTTP request failed with status {response.status_code}: {response.reason}")
        return df

    try:
        df = pd.read_html(response.text)[0]
    except ValueError:
        console.print(f"There was an unexpected error parsing the HTML")
        print(f"Page not found for symbol: {symbol}")
        return df

    soup = BeautifulSoup(response.content, "html.parser")
    data = soup.findAll("div", attrs={"data-test": "forecast-estimate-table"})

    if table == "Revenue Forecast":
        df = pd.read_html(str(data[0]))[0]
    elif table == "Revenue Growth":
        df = pd.read_html(str(data[1]))[0]
    elif table == "EPS Forecast":
        df = pd.read_html(str(data[2]))[0]
    elif table == "EPS Growth":
        df = pd.read_html(str(data[3]))[0]
    colnames = df.iloc[:, 0].values
    df = df.iloc[:, 1:].transpose()
    df.columns = colnames

    return df


@log_start_end(log=logger)
def get_revenue_forecast(symbol: str) -> pd.DataFrame:
    """Get the revenue forecast for ticker

    Parameters
    ----------
    symbol : str
        Stock ticker symbol

    Returns
    -------
    pd.DataFrame
        Dataframe of revenue forecasts
    """
    df = _get_forecast_table(symbol, table="Revenue Forecast")

    return df


@log_start_end(log=logger)
def get_revenue_growth(symbol: str) -> pd.DataFrame:
    """Get the revenue growth forecasts

    Parameters
    ----------
    symbol : str
        Stock ticker symbol

    Returns
    -------
    pd.DataFrame
        Dataframe of revenue growth forecasts
    """
    df = _get_forecast_table(symbol, table="Revenue Growth")

    return df


@log_start_end(log=logger)
def get_eps_forecast(symbol: str) -> pd.DataFrame:
    """Get the earnings forecast for ticker

    Parameters
    ----------
    symbol : str
        Stock ticker symbol

    Returns
    -------
    pd.DataFrame
        Dataframe of earnings forecasts
    """
    df = _get_forecast_table(symbol, table="EPS Forecast")

    return df


@log_start_end(log=logger)
def get_eps_growth(symbol: str) -> pd.DataFrame:
    """Get the earnings growth forecasts for the ticker

    Parameters
    ----------
    symbol : str
        Stock ticker symbol

    Returns
    -------
    pd.DataFrame
        Dataframe of earnings growth forecasts
    """
    df = _get_forecast_table(symbol, table="EPS Growth")

    return df

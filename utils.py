import numpy as np
import pandas as pd


def get_closing_price(data: pd.DataFrame) -> pd.Series:
    return data['c']


def get_high_price(data: pd.DataFrame) -> pd.Series:
    return data['h']


def get_low_price(data: pd.DataFrame) -> pd.Series:
    return data['l']


def minutes_to_secs(minutes: int):
    """
    this function adds the given number of minutes to a given time in unix format
    """
    return 60*minutes


def filter_by_array(arr1: np.ndarray, arr2: np.ndarray):
    """
    this method returns an array with elements that are contained in both of the sorted arrays arr1, arr2
    """
    res = arr1[np.isin(arr1, arr2)]
    return res

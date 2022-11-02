from typing import Union

import numpy as np
from datetime import datetime, timezone, timedelta

from email.message import EmailMessage
import ssl
import smtplib

from secret import EMAIL_PASSWORD

EMAIL_SENDER = 'eleinvestgroup@gmail.com'
EMAIL_RECEIVER = ['eladbeber619@gmail.com', 'elad_beber@walla.com', 'eli.yacobov1@gamil.com']


def minutes_to_secs(minutes: int):
    """
    this function adds the given number of minutes to a given time in unix format
    """
    return 60*minutes


def days_to_secs(days: int):
    return minutes_to_secs(days*60*24)


def filter_by_array(arr1: np.ndarray, arr2: np.ndarray):
    """
    this method returns an array with elements that are contained in both of the sorted arrays arr1, arr2
    """
    res = arr1[np.isin(arr1, arr2)]
    return res


def get_curr_utc_2_timestamp() -> int:
    curr_dt = datetime.now(timezone.utc) + timedelta(hours=2)
    return int(round(curr_dt.timestamp()))


def convert_timestamp_format(ts: int) -> str:
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")


def get_percent(n: int, percent: Union[float, int]) -> float:
    return n * (percent/100)


def send_email(subject=None, body=None):
    em = EmailMessage()
    em['From'] = EMAIL_SENDER
    em['To'] = EMAIL_RECEIVER
    if subject:
        em['Subject'] = subject
    if body:
        em.set_content(body)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
        smtp.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, em.as_string())

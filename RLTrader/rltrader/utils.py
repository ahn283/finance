import time
import datetime
import numpy as np

# 날짜, 시간 관련 문자열 형식
FORMAT_DATE = '%Y%m%d'
FORMAT_DATETIME = '%Y%m%d%H%M%S'

def get_today_str():
    today = datetime.datetime.combine(
        datetime.date.today(), datetime.datetime.min.time()
    )
    # RLTrader에서 날짜의 경우 %Y%m%d 형식을 사용 예: 20200202
    today_str = today.strftime('%Y%m%d')
    return today_str

def get_time_str():
    return datetime.datetime.fromtimestamp(
        int(time.time())
    ).strftime(FORMAT_DATETIME)

def sigmoid(x):
    x = max(min(x, 10), -10)
    return 1. / (1. + np.exp(-x))
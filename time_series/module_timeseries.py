# Install packages from jupyter Notebook
# !python -m pip install --user --upgrade pip
# !pip install tqdm
# !conda config --add channels conda-forge
# !conda install --yes hmmlearn
# !pip install hmmlearn

# Auto reload 
# %reload_ext autoreload
# %autoreload 2

# System related and data input controls

import os
import glob
from urllib.request import urlopen
from io import ZipFile
# os.system('pip install pandas-datareader')
# os.system('pip install missingno')
# os.system('pip install xgboost')
# os.system('pip install lightgbm')
# os.system('pip install arch')

# Ignore the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# datasets
import pandas_datareader.data as web 
from statsmodels import datasets
from sklearn import datasets

# Data manipulation, visualization, and userful functions
import numpy as np  # vectors and matrices
import pandas as pd  # tables and data manipulations
pd.options.display.float_format = '{:,.2f}'.format  # Output formatting
pd.options.display.max_rows = 10 # Limit the number of rows displayed
pd.options.display.max_columns = 20 # Limit the number of columns displayed
from patsy import dmatrix
from itertools import product   # iterative combinations
from tqdm import tqdm       # execution time
import matplotlib.pylab as plt  # plots
import matplotlib.dates as mdates
import matplotlib.mlab as mlab
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import PercentFormatter
import seaborn as sns   # more plots
import missingno as msno # plot missing values

# Modeling algorithms
# General(Statistics/Economics)
from sklearn import preprocessing
import statsmodels
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import norm

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Time series
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
import arch

# Model selection
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Evaluation metrics
# for regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
# for classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

### Feature engineering of defualt

def non_feature_engineering(raw):
    if 'datetime' in raw.columns:
        raw['datetime'] = pd.to_datetime(raw['datetime'])
        raw['Datetime'] = pd.to_datetime(raw['datetime'])
    if raw.index.dtype == 'int64':
        raw.set_index('Datetime', inplace=True)
    # bring back
    # if raw.index.dtype != 'int64':
    #     raw.reset_index(drop=False, inplace=True)
    raw = raw.asfreq('H', method='ffill')
    raw_nfe = raw.copy()
    return raw_nfe

### Feature engineering of all
def feature_engineering(raw):
    if 'dateitme' in raw.columns:
        raw['datetime'] = pd.to_datetime(raw['datetime'])
        raw['Datetime'] = pd.to_datetime(raw['datetime'])
    
    if raw.index.dtype == 'int64':
        raw.set_index('Datetime', inplace=True)
    
    raw = raw.asfreq('H', method='ffill')

    result = sm.tsa.seasonal_decompose(raw['count'], model='additive')

    Y_trend = pd.DataFrame(result.trend)
    Y_trend.fillna(method='ffill', inplace=True)
    Y_trend.fillna(method='bfill', inplace=True)
    Y_trend.columns = ['count_trend']
    Y_seasonal = pd.DataFrame(result.seasonal)
    Y_seasonal.fillna(method='ffill', inplace=True)
    Y_seasonal.fillna(method='bfill', inplace=True)
    Y_seasonal.columns = ['count_seasonal']
    pd.concat([raw, Y_trend, Y_seasonal], axis=1).isnull().sum()
    if 'count_trend' not in raw.columns:
        if 'count_trend' not in raw.columns:
            raw = pd.concat([raw, Y_trend, Y_seasonal], axis=1)
    
    Y_count_Day = raw[['count']].rolling(24).mean()
    Y_count_Day.fillna(method='ffill', inplace=True)
    Y_count_Day.fillna(method='bfill', inplace=True)
    Y_count_Day.columns = ['count_Day']
    Y_count_Week = raw[['count']].rolling(24*7).mean()
    Y_count_Week.fillna(method='ffill', inplace=True)
    Y_count_Week.fillna(method='bfill', inplace=True)
    Y_count_Week.columns = ['count_Week']
    if 'count_Day' not in raw.columns:
        raw = pd.concat([raw, Y_count_Day], axis=1)
    if 'count_Week' not in raw.columns:
        raw = pd.concat([raw, Y_count_Week], axis=1)
    
    Y_diff = raw[['count']].diff()
    Y_diff.fillna(method='ffill', inplace=True)
    Y_diff.fillna(method='bfill', inplace=True)
    Y_diff.columns = ['count_diff']
    if 'count_diff' not in raw.columns:
        raw = pd.concat([raw, Y_diff], axis=1)
    
    raw['temp_group'] = pd.cut(raw['temp'], 10)
    raw['Year'] = raw.datetime.dt.year
    raw['Quarter'] = raw.datetime.dt.quarter
    raw['Quater_ver2'] = raw['Quarter'] + (raw.year - raw.year.min()) * 4
    raw['Month'] = raw.datetime.dt.month
    raw['Day'] = raw.datetime.dt.day
    raw['Hour'] = raw.datetime.dt.hour
    raw['DayofWeek'] = raw.datetime.dt.dayofweek

    raw['count_lag1'] = raw['count'].shift(1)
    raw['count_lag2'] = raw['count'].shift(2)
    raw['count_lag1'].fillna(method='bfill', inplace=True)
    raw['count_lag2'].fillna(method='bfill', inplace=True)

    if 'Quater' in raw.columns:
        if 'Quater_Dummy' not in ['_'.join(col.split('_')[:2]) for col in raw.columns]:
            raw = pd.concat([raw, pd.get_dummies(raw['Quater'], prefix='Quater_Dummy', drop_first=True)], axis=1)
            del raw['Quater']
    
    raw_fe = raw.copy()
    return raw_fe

### duplicate previous year value to next one
def feature_engineering_year_duplicated(raw, target):
    raw_fe = raw.copy()
    for col in target:
        raw_fe.loc['2012-01-01':'2012-02-28', col] = raw.loc['2011-01-01':'2011-02-28', col].values
        raw_fe.loc['2012-03-01':'2012-12-31', col] = raw.loc['2011-03-01':'2011-12-31', col].values
        step = (raw.loc['2011-03-01 00:00:00', col] + step, raw.loc['2011-03-01 00:00:00', col], step)
        step_value = step_value[:24]
        raw_fe.loc['2012-02-29', col] = step_value
    return raw_fe

### modify lagged values of X_test
def feature_engineering_lag_modified(Y_test, X_test, target):
    X_test_lm = X_test.copy()
    for col in target:
        X_test_lm[col] = Y_test.shift(1).values
        X_test_lm[col].fillna(method='bfill', inplace=True)
        X_test_lm[col] = Y_test.shift(2).values
        X_test_lm[col].fillna(method='bfill', inplace=True)
    
    return X_test_lm

### Data split of cross sectional
def datasplit_cs(raw, Y_colname, X_colname, test_size, random_seed=123):
    X_train, X_test, Y_train, Y_test = train_test_split(raw[X_colname], raw[Y_colname], test_size=test_size, random_state=random_seed)
    print('X_train: ', X_train.shape, 'Y_train: ', Y_train.shape)
    print('X_test: ', X_test.shape, 'Y_test: ', Y_test.shape)
    return X_train, X_test, Y_train, Y_test

### Data split of time series
def datasplit_ts(raw, Y_colname, X_colname, criteria):
    raw_train = raw.loc[raw.index < criteria, :]
    raw_test = raw.loc[raw.index >= criteria, :]
    Y_train = raw_train[Y_colname]
    X_train = raw_train[X_colname]
    Y_test = raw_test[Y_colname]
    X_test = raw_test[X_colname]
    print('Train_size: ', raw_train.shape, 'Test_size: ', raw_test.shape)
    print('X_train: ', X_train.shape, 'Y_train: ', Y_train.shape)
    print('X_test: ', X_test.shape, 'Y_test: ', Y_test.shape)
    return X_train, X_test, Y_train, Y_test

### extract non-multicollinearity variables by VIF

            


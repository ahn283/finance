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
import numpy as np
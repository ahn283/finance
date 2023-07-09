# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
import warnings

# settings
warnings.filterwarnings('ignore', '.*output shape of zoom.*',)  # ignore warning and error messages
warnings.simplefilter(action='ignore', category=FutureWarning)

def adf_test(x):
    '''
    Function for performing the Augmented Dickey-Fuller test for stationarity

    Null hypothesis: time series is not stationary
    Alternate hypothesis: time series is stationary

    Parameters
    ----------
    x: pd.Series / np.array
        The time series to be checked for stationarity
    
    Returns
    ----------
    result: pd.DataFrame
        A pDataFrame with the ADF test's results
    '''

    indices =[ 'Test Statistic', 'p-value', '# of Lags Used', '# of Observations Used']

    adf_test = adfuller(x, autolag='AIC')
    result = pd.Series(adf_test[0:4], index=indices)

    for key, value in adf_test[4].items():
        result[f'Critical Value ({key})'] = value

    return result

def kpss_test(x, h0_type='c'):
    '''
    Function for performing the Kwiatkowski-Phillips-Schmidt-Shin test for stationarity

    Null hypothesis: time series is stationary
    Alternate hypothesis: time series is not stationary

    Parameters
    ----------
    x: pd.Series / np.array
        The time series to be checked for stationarity
    h0_type: str{'c', 'ct'}
        Indicates the null hypothesis of the KPSS test:
            * 'c' : The data is stationary around a constant (default)
            * 'ct' : The data is stationary around a trend
    
    Returns
    ----------
    result: pd.DataFrame
        A pDataFrame with the KPSS test's results
    '''
    
    indices = ['Test Statistic', 'p-value', '# of Lags']

    kpss_test = kpss(x, regression=h0_type)
    result = pd.Series(kpss_test[0:3], index=indices)

    for key, value in kpss_test[3].items():
        result[f'Critical Value ({key})'] = value
    
    return result

def test_autocorrelation(x, n_lags=40, alpha=0.5, h0_type='c'):
    '''
    Function for testing the stationarity of a series by using:
    * the ADF test
    * the KPSS test
    * ACF/PACF plots

    Parameters
    ----------
    x: pd.Series / np.array
        The time series to be checked for stationarity
    n_lags: int
        The number of lags for the ACF/PACF plots
    alpha: float
        Significance level for the ACF/PACF plots
    h0_type: str{'c', 'ct'}
        Indicates the null hypothesis of the KPSS test:
            * 'c' : The data is stationary around a constant (default)
            * 'ct' : The data is stationary around a trend

    Returns
    ----------
    fig : matplotlib.figure.Figure
        The figure object with the ACF/PACF plots
    '''

    adf_result = adf_test(x)
    kpss_result = kpss_test(x, h0_type=h0_type)

    print('ADF test statistics: {:.2f} (p-val: {:.2f})'.format(adf_result['Test Statistic'], adf_result['p-value']))
    print('KPSS test statistics: {:.2f} (p-val: {:.2f})'.format(kpss_result['Test Statistic'], kpss_result['p-value']))

    fig, ax = plt.subplots(2, 1, figsize=(16, 10))
    plot_acf(x, lags=n_lags, alpha=alpha, ax=ax[0])
    plot_pacf(x, lags=n_lags, alpha=alpha, ax=ax[1])

    return fig
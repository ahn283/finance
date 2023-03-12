#
# DX Package
#
# Valuation -- Base Class


class valuation_class(object):
    ''' Basic class for single-factor valuation.

    Attrubutes
    ==========
    name : str
        name of the object
    underlying : instance of simulation class
        object modeling the single risk factor
    mar_env : instance of market_environment
        market environment data for valuation
    payoff_func : str
        derivatives payoff in Python syntax
        Example : 'np.maximum(maturity_value - 100, 0)' 
        where maturity_value is the NumPy vector with
        respective values of the underlying
        Exmaple : 'np.maximum(instrument_values - 100, 0)'
        where instrument_values is the NumPy matrix with
        values of the underlying over the whole time/path grid
    
        
    Methods
    =======
    update:
        update selected valuation parameters
    delta:
        returns the delta of the derivative
    vega:
        returns the vega of the derivative
    '''

    
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

    def __init__(self, name, underlying, mar_env, payoff_func=''):
        self.name = name
        self.pricing_date = mar_env.pricing_date
        try:
            # strike is optional
            self.strike = mar_env.get_constant('strike')
        except:
            pass
        self.maturity = mar_env.get_constant('maturity')
        self.currency = mar_env.get_constant('currency')

        # simulation parameters and discount curve from simulation object
        self.frequency = underlying.frequency
        self.paths = underlying.paths
        self.discount_curve = underlying.discount_curve
        self.puyoff_func = payoff_func
        self.underlying = underlying

        # provide pricing_date and maturity to underlying
        self.underlying.special_dates.extend([self.pricing_date, self.maturity])

    
    def update(self, initial_value=None, volatility=None, strike=None, maturity=None):
        if initial_value is not None:
            self.underlying.update(initial_value=initial_value)
        if strike is not None:
            self.strike = strike
        if maturity is not None:
            self.maturity = maturity

            # add new maturity date if not in time_grid
            if maturity not in self.underlying.time_grid:
                self.underlying.special_dates.append(maturity)
                self.underlying.instrument_values = None
    
    def delta(self, interval=None, accuracy=4):
        if interval is None:
            interval = self.underlying.initial_value / 50.

            # forward-difference approximation
    
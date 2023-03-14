import numpy as np
from dx.sn_random_number import sn_random_numbers
from dx.simulation_class import simulation_class

class geometric_brownian_motion(simulation_class):
    ''' Class to generate simulated paths based on
    the Black-Scholes-Merton Brownian motion model.
    
    
    Attributes
    ==========
    name : string
        name of the object
    mar_env : instance of market_environment
        market environment data for simulation
    corr : Boolwan
        True if correlted with other model simulation object

    Methods
    =======
    update :
        updates parameters
    generate_paths:
        returns Monte Carlo paths given the market environment
    
    
    '''

    def __init__(self, name, mar_env, corr=False):
        super(geometric_brownian_motion, self).__init__(name, mar_env, corr)
    
    def update(self, initial_value=None, volatility=None, final_date=None):
        if initial_value is not None:
            self.initial_value = initial_value
        if volatility is not None:
            self.volatility = volatility
        if final_date is not None:
            self.final_date = final_date
        
        self.instrument_values = None
    
    def generate_paths(self, fixed_seed=False, day_count=365.):
        if self.time_grid is None:
            # method from generic simulation class
            self.generate_time_grid()
        
        # number of data for time grid
        M = len(self.time_grid)

        # number of paths
        I = self.paths

        # ndarray inittialization for path simulation
        paths = np.zeros((M, I))

        # initialize first date with initial_value
        paths[0] = self.initial_value
        if not self.correlated:
            # if not correlated, generate random numbers
            rand = sn_random_numbers((1, M, I), fixed_seed=fixed_seed)
        else:
            # if correlated, use random number object as provided in market environment
            rand = self.random_numbers

        short_rate = self.discount_curve.short_rate

        # get short rate for drift of process
        for t in range(1, len(self.time_grid)):
            # select the right time slice for the relevant

            # random number set
            if not self.correlated:
                ran = rand[t]
            else:
                ran = np.dot(self.cholesky_matrix, rand[:, t, :])
                ran = ran[self.rn_set]
            
            dt = (self.time_grid[t] - self.time_grid[t - 1]).days / day_count

            # difference between two dates at year fraction
            paths[t] = paths[t - 1] * np.exp((short_rate - 0.5 * self.volatility ** 2) * dt + self.volatility * np.sqrt(dt) * ran)

        # generate simulated values for the respective data
        self.instrument_values = paths
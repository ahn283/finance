#
# Event-Based Backtesting
#

class BacktestingBase:
    def __init__(self, env, model, amount, ptc, ftc, verbose=False):
        # the relevant Finance environment
        self.env = env
        # the relevant DNN model (from the trading bot)
        self.model = model
        # the initial/current balance
        self.initial_amount = amount
        self.current_balance = amount
        # proportional transaction costs
        self.ptc = ptc
        # fixed transaction costs
        self.ftc = ftc
        # whether the prints are verbose or not
        self.verbose = verbose
        # the initial number of units of the financial intrument traded
        self.units = 0
        # the initial number of trades implemented
        self.trades = 0
    
    def get_date_price(self, bar):
        '''
        returns date and price for a given bar
        '''
        # the relevant date given a certain bar
        date = str(self.env.data.index[bar])[:10]
        # the relevant instrument price at a certain bar
        price = self.env.data[self.env.symbol].iloc[bar]
        return date, price
    
    def print_balance(self, bar):
        '''
        prints the current cash balance for a given bar
        '''
        date, price = self.get_date_price(bar)
        # the output of the data and current balance for a certain bar
        print(f'{date} | current balance = {self.current_balance:.2f}')
        
    def calculate_net_wealth(self, price):
        # the calculation of the net wealth from the current balance and the instrument position
        return self.current_balance + self.units * price
    
    def print_net_wealth(self, bar):
        '''
        prints the net wealth for a given bar
        (cash + position)
        '''
        date, price = self.get_date_price(bar)
        net_wealth = self.calculate_net_wealth(price)
        # the output of the date and the net wealth at a certain bar
        print(f'{date} | net wealth = {net_wealth:.2f}')
        
    def place_buy_order(self, bar, amount=None, units=None):
        '''
        places a buy order for a given bar and for a given amount or number of units
        '''
        date, price = self.get_date_price(bar)
        if units is None:
            # the number of units to be traded given the trade amount
            units = int(amount / price)
            # units = amount / price    # alternative handling
        # the impact of the trade and the assciated costs on the current balance
        self.current_balance -= (1 - self.ptc) * units * price - self.ptc
        # the adjustment of the number of units held
        self.units += units
        # the adjustment of the number of trades implemented
        self.trades += 1
        if self.verbose:
            print(f'{date} | buy {units} units for {price:.4f}')
            self.print_balance(bar)
            
    def place_sell_order(self, bar, amount=None, units=None):
        '''
        places a sell order a given bar and for a given amount or number of units
        '''
        date, price = self.get_date_price(bar)
        if units is None:
            # the number of units to be traded given the trade amount
            units = int(amount / price)
            # units = amount / price  # alternative handling
            # units = amount / price    # alternative handling
        self.current_balance += (1 - self.ptc) * units * price - self.ftc
        # the adjustment of the number of units held
        self.units -= units
        # the adjustment of the number of trades implemented
        self.trades += 1
        if self.verbose:
            print(f'{date} | sell {units} units for {price:.4f}')
            self.print_balance(bar)
    
    def close_out(self, bar):
        '''
        closes out any open position at a given bar
        '''
        date, price = self.get_date_price(bar)
        print(50 * '=')
        print(f'{date} | *** CLOSING OUT ***')
        if self.units < 0:
            # the closing of a short position
            self.place_buy_order(bar, units=self.units)
        else:
            # the closing of a long position
            self.place_sell_order(bar, units=self.units)
        if not self.verbose:
            print(f'{date} | current balance = {self.current_balance:.2f}')
        # 10
        perf = (self.current_balance / self.initial_amount - 1) * 100
        print(f'{date} | net performance [%] = {perf:.4f}')
        print(f'{date} | number of trades [#] = {self.trades}')
        print(50 * '=')
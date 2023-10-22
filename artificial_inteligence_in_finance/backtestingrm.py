#
# Event-Based Backtesting
#

from backtesting import *

class BacktestingBaseRM(BacktestingBase):
    
    def set_prices(self, price):
        '''
        sets prices for tracking of performance
        to test for e.g. trailing stop loss hit
        '''
        # sets the entry price for the most recent trade
        self.entry_price = price
        # sets the initial minimum price since the most recent trade
        self.min_price = price
        # sets the initial maximum price sinthe the most recent trade
        self.max_price = price
    
    def place_buy_order(self, bar, amount=None, units=None, gprice=None):
        '''
        Places a buy order for a given bar and for a given amount of number of units
        '''
        date, price = self.get_date_price(bar)
        if gprice is not None:
            price = gprice
        if units is None:
            units = int(amount / price)
        self.current_balance -= (1 + self.ptc) * units * price + self.ftc
        self.units += units
        self.trades += 1
        # sets the relevant prices after a trade is executed
        self.set_prices(price)
        if self.verbose:
            price(f'{date} | but {units} units for {price:.4f}')
            self.print_balance(bar)
            
    def place_sell_order(self, bar, amount=None, units=None, gprice=None):
        '''
        places a sell order for a given bar and for a given amount or number of units.
        '''
        date, price = self.get_date_price(bar)
        if gprice is not None:
            price = gprice
        if units is None:
            units = int(amount / price)
        self.current_balance += (1 - self.ptc) * units * price - self.ftc
        self.units -= units
        self.trades += 1
        # sets the relevant prices after a trade is executed
        self.set_prices(price)
        if self.verbose:
            print(f'{date} | sell {units} units for {price:.4f}')
        
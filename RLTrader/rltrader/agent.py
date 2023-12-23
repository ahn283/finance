import numpy as np
from . import utils

class Agent:
    ''' 
    Attributes
    ---------
    - environment : 환경 객체
    - initial_balance : 초기 자본금
    - min_trading_price : 최소 단일 매매 금액
    - max_trding_price : 최대 단일 매매 금액
    - balance : 현금 잔고
    - num_stocks : 보유 주식 수
    - portfolio_value : 포트폴리오 가치 (투자금 잔고 + 주식 현재가 * 보유 주식수)
    - num_buy : 매수 횟수
    - num_sell : 매도 횟수
    - num_hold : 관망 횟수
    - ratio_hold : 주식 보유 비율
    - profitloss : 현재 손익
    - avg_buy_price : 주당 매수 단가
    
    Functions
    ----------
    - reset() : 에이전트 상태를 초기화
    - set_balance() : 초기 자본금을 설정
    - get_states() : 에이전트 상태를 획득
    - decide_action() : 탐험 또는 정책 신경망에 의한 행동 결정
    - validate_action() : 행동의 유효성 판단
    - decide_trading_unit() : 매수 또는 매도할 주식 수 결정
    - act() : 행동 수행
    '''
    # 에이전트 상태가 구성하는 값 개수 (차원의 개수)
    ## 주식 보유 비율, 손익률, 주당 매수 단가 대비 주가 등락률
    STATE_DIM = 3
    
    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015        # 거래 수수료 0.015%
    # TRADING_CHARGE = 0.00011        # 거래 수수료 0.011%
    # TRADING_CHARGE = 0              # 거래 수수료 미적용
    TRADING_TAX = 0.02              # 거래세 0.2%
    # TRADING_TAX = 0                 # 거래세 미적용
    
    # 행동
    ACTION_BUY = 0      # 매수
    ACTION_SELL = 1     # 매도
    ACTION_HOLD = 2     # 관망
    
    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS)      # 인공 신경망에서 고려할 출력값의 개수
    
    def __init__(self, environment, initial_balance, min_trading_price, max_trading_price):
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment
        self.initial_balance = initial_balance  # 초기 자본금
        
        # 최소 단일 매매 금액, 최대 단일 매매 금액
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price
        
        # Agent 클래스의 속성
        self.balance = initial_balance      # 현재 현금 잔고 / initial_value : 초기 자본금
        self.num_stocks = 0                 # 보유 주식 수
        
        # 포트폴리오 가치 : balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0
        self.num_buy = 0            # 매수 횟수
        self.num_sell = 0           # 매도 횟수
        self.num_hold = 0           # 관망 횟수
        
        # Agent 클래스의 상태
        self.ratio_hold = 0         # 주식 보유 비율
        self.profitloss = 0         # 손익률
        self.avg_buy_price = 0      # 주당 매수 단가
        
    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.ratio_hold = 0
        self.profitloss = 0
        self.avg_buy_price = 0
        
    def set_balance(self, balance):
        # 초기 자본금 설정
        self.initial_balance = balance

    def get_states(self):
        # 주식 보유 비율 = 보유 주식수 / (포트폴리오의 가치 / 현재 주가)
        self.ratio_hold = self.num_stocks * self.environment.get_price() / self.portfolio_value
        
        return (
            self.ratio_hold,
            self.profitloss,        # 손익률 = (포트폴리오 가치 / 초기 자본금) - 1
            (self.environment.get_price() / self.avg_buy_price) - 1 if self.avg_buy_price > 0 else 0        # 주당 매수 단가 대비 주가 등락률 = (주가 / 주당 매수 단가) - 1
        )
    
    def decide_action(self, pred_value, pred_policy, epsilon):
        # epsilon 확률로 무작위 행동, 그 외에 경우 신경망을 통해 행동
        confidence = 0
        
        # pred_policy가 있으면 pred_policy로 행동을 결정하고, 없으면 pred_value로 행동을 결정
        pred = pred_policy
        if pred is None:
            pred = pred_value
        
        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1
            
            # 매수와 매도 출력값의 차이가 매우 작으면 (0.05) 탐험 수행
            if pred_policy is not None:
                if np.max(pred_policy) - np.min(pred_policy) < 0.05:
                    epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
        else:
            exploration = False
            action = np.argmax(pred)
        
        confidence = .5
        # pred_policy가 있으면 pred_policy로 행동을 결정하고, 없으면 pred_value로 행동을 결정
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])
        
        return action, confidence, exploration
    
    def validate_action(self, action):
        # 결정한 행동의 유효성을 검사
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE):
                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True
    
    def decide_trading_unit(self, confidence):
        # 결정한 행동의 신뢰(confidence)에 따라서 매수 또는 매도의 단위를 조정
        if np.isnan(confidence):
            return self.min_trading_price
        
        # self.min_trading_price + added_trading_priceRK [min_trading_price, max_trading_price] 범위로 매수 금액 정함
        # confidence가 1을 넘어가면 max_trading_value를 넘어버릴 수 있으므로 min()으로 self.max_trading_price - self.min_trading_price를 못 넘게 하고
        # confidence가 음수로 올 경우를 대비해 max()를 취해 added_trading_price가 0 아래로 못넘어가게 조정
        added_trading_price = max(min(
            int(confidence * (self.max_trading_price - self.min_trading_price)),
            self.max_trading_price - self.min_trading_price
        ), 0) 
        trading_price = self.min_trading_price + added_trading_price
        
        return max(int(trading_price / self.environment.get_price()), 1)
    
    def act(self, action, confidence):
        
        '''
        Arguments
        ----------
        - action : 탐험 또는 정책 신경망을 통해 결정한 행동 (0 or 1)
        - confidence : 정책 신경망을 통해 결정한 경우 결정한 행동에 대한 소프트맥스 확률값
        '''
        
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD
        
        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()
        
        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            )
            
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = min(
                    int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))),
                    int(self.max_trading_price / curr_price)
                )
            
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            if invest_amount > 0:
                self.avg_buy_price = (self.avg_buy_price * self.num_stocks + curr_price * trading_unit) / (self.num_stocks + trading_unit)  # 주당 매수 단가 갱신
                self.balance -= invest_amount       # 보유 현금을 갱신
                self.num_stocks += trading_unit     # 보유 주식 수를 갱신
                self.num_buy += 1                   # 매수 횟수 증가
                
        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)
            ) * trading_unit
            
            if invest_amount > 0:
                # 주당 매수 단가 갱신
                self.avg_buy_price = (self.avg_buy_price * self.num_stocks - curr_price * trading_unit) / (self.num_stocks - trading_unit) if self.num_stocks > trading_unit else 0
                self.num_stocks -= trading_unit
                self.balance += invest_amount
                self.num_sell += 1
        
        # 관망
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1
            
        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = self.portfolio_value / self.initial_balance - 1
        return self.profitloss
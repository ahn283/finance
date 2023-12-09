import numpy as np
from RLTrader.rltrader import utils

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
    # 에이전트 상태가 구성하는 값 개수
    # 주식 보유 비율, 손익률, 주당 매수 단가 대비 주가 등락률
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
        self.balance = initial_balance      # 현재 현금 잔고
        self.num_stocks = 0                 # 보유 주식 수
        
        # 포트폴리오 가치 : balance + num_stocks * {현재 주시 가격}
        self.portfolio_value = 0
        self.num_buy = 0            # 매수 횟수
        self.num_sell = 0           # 매도 횟수
        self.num_hold = 0           # 관망 횟수
        
        # Agent 클래스의 상태
        self.ratio_hold = 0         # 주식 보유 비율
        self.profitloss = 0         # 손익률
        self.avg_buy_price = 0      # 주당 매수 단가
        
    
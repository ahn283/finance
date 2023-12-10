import os
import logging
import abc      # 추상 클래스 정의에 사용
import collections
import threading
import time     # 학습 시간 측정
import json
import numpy as np
from tqdm import tqdm

from rltrader.environment import Environment
from rltrader.agent import Agent
from rltrader.networks import Network, DNN, LSTMNetwork, CNN
from rltrader.visualizer import Visualizer
from rltrader import utils
from rltrader import settings

logger = logging.getLogger(settings.LOGGER_NAME)

# DQNLearner
class ReinforcementLearner:
    '''
    Attributes
    ----------
    - stock_code : 강화학습 대상 주식 종목 코드
    - chart_data : 주식 종목의 차트 데이터
    - environment : 강화학습 환경 객체
    - agent : 강화학습 에이전트 객체
    - training_data : 학습 데이터
    - value_network : 가치 신경망
    - policy_network : 정책 신경망
    
    Functions
    ----------
    - init_value_network() : 가치 신경망 생성 함수
    - init_policy_network() : 정책 신경망 생성 함수
    - build_sample() : 환경 객체에서 샘플을 획득하는 함수
    - get_batch() : 배치 학습 데이터 생성 함수
    - update_network() : 가치 신경망 및 정책 신경망 학습 함수
    - fit() : 가치 신경망 및 정책 신경망 학습 요청 함수
    - visualize() : 에포크 정보 가시화 함수
    - fun() : 강화학습 수행 함수
    - save_models() : 가치 신경망 및 정책 신경망 저장 함수
    '''
    
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()
    
    def __init__(self, rl_method='rl', stock_code=None,
                 chart_data=None, trading_data=None,
                 min_trading_price=100000, max_trading_price=10000000,
                 net='dnn', num_steps=1, lr=0.0005,
                 discount_factor=0.9, num_epochs=1000,
                 balance=100000000, start_epsilon=1,
                 value_network=None, policy_network=None,
                 output_path='', reuse_models=True, gen_output=True):
        ''' 
        Attrubutes
        ---------
        - rl_method : 강화학습 기법
            'dqn': DQNLearner, 'pg': PolicyGradient, 'ac': ActorCriticLearner, 'a2c': A2CLearner, 'a3c': A3CLearner
        - stock_code : 주식 종목 코드
        - chart_data : 강화학습 환경에 대항하는 주식 일봉 차트 데이터
        - training_data : 학습을 위한 전처리 데이터
        - min_trading_price, max_trading_price : 단일 거래 최소, 최대 금액
        - net : 신경망
            'dnn', 'lstm', 'cnn'
        - n_steps : LTSM, CNN 신경망에서 사용하는 샘플 묶음의 크기
        - lr : 학습속도
        - discount_factor : 상태-행동 가치를 구할 때 적용할 할인율
        - num_epochs : 총 수행할 반복 학습 횟수
        - balance : 에이전트 초기 투자 자본금
        - start_epsilon : 초기 탐험 비율
        - value_network, policy_network : 가치 신경망, 정책 신경망
        - output_path : 가시화 결과 및 신경망 모델 저장 경로
        - reuse_models : 가치 신경망, 정책 신경망 재활용 여부
        '''
        
        # 인자 확인
        assert min_trading_price > 0
        assert max_trading_price > 0
        assert max_trading_price >= min_trading_price
        assert num_steps > 0
        assert lr > 0
        
        # 강화학습 설정
        self.rl_method = rl_method
        self.discount_factor = discount_factor
        self.num_epochs = num_epochs
        self.start_epsilon = start_epsilon
        
        # 환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        
        # 에이전트 설정
        self.agent = Agent(self.environment, balance, min_trading_price, max_trading_price)
        
        # 학습 데이터
        self.training_data = trading_data
        self.sample = None
        self.training_data_idx = -1
        
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        
        # 가시화 모듈
        self.visualizer = Visualizer()
        
        # 메모리
        self.memory_sample = []     # 학습 데이터 샘플
        self.memory_action = []     # 수행한 행동
        self.memory_reward = []     # 획득한 보상
        self.memory_value = []      # 행동의 예측 가치
        self.memory_policy = []     # 행동의 예측 확률
        self.memory_pv = []         # 포트폴리오 가치
        self.memory_num_stocks = [] # 보유 주식 수
        self.memory_exp_idx = []    # 탐험 위치
        
        # 강화학습 에포크 관련 정보
        self.loss = 0.              # 에포크 동안 학습에서 발생한 손실
        self.itr_cnt = 0            # 수익 발생 횟수
        self.exploration_cnt = 0    # 탐험 횟수
        self.batch_size = 0         # 학습 횟수ㄴ
        
        # 로그 등 출력 검토
        self.output_path = output_path
        self.gen_output = gen_output
        
    def init_value_network(self, shared_network=None, activation='linear', loss='mse'):
        if self.net == 'dnn':
            self.value_network = DNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss
            )
        
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss
            )
        elif self.net == 'cnn':
            self.value_network = CNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss
            )
        
        if self.reuse_models and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)
        
    def init_policy_network(self, shared_network=None, activation='sigmoid', loss='binary_crossentropy'):
        if self.net == 'dnn':
            self.policy_network = DNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss
            )
        
        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss
            )
        elif self.net == 'cnn':
            self.policy_network = CNN(
                input_dim=self.num_features,
                outpupt_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss
            )
        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)
            
    def reset(self):
        self.sample = None
        self.training_data_idx = -1
        
        # 환경 초기화
        self.environment.reset()
        
        # 에이전트 초기화
        self.agent.reset()
        
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0
        self.exploration_cnt = 0
        self.batch_size = 0
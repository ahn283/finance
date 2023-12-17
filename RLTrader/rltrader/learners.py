import os
import logging
import abc      # 추상 클래스 정의에 사용
import collections
import threading
import time     # 학습 시간 측정
import json
import numpy as np
from tqdm import tqdm

from .environment import Environment
from .agent import Agent
from .networks import Network, DNN, LSTMNetwork, CNN
from .visualizer import Visualizer
from . import utils
from . import settings

# from rltrader.environment import Environment
# from rltrader.agent import Agent
# from rltrader.networks import Network, DNN, LSTMNetwork, CNN
# from rltrader.visualizer import Visualizer
# from rltrader import utils
# from rltrader import settings


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
                 chart_data=None, training_data=None,
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
        self.training_data = training_data
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
        
    def build_sample(self):
        # 차트 데이터의 현재 인덱스에서 다음 인덱스 데이터를 읽음
        self.environment.observe()
        # sample 47개 값 + 다음으로 sample 에이전트 상태를 추가해 50개 값으로 구성
        if len(self.training_data) > self.training_data_idx + 1:
            self.training_data_idx += 1
            self.sample = self.training_data[self.training_data_idx].tolist()
            self.sample.extend(self.agent.get_states())
            return self.sample
        return None
    
    # 추상 메서드로 하위 클래스들은 반드시 이 함수를 구현해야 한다.
    @abc.abstractmethod
    def get_batch(self):
        pass
    
    # 배치 학습 데이터 생성 후, 가치 신경망과 정책 신경망 학습 위해 train_on_batch() 함수 호출
    # 가치 신경망 : DQNLearner, ActorCriticLearner, A2CLearner
    # 정책 신경망 : PolicyGradientLearner, ActorCriticLearner, A2CLeaner
    # 학습 후 발생하는 loss를 인스턴스 속성으로 저장. 가치 신경망과 정책 신경망을 모두 학습하는 경우 손실을 합산해 반환
    def fit(self):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch()
        # 손실 초기화
        self.loss = None
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_policy)
            self.loss = loss
    
    # 하나의 에포크가 완료되어 에포크 관련 정보를 가시화하는 부분
    # LSTM, CNN 신경망을 사용하는 경우 에이전트 행동, 보유 주식수, 가치 신경망 출력, 정책 신경망 출력. 포트폴리오 가치는 환경의 일봉 수보다 (num_steps - 1)만큼 부족하므로 (num_steps - 1)만큼 의미 없는 값을 첫 부분에 채워준다.  
    def visualize(self, epoch_str, num_epochs, epsilon):
        self.memory_action = [Agent.ACTION_HOLD] * (self.num_steps - 1) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps - 1) + self.memory_num_stocks
        if self.value_network is not None:
            self.memory_value = [np.array([np.nan] * len(Agent.ACTIONS))] * (self.num_steps - 1) + self.memory_value
        if self.policy_network is not None:
            self.memory_policy = [np.array([np.nan] * len(Agent.ACTIONS))] * (self.num_steps - 1) + self.memory_policy
        
        self.memory_pv = [self.agent.initial_balance] * (self.num_steps - 1) + self.memory_pv
        self.visualizer.plot(
            epoch_str=epoch_str, num_epochs=num_epochs,
            epsilon=epsilon, action_list=Agent.ACTIONS,
            actions=self.memory_action,
            num_stocks=self.memory_num_stocks,
            outvals_value=self.memory_value,
            outvals_policy=self.memory_policy,
            exps=self.memory_exp_idx,
            initial_balance=self.agent.initial_balance,
            pvs=self.memory_pv,
        )
        self.visualizer.save(os.path.join(self.epoch_summary_dir, f'epoch_summary_{epoch_str}.png'))
    
    def run(self, learning=True):
        '''
        Arguments
        ---------
        - learning : 학습 유무를 정하는 Boolean 값
            True : 학습을 마치면 학습된 가치 신경망 모델과 정책 신경망 모델이 만들어지는데 이처럼 신경망 모델을 만드는 경우
            False : 학습된 모델로 투자 시뮬레이션만 하는 경우
        '''
        info = (
            f'[{self.stock_code}] RL:{self.rl_method} NET:{self.net}'
            f'LR:{self.lr} DF:{self.discount_factor}'
        )
        with self.lock:
            logger.debug(info)
            
        # start time
        time_start = time.time()
        
        # 가시화 준비
        # 차트데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data, info)
        
        # 가시화 결과 저장할 폴더 준비
        if self.gen_output:
            # 가시화 결과는 output_path 경로 하위의 epoch_summary_* 폴더에 저장
            self.epoch_summary_dir = os.path.join(self.output_path, f'epoch_summary_{self.stock_code}')
            if not os.path.isdir(self.epoch_summary_dir):
                os.makedirs(self.epoch_summary_dir)
            else:
                for f in os.listdir(self.epoch_summary_dir):
                    os.remove(os.path.join(self.epoch_summary_dir, f))
                    
        # 학습에 대한 정보 초기화
        # max_portfolio_value 변수에는 수행한 epoch 중에서 가장 높은 포트폴리오 가치를 저장
        max_portfolio_value = 0
        # 수행한 epoch 중에서 수익이 발생한 epoch 수 저장
        epoch_win_cnt = 0
        
        # iterate epochs
        for epoch in tqdm(range(self.num_epochs)):
            # epoch 시작 시간 기록
            time_start_epoch = time.time()
            
            # step 샘플을 만들기 위한 큐
            q_sample = collections.deque(maxlen=self.num_steps)
            
            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()
            
            # 학습을 진행할 수 수록 탐험 비율 감소
            if learning:
                epsilon = self.start_epsilon * (1 - (epoch / (self.num_epochs - 1)))
            else:
                epsilon = self.start_epsilon
            
            for i in tqdm(range(len(self.training_data)), leave=False):
                # 샘플 생성
                next_sample = self.build_sample()
                if next_sample is None:
                    break
                
                # num_steps만큼 샘플이 준비돼야 행동을 결정할 수 있으므로 num_steps만큼 샘플 저장
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue
                
                # 가치, 정책 신경망 예측
                pred_value = None
                pred_policy = None
                # 예측 행동 가치 도출
                if self.value_network is not None:
                    pred_value = self.value_network.predict(list(q_sample))
                # 예측 행동 확률 도출
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(list(q_sample))
                
                # 예측 가치와 확률로 투자 행동을 결정    
                # 신경망 또는 탐험에 의한 행동 결정
                # 행동 결정은 epsilon 값을 확률로 무작위 또는 신경망의 출력을 통해 결정
                # 정책 신경망의 출력은 매수를 했을 때와 매도를 했을 때의 포트폴리오 가치를 높일 확률을 의미 -> 매수에 대한 정책 신경망 출력이 매도에 대한 출력보다 높으면 매수, 반대면 매도
                # 정책 신경망 출력이 없으면 가치 신경망의 출력값이 높은 행동을 선택
                action, confidence, exploration = self.agent.decide_action(pred_value, pred_policy, epsilon)
                
                # 결정한 행동을 수행하고 보상 획득
                reward = self.agent.act(action, confidence)
                
                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)
                    
                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0
                
            # 에포크 종료 후 신경망 학습
            if learning:
                self.fit()
                
            # epoch 관련 정보 로그 기록
            # epoch 수의 문자열 길이를 확인 1,000이면 4
            num_epochs_digit = len(str(self.num_epochs))
            # num_epochs_digit 길이의 문자열을 만들어 앞에 0을 채운다. 예) '0001'
            epoch_str = str(epoch + 1).rjust(num_epochs_digit, '0')
            time_end_epoch = time.time()
            # epoch 수행 시간 기록
            eplapsed_time_epoch = time_end_epoch - time_start_epoch
            logger.debug(f'[{self.stock_code}][Epoch {epoch_str}/{self.num_epochs}]'
                         f'Epsilon:{epsilon:.4f} #Expl.:{self.exploration_cnt}/{self.itr_cnt} '
                         f'#Buy:{self.agent.num_buy} #Sell:{self.agent.num_sell} #Hold:{self.agent.num_hold} '
                         f'#Stocks:{self.agent.num_stocks} PV:{self.agent.portfolio_value:,.0f} '
                         f'Loss:{self.loss:.6f} ET:{eplapsed_time_epoch:.4f}'
                         )
            
            # 에포크 관련 정보 가시화
            if self.gen_output:
                if self.num_epochs == 1 or (epoch + 1) % max(int(self.num_epochs / 10), 1) == 0:
                    self.visualize(epoch_str, self.num_epochs, epsilon)
                    
            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value
            )
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1
        
        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start
        
        # 학습 관련 정보 로그 기록
        with self.lock:
            logger.debug(f'[{self.stock_code}] Elapsed Time:{elapsed_time:.4f}'
                         f'Max PV:{max_portfolio_value:,.0f} #Win:{epoch_win_cnt}')
            
    
    def save_models(self):
        if self.value_network is not None and self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)
    
    # 학습은 수행하지 않고 불러온 모델을 이용하여 샘플에 대해 수행할 행동과 그 확률을 예측하여 반환        
    def predict(self):
        # 에이전트 초기 자본금 설정
        # self.agent.set_balance(balance)
        
        # 에이전트 초기화
        self.agent.reset()
        
        # step 샘플을 만들기 위한 큐
        q_sample = collections.deque(maxlen=self.num_steps)
        
        result = []
        while True:
            # 샘플 생성
            next_sample = self.build_sample()
            if next_sample is None:
                break
            
            # num_steps만큼 샘플 저장
            q_sample.append(next_sample)
            if len(q_sample) < self.num_steps:
                continue
            
            # 가치, 정책 신경망 예측
            pred_value = None
            pred_policy = None
            if self.value_network is not None:
                pred_value = self.value_network.predict(list(q_sample)).tolist()
            if self.policy_network is not None:
                pred_policy = self.policy_network.predict(list(q_sample)).tolist()
            
            # 신경망에 의한 행동 결정
            result.append((self.environment.observation[0], pred_value, pred_policy))
        
        if self.gen_output:
            with open(os.path.join(self.output_path, f'pred_{self.stock_code}.json'), 'w') as f:
                print(json.dumps(result), file=f)
                
        return result

class DQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super.__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        # 가치 신경망 생성
        self.init_value_network()

    # Reinforcement 추상 메서드 구현
    def get_batch(self):
        # 메모리 배열을 역으로 묶어줌
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_reward),
        )
        # 샘플 배열 x, 레이블배열 y_value를 0으로 준비
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        # 메모리를 역으로 취했기 떄문에 학습 데이터 마지막부터 처리
        for i, (sample, action, value, reward) in enumerate(memory):
            # 샘플
            x[i] = sample
            # 학습에 사용할 보상
            ## memory_reward[-1] : 배치 데이터 내에서의 마지막 손익률
            ## reward : 행동을 수행한 시점에서의 손익률
            r = self.memory_reward[-1] - reward
            # 가치 신경망 출력
            y_value[i] = value
            # 상태-행동 가치
            y_value[i, action] = r + self.discount_factor * value_max_next
            # 다음 상태의 최대 가치는 저장
            value_max_next = value.max()
        
        # 샘플 배열, 가치 신경망 학습 레이블 배열, 정책 신경망 학습 레이블 배열 반환
        # DQN은 정책 신경망을 사용하지 않으므로 None 처리
        return x, y_value, None
    
class PolicyGradientLearner(ReinforcementLearner):
    def __init__(self, *args, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_network_path = policy_network_path
        self.init_policy_network()
        
    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        # 학습 데이터 및 에이전트 상태로 구성된 샘플 배열
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        # 정책 신경망 학습을 위한 레이블 배열
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        for i, (sample, action, policy, reward) in enumerate(memory):
            # 특징 벡터
            x[i] = sample
            r = self.memory_reward[-1] - reward
            # 정책 신경망 출력
            y_policy[i, :] = policy
            # 보상(r)에 sigmoid 함수를 취해서 정책 신경망 학습 레이블로 정한다.
            y_policy[i, action] = utils.sigmoid(r)
        
        # PG는 가치 신경망이 없으므로 두번쨰 값 None
        return x, None, y_policy
    
class ActorCriticLearner(ReinforcementLearner):
    def __init__(self, *args, shared_network=None,
                 value_network_path=None, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        # 가치 신경망과 정책 신경망 상단부 레이어를 공유
        if shared_network is None:
            self.shared_network = Network.get_shared_network(
                net=self.net, num_steps=self.num_steps,
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS
            )
        else:
            self.shared_network = shared_network
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=self.shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=self.shared_network)
    
    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = self.memory_reward[-1] - reward
            # DQN과 동일하게 레이블을 넣어준다.
            y_value[i, :] = value
            y_value[i, action] = r + self.discount_factor * value_max_next
            y_policy[i, :] = policy
            # 정책 신경망의 레이블
            y_policy[i, action] = utils.sigmoid(r)
            value_max_next = value.max()
        return x, y_value, y_policy
    
class A2CLearner(ActorCriticLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            r = reward_next + self.memory_reward[-1] - reward * 2
            reward_next = reward
            y_value[i, :] = value
            y_value[i, action] = np.tanh(r + self.discount_factor * value_max_next)
            advantage = y_value[i, action] - y_value[i].mean()
            y_policy[i, :] = policy
            # davantage를 시그모이드 함수에 적용해 정책 신경망의 학습 레이블로 적용
            y_policy[i, action] = utils.sigmoid(advantage)
            value_max_next = value.max()
        return x, y_value, y_policy

class A3CLearner(ReinforcementLearner):
    def __init__(self, *args, list_stock_code=None,
                 list_chart_data=None, list_training_data=None,
                 list_min_trading_price=None, list_max_trading_price=None,
                 value_network_path=None, policy_network_path=None,
                 **kwargs):
        assert len(list_training_data) > 0
        super().__init__(*args, **kwargs)
        self.num_features += list_training_data[0].shape[1]
        
        '''
        생성자의 인자들은 리스트로 학습할 종목 코드, 차트 데이터, 학습 데이터, 최소 및 최대 투자 단위를 받는다.
        이 리스트들의 크기만큼 A2CLearner 클래스 객체들을 생성한다.
        각 A2CLearner 클래스 객체는 가치 신경망과 정책 신경망을 공유한다.
        '''
        
        # 공유 신경망 생성
        self.shared_network = Network.get_shared_network(
            net=self.net, num_steps=self.num_steps,
            input_dim=self.num_features,
            output_dim=self.agent.NUM_ACTIONS
        )
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=self.shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=self.shared_network)
        
        # A2CLearner 생성
        self.learners = []
        for (stock_code, chart_data, training_data,
             min_trading_price, max_trading_price) in zip(
                 list_stock_code, list_chart_data, list_training_data,
                 list_max_trading_price, list_max_trading_price
             ):
                 learner = A2CLearner(*args, stock_code=stock_code,
                                      chart_data=chart_data,
                                      training_data=training_data,
                                      min_trading_price=min_trading_price,
                                      max_trading_price=max_trading_price,
                                      shared_network=self.shared_network,
                                      value_network=self.value_network,
                                      policy_network=self.policy_network, **kwargs)
                 self.learners.append(learner)
    
    def run(self, learning=True):
        threads = []
        # A2CLearner 클래스 객체의 run()을 동시에 실행
        for learner in self.learners:
            threads.append(threading.Thread(
                target=learner.run, daemon=True, kwargs={'learning': learning}
            ))
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
            
    def predict(self):
        threads = []
        for learner in self.learners:
            threads.append(threading.Thread(
                target=learner.predict, daemon=True
            ))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
class PPOLearner(A2CLearner):
    def __init__(self, *args, lmb=0.95, eps=0.1, K=3, **kwargs):
        kwargs['value_network_activation'] = 'tanh'
        kwargs['policy_network_activation'] = 'tanh'
        super().__init__(*args, **kwargs)
        self.lmb = lmb
        self.eps = eps
        self.K = K
        
    def get_batch(self):
        memory = zip(
            reversed(self.memory_sample),
            reversed(self.memory_action),
            reversed(self.memory_value),
            reversed(self.memory_policy),
            reversed(self.memory_reward),
        )
        x = np.zeros((len(self.memory_sample), self.num_steps, self.num_features))
        y_value = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        y_policy = np.zeros((len(self.memory_sample), self.agent.NUM_ACTIONS))
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) in enumerate(memory):
            x[i] = sample
            # r = reward_next - reward
            # reward_next = reward
            y_value[i, :] = value
            y_value[i, action] = np.tanh(reward + self.discount_factor * value_max_next)
            advantage = y_value[i, action] - y_value[i].mean()
            y_policy[i, :] = policy
            y_policy[i, action] = advantage
            value_max_next = value.max()
        return x, y_value, y_policy
    
    def fit(self):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch()
        # 손실 초기화
        self.loss = None
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch_for_ppo(x, y_policy, list(reversed(self.memory_action)), self.eps, self.K)
            self.loss = loss
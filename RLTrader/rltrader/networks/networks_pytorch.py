import threading
import abc
import numpy as np

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Network:
    '''
    RLTrader에서 신경망이 공통으로 가질 속성과 함수를 정의해 놓은 클래스
    Network 클래스를 상속한 DNN LSTMNetwork, CNN 클래스를 사용
    
    Attributes
    ----------
    - input_dim : 입력 데이터의 차원
    - output_dim : 출력 데이터의 차원
    - lr : 신경망의 학습 속도
    - shared_network : 신경망의 상단부로 여러 신경망이 공유할 수 있음
        예: A2C에서는 가치 신경망과 정책 신경망이 신경망의 상단부를 공유하고, 하단 부분만 가치 예측과 확률 예측을 위해 달라짐
    - activation : 신경망의 출력 레이어 활성화 함수 이름 
        'linear', 'sigmoid', 'tanh', 'softmax' 등
    - loss : 신경망의 손실함수
    - model : PyTorch 또는 Keras 프레임워크로 구성한 최종 신경망 모델
    
    Functions
    ----------
    - predict() : 신경망을 통해 투자 행동별 가치나 확률 계산
    - train_on_batch() : 배치 학습을 위한 데이터 생성
    - save_model() : 학습한 신경망을 파일로 저장
    - load_model() : 파일로 저장한 신경망 업로드
    - get_shared_nerwork() : 신경망의 상단부를 생성하는 클래스 함수
    '''
    
    # A3C에서 필요한 스레드 락
    lock = threading.Lock()
    
    def __init__(self, input_dim=0, output_dim=0, lr=0.001,
                 shared_network=None, activation='sigmoid', loss='mse'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss
        
        # 신경망의 데이터 형태
        # CNN, LSTMNetwork 클래스는 3차원으로 구성하므로 inp를 (num_steps, input_dim)으로 설정하고, DNN은 (input_dim,)으로 설정
        inp = None
        if hasattr(self, 'num_steps'):
            inp = (self.num_steps, input_dim)
        else:
            inp = (self.input_dim,)
        
        # 공유 신경망 사용
        self.head = None
        if self.shared_network is None:
            self.head = self.get_network_head(inp, self.output_dim)
        else:
            self.head = self.shared_network
            
        # 공유 신경망 미사용
        # self.head = self.get_network_head(inp, self.output_dim)
        
        # 신경망 모델
        ## 신경망의 앞단인 head로 신경망 모델 생성 
        self.model = torch.nn.Sequential(self.head)
        if self.activation == 'linear':
            pass
        elif self.activation == 'relu':
            self.model.add_module('activation', torch.nn.ReLU())
        elif self.activation == 'leaky_relu':
            self.model.add_module('activation', torch.nn.LeakyReLU())
        elif self.activation == 'sigmoid':
            self.model.add_module('activation', torch.nn.Sigmoid())
        elif self.activation == 'tanh':
            self.model.add_module('avtivation', torch.nn.Tanh())
        elif self.activation == 'softmax':
            self.model.add_module('activation', torch.nn.Softmax(dim=1))
        self.model.apply(Network.init_weights)
        self.model.to(device)
        
        # optimizer
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.NAdam(self.model.parameters(), lr=self.lr)
        
        # loss function
        self.criterion = None
        if loss == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif loss == 'binary_crossentropy':
            self.criterion = torch.nn.BCELoss()
    
    def predict(self, sample):
        # 주어진 샘플에 대해서 매수, 매도, 관망의 각 행동들의 예측값을 반환
        # 예측값은 가치 신경망의 경우 주어진 샘플에 대한 각 행동의 가치, 정책 신경망의 경우 각 행동의 확률값
        with self.lock:
            # 평가 모드 전환 : Drop out 같은 학습에만 사용되는 계측 비활성화
            self.model.eval()
            with torch.no_grad():
                x = torch.from_numpy(sample).float().to(device)
                pred = self.model(x).detach().cpu().numpy()
                pred = pred.flatten()
            return pred
    
    def train_on_batch(self, x, y):
        loss = 0
        with self.lock:
            # 학습 모드 전환
            self.model.train()
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)
            y_pred = self.model(x)
            _loss = self.criterion(y_pred, y)
            self.optimizer.zero_grad()
            _loss.backward()
            self.optimizer.step()
            loss += _loss.item()
        return loss
    
    @classmethod
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0, output_dim=0):
        if net == 'dnn':
            return DNN.get_network_head((input_dim), output_dim)
        elif net == 'lstm':
            return LSTMNetwork.get_network_head((num_steps, input_dim), output_dim)
        elif net == 'cnn':
            return CNN.get_shared_head((num_steps, input_dim), output_dim)
        
    @abc.abstractmethod
    def get_network_head(inp, output_dim):
        pass
    
    @staticmethod
    def init_weights(m):
        # 가중치의 정규분포 방식 초기화
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=0.01)
        elif isinstance(m, torch.nn.LSTM):
            for weights in m.all_weights:
                for weight in weights:
                    torch.nn.init.normal_(weight, std=0.01)
    
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            torch.save(self.model, model_path)
    
    def load_model(self, model_path):
        if model_path is not None:
            self.model = torch.load(model_path)
             
        
class DNN(Network):
    @staticmethod
    def get_network_head(inp, output_dim):
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]),
            torch.nn.Linear(inp[0], 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim),
        )
        
    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.input_dim))
        return super().train_on_batch(x, y)
    
    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample)

class LSTMNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        self.num_steps = num_steps
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def get_network_head(inp, outpu_dim):
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]),
            LSTMModule(inp[1], 128, batch_first=True, use_last_only=True),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, outpu_dim),
        )
        
    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)
    
    def predict(self, sample):
        sample = np.array(sample).reshape((-1, self.num_steps, self.input_dim))
        return super().predict(sample)

class LSTMModule(torch.nn.LSTM):
    def __init__(self, *args, use_last_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_last_only = use_last_only
    
    def forward(self, x):
        output, (h_n, _) = super().forward(x)
        if self.use_last_only:
            return h_n[-1]
        return output

class CNN(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        self.num_steps = num_steps
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def get_network_head(inp, output_dim):
        kernel_size = 2
        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp[0]),
            torch.nn.Conv1d(inp[0], 1, kernel_size),
            torch.nn.BatchNorm1d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(inp[1] - (kernel_size - 1), 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(32, output_dim),
        )       
        
    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)
    
    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.num_steps, self.input_dim))
        return super().predict(sample)
        
        
        
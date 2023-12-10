import threading
import numpy as np
import matplotlib.pyplot as plt
import datetime
plt.switch_backend('agg')

from mplfinance.original_flavor import candlestick_ohlc
from rltrader.agent import Agent

lock = threading.Lock()

class Visualizer:
    ''' 
    Attributes
    ---------
    - fig : 캔버스 같은 역할을 하는 matplotlib의 Figure 클래스 객체
    - axes : 차트를 그리기 위한 maplotlib의 Axes 클랙스 객체
    - title : 가시화될 그림의 제목
    
    Functions
    ---------
    - prepare() : Figure를 초기화하고 일봉 차트를 출력
    - plot() : 일봉 차트를 제외한 나머지 차트를 출력
    - save() : Figre를 그림 파일로 저장
    - clear() : 일봉 차트를 제외한 나머지 차트를 초기화
    
    Returns
    ---------
    - Figure 제목 : 파라미터, 에포크 및 탐험률
    - Axes 1 : 종목의 일봉 차트
    - Axes 2 : 보유 주식 수 및 에이전트 행동 차트
    - Axes 3 : 가치 신경망 출력
    - Axes 4 : 정책 신경망 출력 및 탐험 차트
    - Axes 5 : 포트폴리오 가치 및 학습 지점 차트   
    '''
    
    COLORS = ['r', 'b', 'g']
    
    def __init__(self):
        self.canvas = None
        # 캔버스 같은 역할을 하는 matplotlib의 Figure 클래스
        self.fig = None
        # 차트를 그리기 위한 matplotlib의 Axes 클래스
        self.axes = None
        self.title = ''     # 그림 제목
        self.x = []
        self.xticks = []
        self.xlabels = []
        
    def prepare(self, chart_data, title):
        self.title = title
        # 모든 차트가 공유할 x축 데이터
        self.x = np.arange(len(chart_data['date']))
        self.x_label = [datetime.strptime(date, '%Y%m%d').date() for date in chart_data['date']]
        with lock:
            # 캔버스를 초기화하고 5개 차트를 그릴 준비
            self.fig, self.axes = plt.subplots(
                nrows=5, ncols=1, facecolor='w', sharex=True
            )
            for ax in self.axes:
                # 보기 어려운 과학적 표기 비활성화
                ax.get_xaxis().get_major_formatter().set_scientific(False)
                ax.get_yaxis().get_major_formatter().set_scientific(False)
                # y axis 위치 오른쪽으로 변경
                ax.yaxis.tick_right()
            
            # 차트 1. 일봉 차트
            self.axes[0].set_ylabel('Env.')     # y축 레이블 표시
            x = np.arange(len(chart_data))
            # open, high, low, close 순서로 2차원 배열
            ohlc = np.hstack((
                x.reshape(-1, 1), np.array(chart_data)[:, 1:-1]
            ))
            # 양봉은 빨간색으로 음봉은 파란색으로 표시
            candlestick_ohlc(self.axes[0], ohlc, colorup='r', colordown='b')
            # 거래량 가시화
            ax = self.axes[0].twinx()
            volume = np.array(chart_data)[:, -1].tolist()
            ax.bar(x, volume, color='b', alpha=0.3)
            # x축 설정
            self.x = np.arange(len(chart_data['date']))
            self.xticks = chart_data.index[[0, -1]]
            self.xlabels = chart_data.oloc[[0, -1]]['date']
            
    def plot(self, epoch_str=None, num_epochs=None, epsilon=None,
             action_list=None, actions=None, num_stocks=None,
             outvals_value=[], outvals_policy=[], exps=None,
             initial_balance=None, pvs=None):
        '''
        Attributes
        ---------
        - epoch_str: Figure 제목으로 표시할 에포크
        - num_epochs : 총 수행할 에포크 수
        - epsilon : 탐험률
        - action_list : 에이전트가 수행할 수 있는 전체 행동 리스트
        - actions : 에이전트가 수행한 행동 배열
        - num_stocks : 주식 보유 수 배열
        - outvals_value : 가치 신경망의 출력 배열
        - outvals_policy : 정책 신경망의 출력 배열
        - exps : 탐험 여부 배열
        - initial_balance : 초기 자본금
        - pvs : 포트폴리오 가치 배열 
        '''
        # matplotlib이 numpy() 배열을 받으므로 리스트를 모두 numpy 배열로 변환
        
        with lock:
            # actions, num_stocks, outvals_value, outvals_policy, pvs 크기가 모두 같음
            # actions의 크기만큼 배열을 생성해 x축으로 사용
            actions = np.array(actions)     # 에이전트의 행동 배열
            # 가치 신경망의 출력 배열
            outvals_value = np.array(outvals_value)
            # 정책 신경망의 출력 배열
            outvals_policy = np.array(outvals_policy)
            # 초기 자본금 배열
            pvs_base = np.zeros(len(actions)) + initial_balance
            
            # 차트 2. 에이전트 상태 (행동, 보유 주식 수)
            for action, color in zip(action_list, self.COLORS):
                for i in self.x[actions == action]:
                    # 배경 색으로 행동 표시 : 매수 빨간색, 매도 파란색
                    self.axes[1].axvline(i, color=color, alpha=0.1)
            self.axes[1].plot(self.x, num_stocks, '-k')     # 보유 주식 수 그리기
            
            # 차트 3. 가치 신경망 (행동에 대한 예측 가치를 라인 차트로 그림)
            if len(outvals_value) > 0:
                max_actions = np.argmax(outvals_value, axis=1)
                for action, color in zip(action_list, self.COLORS):
                    # 배경 그리기
                    for idx in self.x:
                        if max_actions[idx] == action:
                            self.axes[2].axvline(idx, color=color, alpha=0.1)
                    # 가치 신경망 출력 그리기
                    ## 매수는 빨간색, 매도는 파란색, 관망은 초록색
                    ## 가치를 예측할 행동에 관망이 없으면 초록색 라인 차트는 그리지 않음
                    self.axes[2].plot(self.x, outvals_value[:, action], color=color, linestyle='-')
            
            # 차트 4. 정책 신경망 
            # 탐험을 노란색 배경으로 그리기
            for exp_idx in exps:
                self.exes[3].axvline(exp_idx, color='y')
            # 행동을 배경으로 그리기
            _outvals = outvals_policy if len(outvals_policy) > 0 else outvals_value
            for idx, outval in zip(self.x, _outvals):
                color = 'white'
                if np.isnan(outval.max()):
                    continue
                # 탐험하지 않은 지점에서 매수 빨간색, 매도 파란색
                if outval.argmax() == Agent.ACTION_BUY:
                    color = self.COLORS[0]      # 매수 빨간색
                elif outval.argmax() == Agent.ACTION_SELL:
                    color = self.COLORS[1]      # 매도 파란색
                elif outval.argmax() == Agent.ACTION_HOLD:
                    color = self.COLORS[2]      # 관망 초록색
                self.axes[3].axvline(idx, color=color, alpha=0.1)
            
            # 정책 신경망의 출력 그리기
            # 매수에 대한 정책 신경망 출력을 빨간색 선, 매도에 대한 정책 신경망 출력값을 파란색 선
            # 빨간색 선이 파란색 선보다 위에 위치하면 에이전트 매수, 그 반대는 매도
            if len(outvals_policy) > 0:
                for action, color in zip(action_list, self.COLORS):
                    self.axes[3].plot(
                        self.x, outvals_policy[:, action],
                        color=color, linestyle='-'
                    )
                    
            # 차트 5. 포트폴리오 가치
            # 초기자본금 가로 일직선으로 표시
            self.axes[4].axhline(
                initial_balance, linestyle='-', color='gray'
            )
            # 초기 자본금보다 높은 부분은 빨간색 배경
            self.axes[4].fill_between(
                self.x, pvs, pvs_base,
                where=pvs > pvs_base, facecolor='r', alpha=0.1
            )
            # 초기 자본금보다 낮은 부분은 파란색 배경
            self.axes[4].fill_between(
                self.x, pvs, pvs_base,
                where=pvs < pvs_base, facecolor='b', alpha=0.1
            )
            self.axes[4].plot(self.x, pvs, '-k')
            self.axes[4].xaxis.set_ticks(self.xticks)
            self.axes[4].xaxis.set_ticklabels(self.xlabels)
            
            # 에포크 및 탐험 비율
            self.fig.suptitle(f'{self.title}\nEPOCH:{epoch_str}/{num_epochs} EPSILON:{epsilon:.2f}')
            # 캔버스 레이아웃 조정
            self.fig.tight_layout()
            self.fig.subplots_adjust(top=0.85)
    
    # 가시 정보 초기화
    def clear(self, xlim):
        with lock:
            _axes = self.axes.tolist()
            # 변하지 않는 값 외 그 외 차트 초기화
            for ax in _axes[1:]:
                ax.cla()       # 그린 차트 지우기
                ax.relim()      # limit 초기화
                ax.autoscale()  # 스케일 재설정
            
            # y축 레이블 재설정
            self.axes[1].set_ylabel('Agent')
            self.axes[2].set_yalbel('V')
            self.axes[3].set_ylabel('P')
            self.axes[4].set_ylabel('PV')
            for ax in _axes:
                ax.set_xlim(xlim)       # x축 limit 재설정
                ax.get_xaxis().get_major_formatter().set_scientific(False)  # x축의 과학적 표기 비활성화
                ax.get_yaxis().get_major_formatter().set_scientific(False)  # y축의 과학적 표기 비활성화
                # x축 간격을 일정하게 설정
                ax.xticklabel_format(useOffset=False)
                
    # 결과 저장
    def save(self, path):
        with lock:
            self.fig.savefig(path)

                
            
            
            
            
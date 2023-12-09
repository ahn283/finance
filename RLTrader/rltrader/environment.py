class Environment:
    '''
    Attributions
        chart_data : 주식 종목의 차트 데이터
        observation : 현재 관측치
        idx : 차트 데이터에서의 현재 위치
    
    Functions
        reset() : idx와 observation을 초기화
        observe() : idx를 다음 위치로 이동하고 observation을 업데이트
        get_price() : 현재 observation에서 종가를 획득
    '''
    PRICE_IDX = 4       # 종가의 위치
    
    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.idx = -1
    
    def reset(self):
        self.observation = None
        self.idx = -1
        
    def observe(self):
        # 하루 앞으로 이동하여 차트 데이터에서 observation(관측 데이터)를 제공
        # 더 이상 제공할 데이터가 없을 때는 None을 반환
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None
    
    def get_price(self):
        # 관측 데이터로부터 종가를 가져와서 반환
        # 종가 close가 5번째 열이기 때문에 PRICE_IDX=4
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None
        
    
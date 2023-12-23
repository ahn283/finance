import os
import sys
import logging
import argparse
import json


from rltrader import settings
from rltrader import utils
from rltrader import data_manager
# from RLTrader.rltrader import settings
# from RLTrader.rltrader import utils
# from RLTrader.rltrader import data_manager

# 프로그램 인자 숼정

os.environ['RLTRADER_BASE'] = 'C:\project\github\finance\RLTrader\rltrader'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 실행모드 : train, test, update, predict 중 하나로 정하며 이 값에 따라 학습기에 입력할 파라미터 수정
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'predict'], default='train')
    # RL Trader 버전을 명시
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3', 'v4', 'v4.1', 'v4.2'], default='v4.1')
    # 로그 등의 풀력물을 저장할 폴더명과 모델 파일명에 사용하는 문자열
    parser.add_argument('--name', default=utils.get_time_str())
    # 강화학습의 환경이 될 주식의 종목코드, A3C의 경우 여러 개의 종목 코드 입력
    parser.add_argument('--stock_code', nargs='+')
    # 강화학습 방식을 선택
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'ppo', 'monkey'], default='a2c')
    # 가치신명망와 정책신경망에서 사용할 신경망 유형
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn', 'monkey'], default='dnn')
    # keras의 백엔드로 사용할 프레임워크 선택
    parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'plaidml'], default='pytorch')
    # 차트 데이터 및 학습 데이터 시작 날짜 
    parser.add_argument('--start_date', default='20200101')
    # 차트 데이터 및 학습 데이터 끝 날짜
    parser.add_argument('--end_date', default='20201231')
    # learning rate
    parser.add_argument('--lr', type=float, default=0.0001)
    # discount factor
    parser.add_argument('--discount_factor', type=float, default=0.7)
    # 초기 자본금
    parser.add_argument('--balance', type=int, default=100000000)
    args = parser.parse_args()

    # 학습기 파라미터 설정
    # 로그, 가시화 파일 등의 출력 파일을 저장할 폴더 이름 구성 
    output_name = f'{args.mode}_{args.name}_{args.rl_method}_{args.net}'
    # 강화학습 유무를 지정하는 플래그
    learning = args.mode in ['train', 'update']
    # 모델 재사용 여부 플래그
    reuse_models = args.mode in ['test', 'update', 'predict']
    value_network_name = f'{args.name}_{args.rl_method}_{args.net}_value.mdl'
    policy_network_name = f'{args.name}_{args.rl_method}_{args.net}_policy.mdl'
    start_epsilon = 1 if args.mode in ['train', 'update'] else 0
    num_epochs = 1000 if args.mode in ['train', 'update'] else 1
    num_steps = 5 if args.net in ['lstm', 'cnn'] else 1
    
    # Backend 설정
    os.environ['RLTRADER_BACKEND'] = args.backend
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKED'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
        
    # 출력 경로 생성
    output_path = os.path.join(settings.BASE_DIR, 'output', output_name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    # 파라미터 기록
    # 입력받은 프로그램 인자를 json 형태로 저장 후, 프로그램 인자를 딕셔너리로 만든 다음 params.json 파일에 저장한다. 
    # 위 출력 경로에 로그파일 생성한다.
    params = json.dumps(vars(args))
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(params)
        
    # 모델 경로 준비
    # 모델 포맷은 tensorflow는 h5, pytorch는 pickle
    value_network_path = os.path.join(settings.BASE_DIR, 'models', value_network_name)
    policy_network_path = os.path.join(settings.BASE_DIR, 'models', policy_network_name)    
    
        
    # 로그 기록 설정
    # 로그 레벨 DEBUG < INFO < WARNING < ERROR < CRITICAL 5단계 중 DEBUG 이상의 로그 기록
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(settings.LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info(params)
    
    # backend 설정, 로그 설정을 먼저하고 RLTrader 모듈들을 임포트해야 함
    from rltrader.learners import ReinforcementLearner, DQNLearner, PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner, PPOLearner
    
    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_price = []
    list_max_trading_price = []
    
    for stock_code in args.stock_code:
        # 차트 데이터, 학습 데이터 준비
        chart_data, training_data = data_manager.load_data(
            stock_code, args.start_date, args.end_date, ver=args.ver
        )
        
        assert len(chart_data) >= num_steps
        
        # 최소/최대 단일 매매 금액 정책
        min_trading_price = 10000
        max_trading_price = 10000000
        
        # 공통 파라미터 설정
        common_params = {
            'rl_method': args.rl_method,
            'net': args.net, 'num_steps': num_steps, 'lr': args.lr,
            'balance': args.balance, 'num_epochs': num_epochs,
            'discount_factor': args.discount_factor, 'start_epsilon': start_epsilon,
            'output_path': output_path, 'reuse_models': reuse_models
        }
        
        # 강화학습 시작
        # A3C에서는 여러 주식 종목의 강화학습을 병렬로 진행하기 떄문에 종목 코드, 차트 데이터, 학습 데이터, 최소 및 최대 투자 단위를 리스트로 보관한다.
        learner = None
        if args.rl_method != 'a3c':
            common_params.update({
                'stock_code': stock_code,
                'chart_data': chart_data,
                'training_data': training_data,
                'min_trading_price': min_trading_price,
                'max_trading_price': max_trading_price
            })
        
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params, 'value_network_path': value_network_path})
            
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params, 'policy_network_path': policy_network_path})
                
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params, 'value_network_path': value_network_path, 'policy_network_path': policy_network_path})
            
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params,
                                        'value_network_path': value_network_path,
                                        'policy_network_path': policy_network_path})
            
            elif args.rl_method == 'ppo':
                learner = PPOLearner(**{**common_params,
                                        'value_network_path': value_network_path,
                                        'policy_network_path': policy_network_path})
            
            elif args.rl_method == 'monkey':
                common_params['net'] = args.rl_method
                common_params['num_epochs'] = 10
                common_params['start_epsilon'] = 1
                learning = False
                learner = ReinforcementLearner(**common_params)
                
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_price.append(min_trading_price)
            list_max_trading_price.append(max_trading_price)
    
    # A3CLearner 클래스는 차트 데이터, 학습 데이터 등의 인자를 리스트로 받으며, 이 리스트의 크기만큼 A2CLearner 인스턴스를 생성        
    if args.rl_method == 'a3c':
        learner = A3CLearner(**{
            **common_params,
            'list_stock_code': list_stock_code,
            'list_chart_data': list_chart_data,
            'list_training_data': list_training_data,
            'list_min_trading_price': list_min_trading_price,
            'list_max_trading_price': list_max_trading_price,
            'value_network_path': value_network_path,
            'policy_network_path': value_network_path
        })
        
    # 학습 실행
    # 학습기가 None 아닌지 확인
    assert learner is not None
    
    if args.mode in ['train', 'test', 'update']:
        learner.run(learning=learning)
        if args.mode in ['train', 'update']:
            learner.save_models()
    elif args.mode == 'predict':
        learner.predict()


        
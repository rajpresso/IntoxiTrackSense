
# 데이터 관련 설정
config_path = "./config/config.py" 

device = "cuda:1"

# ['xm','ym','m1', 's1', 'm2', 's2', 'm3', 's3', 'm4', 's4', 'm5', 's5','m6', 's6', 'm7', 's7', 'm8', 's8', 'm9', 's9', 'm10', 's10', 'm11', 's11', 'm12', 's12', 'm13', 's13', 'm14', 's14', 'm15', 's15', 'm16', 's16', 'm17', 's17', 'y']
DATA_CONFIG = {
    "train_tb_data":'',
    "test_tb_data": '',
    "data_version": '',
    "seq_length": 90,                # 시퀀스 길이
    "coordinate_cols": ['frame','m1', 's1', 'm2', 's2', 'm3', 's3', 'm4', 's4', 'm5', 's5','m6', 's6', 'm7', 's7', 'm8', 's8', 'm9', 's9', 'm10', 's10', 'm11', 's11', 'm12', 's12', 'm13', 's13', 'm14', 's14', 'm15', 's15', 'm16', 's16', 'm17', 's17', 'y'],
    "group_col": ["label","filename"],
    "target_col": "y"
}


# LSTM 모델 관련 설정
MODEL_CONFIG = {
    "hidden_dim": 128,     # LSTM hidden layer size
    "num_layers": 1,       # LSTM 레이어 수
    "dropout": 0.0,        # 드롭아웃 비율 
    "output_dim": 1,        # 출력 크기 (회귀 또는 분류)
    "sigmoid" : 0.5
}

TRAIN_DATA_CONFIG = {
    'valid_size': 0.2, 
    'scaler': 'std'   # std= StandardScaler , nor = Normalizer
}

# 학습 관련 설정
TRAINING_CONFIG = {
    "batch_size": 32,      # 배치 크기
    "epochs": 90,          # 학습 에포크
    "learning_rate": 0.0001,  # 학습률
    "save_path": "./models/saved_models",  # 모델 저장 경로
    "model_name":"lstm_model.pth"
}

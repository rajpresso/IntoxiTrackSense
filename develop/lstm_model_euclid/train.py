# train.py

import torch
import pandas as pd

from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader

# 프로젝트 구성 요소 임포트
from config.config import DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, config_path


from util.util import (
    load_data,
    center_node,
    transform_keypoints,
    create_sequences,
    train_test_set
)

from  util.data_import import import_data

from model.lstm_model import (
    BinaryLSTMModel,
    evaluate,
    train_model,
    save_model_and_config,
    evaluate_and_save_confusion_matrix
)

def train():
    # 디바이스 설정: CUDA 또는 CPU
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # 데이터 경로 및 시퀀스 길이 설정
    #data_path = DATA_CONFIG["data_path"]
    #print(f"Data path: {data_path}")

    # CSV 파일 읽어오기
    # df = pd.read_csv(data_path)
    
    # df = import_df()
    #######################
    version = '3.0'
    category = 'normal'
    df_n = import_data(version, category)
    df_n['y'] = 0
    print("y = 0 데이터 import 완료 ")
    print(f'version:{version}, category : {category} ')
    print(f'df.shape y=0 : {df_n.shape} ')
    #######################
    version = '2.0'
    category = 'croki'
    df_c = import_data(version, category)
    df_c['y'] = 1
    #######################
    print("y= 1 데이터 import 완료 ")
    print(f'version:{version}, category : {category} ')
    print(f'df.shape y=1 : {df_c.shape} ')
    #######################


    if df_c.empty or df_n.empty:
        print('데이터 empty')
        return 

    # 1. 공통 컬럼 추출
    common_columns = df_c.columns.intersection(df_n.columns)

    # 2. 공통 컬럼만 유지
    df1_filtered = df_c[common_columns]
    df2_filtered = df_n[common_columns]

    # 3. 위아래로 결합 (concat)
    df = pd.concat([df1_filtered, df2_filtered], ignore_index=True)


    # 키포인트 데이터의 중심점 계산
    xm, ym = center_node(df)

    # 키포인트 데이터를 기울기와 유클리드 거리로 변환
    mdf = transform_keypoints(df, xm, ym)

    # 시퀀스 데이터 생성
    seq_length = DATA_CONFIG["seq_length"]
    X_seq, y_seq = create_sequences(mdf, seq_length)
    print(f'test data X_seq.shpae : {X_seq.shape}, y_seq.shape : {y_seq.shape}')
    # print("X_seq shape:", X_seq.shape)  # 입력 데이터 형태 확인
    # print("y_seq shape:", y_seq.shape)  # 타겟 데이터 형태 확인

    # 데이터셋 분리 및 DataLoader 생성
    train_loader, valid_loader = train_test_set(X_seq, y_seq)

    # 모델 학습
    model = train_model(train_loader, valid_loader, X_seq, device)

    # 학습된 모델과 설정 저장
    save_dir = save_model_and_config(model)

    # 
    # 모델 평가 및 Confusion Matrix 저장
    # evaluate_and_save_confusion_matrix(BinaryLSTMModel, valid_loader, save_dir, X_seq)
    testing(save_dir)

def testing(save_dir):
    # df = pd.read_csv('/home/alpaco/project/jsw_model/sss.csv')
    version = 'test3.0'
    category = 'test'
    df = import_data(version, category)
    print(f'version:{version}, category : {category} ')
    print(f'테스트용 df.shape : {df.shape} ')
    
    
    # 키포인트 데이터의 중심점 계산
    xm, ym = center_node(df)

    # 키포인트 데이터를 기울기와 유클리드 거리로 변환
    mdf = transform_keypoints(df, xm, ym)

    # 시퀀스 데이터 생성
    seq_length = DATA_CONFIG["seq_length"]
    X_seq, y_seq = create_sequences(mdf, seq_length)
    print('테스트용 ')
    print(f'test data X_seq.shpae : {X_seq.shape}, y_seq.shape : {y_seq.shape}')

        # 6. 데이터 PyTorch 텐서로 변환 및 데이터로더 준비
    # 학습 데이터와 테스트 데이터를 PyTorch 텐서로 변환하여 모델 학습에 사용합니다.
    train_X_tensor = torch.FloatTensor(X_seq)
    train_y_tensor = torch.LongTensor(y_seq)

    # PyTorch의 DataLoader를 사용해 데이터를 묶어 관리할 수 있습니다.
    batch_size = TRAINING_CONFIG['batch_size']  # 배치 사이즈는 한 번에 학습하는 데이터 개수를 뜻합니다.
    train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
    test_loder =  DataLoader(train_dataset,batch_size = batch_size)


    # 데이터셋 분리 및 DataLoader 생성
    # train_loader, valid_loader = train_test_set(X_seq, y_seq)


    # 모델 평가 및 Confusion Matrix 저장
    evaluate_and_save_confusion_matrix(BinaryLSTMModel, test_loder, save_dir, X_seq)




if __name__ == "__main__":
    # train 함수 실행
    train()
    # testing("")

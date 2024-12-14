# util/util.py

import pandas as pd
import numpy as np
import math
from config.config import DATA_CONFIG
from config.config import TRAIN_DATA_CONFIG
from config.config import TRAINING_CONFIG


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def train_test_set(X_seq, y_seq):
    print('train_test_set')
    TRAIN_DATA_CONFIG

    # 데이터셋 분리 및 셔플
    # 학습 데이터와 테스트 데이터로 나누고, 라벨의 비율을 유지합니다.
    tsize = TRAIN_DATA_CONFIG['valid_size'] 
    train_X, valid_X, train_y, valid_y = train_test_split(X_seq, y_seq, test_size=tsize, stratify=y_seq, random_state=42)

    # Scaler를 사용하여 정규화(표준화)
    if TRAIN_DATA_CONFIG['scaler'] == 'std':
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
        valid_X = scaler.transform(valid_X.reshape(-1, valid_X.shape[-1])).reshape(valid_X.shape)
    elif TRAIN_DATA_CONFIG['scaler'] == 'nor':
        scaler = Normalizer()
        train_X = scaler.fit_transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
        valid_X = scaler.transform(valid_X.reshape(-1, valid_X.shape[-1])).reshape(valid_X.shape)
    else:
        pass

    # 학습 데이터를 다시 셔플하여 모델이 순서에 너무 의존하지 않도록 합니다.
    train_indices = np.arange(len(train_X))
    np.random.shuffle(train_indices)
    train_X, train_y = train_X[train_indices], train_y[train_indices]

    # 최댓값 출력
    print("Max values before converting to tensors:")
    print("Max value in train_X:", np.nanmax(train_X))  # ignore NaN for max
    print("Max value in valid_X:", np.nanmax(valid_X))  # ignore NaN for max

    # 6. 데이터 PyTorch 텐서로 변환 및 데이터로더 준비
    # 학습 데이터와 테스트 데이터를 PyTorch 텐서로 변환하여 모델 학습에 사용합니다.
    train_X_tensor = torch.FloatTensor(train_X)
    train_y_tensor = torch.LongTensor(train_y)
    valid_X_tensor = torch.FloatTensor(valid_X)
    valid_y_tensor = torch.LongTensor(valid_y)

    # PyTorch의 DataLoader를 사용해 데이터를 묶어 관리할 수 있습니다.
    batch_size = TRAINING_CONFIG['batch_size']  # 배치 사이즈는 한 번에 학습하는 데이터 개수를 뜻합니다.
    train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
    valid_dataset = TensorDataset(valid_X_tensor, valid_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

#############################################################################

# 시퀀스 데이터 생성 
def create_sequences(data, seq_length):
    print('create_sequences')

    # 데이터 설정
    coordinate_cols = DATA_CONFIG["coordinate_cols"]
    target_col = DATA_CONFIG["target_col"]
    label_col = DATA_CONFIG["group_col"][0]
    filename = DATA_CONFIG["group_col"][1]


    print("Coordinate columns:", coordinate_cols)
    print("Target column:", target_col)
    print("Label column:", label_col)
    print("filename column:", filename)
    print("Data columns:", data.columns.tolist())


    # 실제 데이터에 존재하는 coordinate_cols만 사용
    available_cols = [col for col in coordinate_cols if col in data.columns]
    missing_cols = set(coordinate_cols) - set(data.columns)
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")

    # 학습 컬럼 확인
    print("Using columns for sequences:", available_cols)

    xs, ys = [], []

    available_cols.remove(target_col)
    for _, group in data.groupby([filename,label_col]):
        group = group.sort_values(by=['frame']).reset_index(drop=True)
        data_X = group[available_cols].values  # 선택된 컬럼만 사용
        data_y = group[target_col].values

        # 시퀀스 생성
        for i in range(len(data_X) - seq_length + 1):
            x = data_X[i:i + seq_length]
            y = data_y[i + seq_length - 1]  # 마지막 시퀀스 값 사용
            xs.append(x)
            ys.append(y)

    return np.array(xs), np.array(ys)

#############################################################################

# 기울기와 유클리안거리로 변경
def transform_keypoints(df, xm, ym):
    print('transform_keypoints')
    mdf = pd.DataFrame()
    # 무한대 방지를 위한 작은 값
    epsilon = 1e-6
    # x1부터 x17까지의 컬럼 값을 x_mid로 뺀 값의 제곱으로 업데이트
    for i in range(1, 18):
        m_col = f'm{i}'  # 기울기
        s_col = f's{i}'  # 유클리드 거리
        x_col = f'x{i}'
        y_col = f'y{i}'
        # 기울기 계산 (소수점 3자리까지 반올림)
        mdf[m_col] = ((df[y_col] - ym) / (df[x_col] - xm + epsilon)).round(3)  
        # 거리 계산 (소수점 3자리까지 반올림)
        mdf[s_col] = ((df[x_col] - xm) ** 2 + (df[y_col] - ym) ** 2).apply(math.sqrt).round(3)
    
    mdf['label'] = df['label']
    mdf['filename'] = df['filename'] 
    mdf['frame'] = df['frame']
    mdf['y'] = df['y'] 
    mdf['xm'] = xm
    mdf['ym'] = ym
    # 결측치 제거
    mdf = mdf.fillna(0)
    return mdf

## 중간점 계산 
def center_node(df):
    print('center_node')
    # Series 연산으로 xm, ym 계산
    xm = (
        df['x12'].where(df['x11'] == 0, df['x11'].where(df['x12'] == 0, (df['x11'] + df['x12']) / 2))
    )
    ym = (
        df['y12'].where(df['y11'] == 0, df['y11'].where(df['y12'] == 0, (df['y11'] + df['y12']) / 2))
    )
    return xm,ym

#CSV 파일을 로드하고 시퀀스 데이터 생성
def load_data():
    print('load_data')
    data_path = DATA_CONFIG["data_path"]
    seq_length = DATA_CONFIG["seq_length"]
    print(f'data_path : {data_path} ')
    # csv 파일 읽어오기
    df = pd.read_csv(data_path)
    # 키포인트의 중간점 
    xm , ym = center_node(df)
    # 기울기와 유클리안거리로 변경
    mdf = transform_keypoints(df, xm, ym)
    #시퀀스 데이터 생성
    X, y = create_sequences(mdf, seq_length)
    return X, y
    
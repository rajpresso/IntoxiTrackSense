import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

# Device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformer 모델 정의
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = x.transpose(0, 1)  # (seq_len, batch_size, features)
        attn_output, _ = self.attention(x, x, x)
        transformer_output = self.transformer(attn_output)
        output = self.fc(transformer_output[-1, :, :])  # 마지막 시퀀스 출력
        return output

# 데이터 전처리 함수
def preprocess_data(df, coordinate_cols, scaler_type):
    X = df[coordinate_cols].values
    if scaler_type == 'StandardScaler':
        scaler_X = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler_X = MinMaxScaler()
    else:
        raise ValueError("Unknown scaler type")
    X_normalized = scaler_X.fit_transform(X)
    df[coordinate_cols] = X_normalized
    return df

# 시퀀스 생성 함수
def create_sequences(df, seq_length):
    xs, ys = [], []
    for _, group in df.groupby(['FILENAME', 'label']):
        group = group.sort_values(by=['frame']).reset_index(drop=True)
        data_X = group.drop(columns=['frame', 'FILENAME', 'label', 'y'], errors='ignore').values
        data_y = group['y'].values
        for i in range(len(data_X) - seq_length + 1):
            x = data_X[i:i + seq_length]
            y = data_y[i + seq_length - 1]
            xs.append(x)
            ys.append(y)
    return np.array(xs), np.array(ys)

# 훈련 함수
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output.squeeze(), batch_y.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    # # 모델 저장
    # model_save_path = './transformer.pt'
    # torch.save(model.state_dict(), model_save_path)
    # print(f"Model saved to {model_save_path}")

# 평가 함수
def evaluate_model(model, test_loader):
    # 모델 불러오기
    # loaded_model = TransformerModel(input_size, hidden_size, 1, num_heads, num_layers)
    # loaded_model.load_state_dict(torch.load(model_save_path))
    # loaded_model.to(device)
    # loaded_model.eval()  # 평가 모드로 전환
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            all_preds.extend(preds.astype(int).squeeze())
            all_labels.extend(labels.cpu().numpy().astype(int).squeeze())
            correct += np.sum(preds.astype(int).squeeze() == labels.cpu().numpy())
            total += labels.size(0)
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f'Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')
    return all_preds, all_labels, f1, precision, recall

# Main workflow
pram = pd.read_csv('/home/alpaco/project/jsw_model/parameter_combinations.csv')
angle_df = pd.read_csv('./degree_combined.csv')
angle_df.drop('Unnamed: 0', axis=1, inplace=True)

st, en = 1, 2
coordinate_cols = ['right_arm', 'left_arm', 'right_leg', 'left_leg']

for index in range(st, en):
    row = pram.iloc[index]
    print(f"Index: {index}, Parameters: {row.to_dict()}")

    # Hyperparameters
    num_heads = row[5]
    num_layers = row[6]
    lr = row[3]
    scaler_type = row[4]
    optimizer_type = row[2]
    hidden_size = row[1]

    # 데이터 전처리
    angle_df = preprocess_data(angle_df, coordinate_cols, scaler_type)

    # 시퀀스 생성
    sequence_length = 90
    X_seq, Y_seq = create_sequences(angle_df, sequence_length)
    train_X, valid_X, train_y, valid_y = train_test_split(X_seq, Y_seq, test_size=0.2, stratify=Y_seq, random_state=42)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(train_X), torch.LongTensor(train_y)), batch_size=16, shuffle=True)
    valid_loader = DataLoader(TensorDataset(torch.FloatTensor(valid_X), torch.LongTensor(valid_y)), batch_size=16, shuffle=False)

    # 모델 및 손실 함수
    input_size = len(coordinate_cols)
    model = TransformerModel(input_size, hidden_size, 1, num_heads, num_layers)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = getattr(torch.optim, optimizer_type)(model.parameters(), lr)

    # 훈련
    train_model(model, train_loader, criterion, optimizer, epochs=3)

    # 평가
    test_df = pd.read_csv('./degree_test.csv')
    test_df.drop('Unnamed: 0', axis=1, inplace=True)
    test_df = preprocess_data(test_df, coordinate_cols, scaler_type)
    test_X_seq, test_y_seq = create_sequences(test_df, sequence_length)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(test_X_seq), torch.LongTensor(test_y_seq)), batch_size=16, shuffle=False)

    _, _, f1, precision, recall = evaluate_model(model, test_loader)

    # 결과 저장
    save_dir = './log'
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"No.{index}_hidden_dim_{hidden_size}_optimizer_{optimizer_type}_lr_{lr}_scaler_{scaler_type}_f1_{f1:.3f}_precision_{precision:.2f}_recall_{recall:.2f}.txt"
    with open(os.path.join(save_dir, file_name), 'w') as f:
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"F1 Score: {f1:.2f}\n")
    print(f"Saved results to {file_name}")

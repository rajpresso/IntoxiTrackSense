import os
import json
import shutil
from datetime import datetime

import torch
import torch.nn as nn
# from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay,
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from config.config import TRAINING_CONFIG, MODEL_CONFIG, config_path

# 디바이스 설정
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# LSTM 모델 정의
class BinaryLSTMModel(nn.Module):
    """
    이진 분류를 위한 LSTM 모델 정의.
    """
    def __init__(self, input_size, hidden_size, num_layers,dropout):
        super(BinaryLSTMModel, self).__init__()
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 이진 분류 (1개의 출력 노드)

    def forward(self, x):
        # 초기 hidden state와 cell state 생성
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # LSTM 통과
        out, _ = self.lstm(x, (h0, c0))

        # 마지막 타임스텝의 출력 사용
        out = self.fc(out[:, -1, :])
        return out


# 모델 학습 함수
def train_model(train_loader, valid_loader, X_seq, device):
    """
    LSTM 모델을 학습하고 검증 데이터로 성능을 평가.
    """
    # 모델 초기화
    input_size = X_seq.shape[2]
    hidden_size = MODEL_CONFIG['hidden_dim']
    num_layers = MODEL_CONFIG['num_layers']
    droupout = MODEL_CONFIG['dropout']
    model = BinaryLSTMModel(input_size, hidden_size, num_layers,droupout).to(device)

    # 손실 함수 및 옵티마이저
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])

    # 학습 루프
    for epoch in range(TRAINING_CONFIG['epochs']):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Epoch별 결과 출력
        f1 = evaluate(model, valid_loader)
        print(f"Epoch [{epoch+1}/{TRAINING_CONFIG['epochs']}], Loss: {running_loss/len(train_loader):.4f}, F1 Score: {f1:.4f}")

    return model


# 모델 저장 함수
def save_model_and_config(model):
    """
    학습된 모델과 config를 저장.
    """
    # 저장 디렉토리 생성
    base_dir = TRAINING_CONFIG['save_path']
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # 모델 저장
    model_path = os.path.join(save_dir, TRAINING_CONFIG["model_name"])
    torch.save(model.state_dict(), model_path)

 

    print(f"Model and config saved in: {save_dir}")
    return save_dir


# Confusion Matrix 계산 및 저장
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
import os
import torch
import matplotlib.pyplot as plt

def evaluate_and_save_confusion_matrix(model_class, test_loader, save_dir, X_seq):
    """
    모델을 로드하여 테스트 데이터를 평가하고 Confusion Matrix를 저장.
    """
    # 모델 초기화 및 가중치 로드
    model_path = os.path.join(save_dir, TRAINING_CONFIG["model_name"])
    input_size = X_seq.shape[2]
    hidden_size = MODEL_CONFIG['hidden_dim']
    num_layers = MODEL_CONFIG['num_layers']
    droupout = MODEL_CONFIG['dropout']
    model = BinaryLSTMModel(input_size, hidden_size, num_layers, droupout).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    sigmoid = MODEL_CONFIG['sigmoid']
    # 테스트 데이터 평가
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy() > sigmoid # 0.5
            all_preds.extend(preds.astype(int))
            all_labels.extend(labels.cpu().numpy().astype(int))

        # Confusion Matrix 계산
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])

        # 시각화 및 저장
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")

        # Precision, Recall, F1, Accuracy 계산
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        accuracy = accuracy_score(all_labels, all_preds)

        # 그래프 하단부에 Precision, Recall, F1, Accuracy, save_dir 표시
        metrics_text = (f"Precision: {precision:.2f} | Recall: {recall:.2f} | "
                        f"F1 Score: {f1:.2f} | Accuracy: {accuracy:.2f}")
        save_dir_text = f" {save_dir}"
        plt.figtext(0.5, -0.1, metrics_text, wrap=True, horizontalalignment='center', fontsize=10)
        plt.figtext(0.5, -0.15, save_dir_text, wrap=True, horizontalalignment='center', fontsize=10)

        # 그래프 저장
        cm_image_path = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(cm_image_path, bbox_inches="tight")
        print(f"Confusion Matrix image saved at: {cm_image_path}")

        # 결과 출력
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}')
        print(f'Accuracy: {accuracy:.2f}')
        print(f"Results saved in: {save_dir}")
        plt.close()

        # config 파일 저장
        config_save_path = os.path.join(save_dir, "config.txt")
        shutil.copy(config_path, config_save_path)

        # config 파일에 추가 정보 쓰기
        with open(config_save_path, 'a') as f:
            f.write("\n# Evaluation Results\n")
            f.write(f"Precision: {precision:.2f}\n")
            f.write(f"Recall: {recall:.2f}\n")
            f.write(f"F1 Score: {f1:.2f}\n")
            f.write(f"Accuracy: {accuracy:.2f}\n")
            f.write(f"Results saved in: {save_dir}\n")



# 모델 평가 함수
def evaluate(model, val_loader):
    """
    검증 데이터에 대한 F1 Score 계산.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
            all_preds.extend(preds.astype(int))
            all_labels.extend(labels.cpu().numpy().astype(int))

    return f1_score(all_labels, all_preds)

# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNGRUHybridModel(nn.Module):
    """
    송폼 인식을 위한 CNN-GRU 하이브리드 모델입니다.
    오디오 특징 시퀀스를 입력으로 받아 각 프레임에 대한 송폼 클래스를 예측합니다.
    """
    def __init__(self, input_feature_dim, num_classes, cnn_out_channels, gru_hidden_size, gru_num_layers, dropout_rate):
        super(CNNGRUHybridModel, self).__init__()

        self.input_feature_dim = input_feature_dim
        self.num_classes = num_classes
        self.cnn_out_channels = cnn_out_channels
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.dropout_rate = dropout_rate

        # CNN 레이어 정의
        # 입력 특징은 (Batch, Sequence_Length, Features) 형태입니다.
        # CNN은 (Batch, Channels, Height, Width) 형태를 기대하므로,
        # 특징을 (Batch, 1, Sequence_Length, Features)로 reshape하거나,
        # Conv1d를 사용하여 (Batch, Features, Sequence_Length) 형태로 입력받습니다.
        # 여기서는 Conv1d를 사용하고, 입력 특징을 (Batch, Features, Sequence_Length)로 변환합니다.
        
        # 첫 번째 Conv1d 레이어
        self.conv1 = nn.Conv1d(in_channels=input_feature_dim, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_out_channels) # 배치 정규화
        
        # 두 번째 Conv1d 레이어
        self.conv2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_out_channels * 2)

        # Max Pooling (시간 축을 따라 특징을 압축)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # GRU 레이어 정의
        # CNN 출력의 특징 차원을 계산해야 합니다.
        # Conv1d의 출력은 (Batch, Out_channels, Sequence_Length') 입니다.
        # GRU는 (Batch, Sequence_Length', Features') 형태를 기대하므로,
        # permute하여 (Batch, Sequence_Length', Features')로 변환합니다.
        
        # CNN 출력 특징 차원 계산 (예시, 실제는 모델 구조에 따라 달라짐)
        # Conv1d 두 번 후 MaxPool1d 한 번이면, Sequence_Length는 1/2로 줄어들고,
        # 특징 차원은 cnn_out_channels * 2가 됩니다.
        gru_input_size = cnn_out_channels * 2

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,  # 배치 차원이 첫 번째가 되도록 설정
            dropout=dropout_rate if gru_num_layers > 1 else 0 # 레이어가 1개면 드롭아웃 적용 안함
        )

        # 출력 레이어 (선형 레이어)
        self.fc = nn.Linear(gru_hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (Batch, Sequence_Length, Features)
        
        # CNN 입력 형태 변환: (Batch, Features, Sequence_Length)
        x = x.permute(0, 2, 1) 

        # CNN 레이어
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # GRU 입력 형태 변환: (Batch, Sequence_Length', Features')
        x = x.permute(0, 2, 1) 

        # GRU 레이어
        # GRU의 출력은 (output, h_n) 튜플입니다. output만 사용합니다.
        # output: (Batch, Sequence_Length', gru_hidden_size * num_directions)
        # h_n: (num_layers * num_directions, Batch, gru_hidden_size)
        gru_out, _ = self.gru(x)

        # 드롭아웃 적용
        gru_out = self.dropout(gru_out)

        # 출력 레이어
        # 각 시간 프레임에 대해 분류를 수행하므로, gru_out의 모든 시간 스텝을 사용합니다.
        # (Batch * Sequence_Length', gru_hidden_size) -> (Batch * Sequence_Length', num_classes)
        # 그리고 다시 (Batch, Sequence_Length', num_classes)로 reshape
        output = self.fc(gru_out)

        return output # (Batch, Sequence_Length', num_classes)

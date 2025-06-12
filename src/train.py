# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from tqdm import tqdm

from config import config
from model import CNNGRUHybridModel
from data_loader import get_data_loaders
from utils import get_label_mapping, evaluate_segmentation

def train_model(config):
    """
    송폼 인식 모델을 학습하고 검증합니다.
    """
    # 데이터 로더 및 레이블 매핑 가져오기
    train_loader, val_loader, test_loader, label_to_id, id_to_label = get_data_loaders(config)
    
    # 특징 차원 결정 (첫 번째 배치의 특징을 사용하여)
    # features: (Batch, Sequence_Length, Features)
    # ssm: (Batch, SSM_SIZE, SSM_SIZE)
    # labels: (Batch, Sequence_Length)
    sample_features, _, _ = next(iter(train_loader))
    input_feature_dim = sample_features.shape[2] # 특징 차원

    # 모델 초기화
    model = CNNGRUHybridModel(
        input_feature_dim=input_feature_dim,
        num_classes=config.NUM_CLASSES,
        cnn_out_channels=config.CNN_OUT_CHANNELS,
        gru_hidden_size=config.GRU_HIDDEN_SIZE,
        gru_num_layers=config.GRU_NUM_LAYERS,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)

    # 손실 함수 정의 (패딩 레이블 -1은 무시)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # 옵티마이저 정의
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # TensorBoard SummaryWriter 초기화
    writer = SummaryWriter(log_dir=os.path.join(config.BASE_DIR, 'runs', 'songform_recognition'))

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"모델 학습 시작 (장치: {config.DEVICE})")
    for epoch in range(config.NUM_EPOCHS):
        model.train() # 학습 모드
        total_train_loss = 0
        
        # 학습 루프
        for batch_idx, (features, ssm, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            features = features.to(config.DEVICE)
            # SSM은 현재 모델에 직접 입력되지 않지만, 나중에 활용 가능
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad() # 기울기 초기화
            
            outputs = model(features) # 모델 예측 (Batch, Sequence_Length_after_CNN, Num_Classes)
            
            # CNN 레이어에서 시퀀스 길이가 줄어들었으므로, 레이블도 동일하게 조정해야 합니다.
            # outputs의 실제 시퀀스 길이에 맞춰 labels를 자릅니다.
            # 예를 들어, CNN 풀링에 의해 시퀀스 길이가 1/4로 줄었다면, labels도 1/4로 줄입니다.
            target_sequence_length = outputs.shape[1]
            labels_aligned = labels[:, :target_sequence_length]

            # 손실 계산을 위해 출력과 레이블 형태 조정
            # outputs: (Batch * Sequence_Length_after_CNN, Num_Classes)
            # labels_aligned: (Batch * Sequence_Length_after_CNN)
            loss = criterion(outputs.view(-1, config.NUM_CLASSES), labels_aligned.contiguous().view(-1))
            
            loss.backward() # 역전파
            optimizer.step() # 가중치 업데이트
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)

        # 검증 루프
        model.eval() # 평가 모드
        total_val_loss = 0
        all_val_predictions = []
        all_val_labels = []
        
        with torch.no_grad(): # 기울기 계산 비활성화
            for batch_idx, (features, ssm, labels) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")):
                features = features.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                outputs = model(features)
                
                # 검증 루프에서도 레이블 조정 적용
                target_sequence_length = outputs.shape[1]
                labels_aligned = labels[:, :target_sequence_length]

                loss = criterion(outputs.view(-1, config.NUM_CLASSES), labels_aligned.contiguous().view(-1))
                total_val_loss += loss.item()
                
                # 예측 결과 저장 (패딩 인덱스 -1 제외)
                predictions = torch.argmax(outputs, dim=-1) # (Batch, Sequence_Length_after_CNN)
                
                # 패딩된 레이블을 제외하고 예측과 실제 레이블을 수집
                for i in range(labels_aligned.shape[0]): # 배치 내 각 샘플
                    valid_indices = (labels_aligned[i] != -1)
                    all_val_predictions.extend(predictions[i][valid_indices].cpu().numpy())
                    all_val_labels.extend(labels_aligned[i][valid_indices].cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

        # 검증 세그멘테이션 평가 (mir_eval)
        # 실제 mir_eval을 적용하려면 각 샘플별 원본 경계와 예측된 경계, 레이블이 필요합니다.
        # 현재 DataLoader는 패딩된 시퀀스를 반환하므로, mir_eval을 적용하기 위해서는
        # DataLoader에서 원본 경계 정보도 함께 반환하도록 수정하거나,
        # 추론 단계에서 개별 파일에 대해 mir_eval을 적용하는 것이 더 적합합니다.
        # 여기서는 간단히 프레임별 정확도를 계산합니다.
        
        # 프레임별 정확도 계산
        correct_predictions = np.sum(np.array(all_val_predictions) == np.array(all_val_labels))
        total_frames = len(all_val_labels)
        frame_accuracy = correct_predictions / total_frames if total_frames > 0 else 0

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Frame Accuracy = {frame_accuracy:.4f}")

        # Early Stopping 및 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)
            print(f"최고 검증 손실 달성. 모델 저장: {config.CHECKPOINT_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping: 검증 손실이 {config.PATIENCE} 에폭 동안 개선되지 않았습니다.")
                break
    
    writer.close()
    print("모델 학습 완료.")

if __name__ == "__main__":
    train_model(config)

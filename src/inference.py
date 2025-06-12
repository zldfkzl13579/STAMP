# src/inference.py

import torch
import numpy as np
import os
from tqdm import tqdm
import mir_eval # mir_eval 임포트 추가

from config import config
from model import CNNGRUHybridModel
from data_loader import SongFormDataset # 추론 시에는 Dataset만 필요
from utils import get_label_mapping, load_annotations_for_id, evaluate_segmentation

def infer_and_evaluate(config):
    """
    학습된 모델을 사용하여 송폼을 추론하고 성능을 평가합니다.
    """
    # 레이블 매핑 가져오기
    label_to_id, id_to_label = get_label_mapping()
    
    # 테스트 데이터셋 로드
    test_dataset = SongFormDataset(
        feature_dir=os.path.join(config.FEATURE_SAVE_DIR, 'Test'),
        annotation_base_dir=config.ANNOTATIONS_PATH,
        label_to_id=label_to_id,
        id_to_label=id_to_label
    )

    # 특징 차원 결정 (첫 번째 샘플의 특징을 사용하여)
    if len(test_dataset) == 0:
        print("테스트 데이터셋이 비어 있습니다. 특징 추출이 제대로 되었는지 확인하세요.")
        return

    # 모델 초기화
    sample_features, _, _ = test_dataset[0] # 첫 번째 샘플 로드
    input_feature_dim = sample_features.shape[1] # (Sequence_Length, Features) -> Features 차원

    model = CNNGRUHybridModel(
        input_feature_dim=input_feature_dim,
        num_classes=config.NUM_CLASSES,
        cnn_out_channels=config.CNN_OUT_CHANNELS,
        gru_hidden_size=config.GRU_HIDDEN_SIZE,
        gru_num_layers=config.GRU_NUM_LAYERS,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)

    # 학습된 모델 가중치 로드
    if os.path.exists(config.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE))
        print(f"모델 가중치 로드 완료: {config.CHECKPOINT_PATH}")
    else:
        print(f"모델 가중치 파일이 없습니다: {config.CHECKPOINT_PATH}. 먼저 모델을 학습시켜야 합니다.")
        return

    model.eval() # 평가 모드

    all_scores = []

    print("\n--- 테스트 데이터셋 추론 및 평가 시작 ---")
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Inferring and Evaluating"):
            features_tensor, ssm_tensor, true_frame_labels_tensor = test_dataset[idx]
            
            # 배치 차원 추가 (모델 입력은 (Batch, Seq_Len, Features) 형태)
            features_tensor = features_tensor.unsqueeze(0).to(config.DEVICE)
            
            outputs = model(features_tensor) # (1, Sequence_Length_after_CNN, Num_Classes)
            predictions = torch.argmax(outputs, dim=-1).squeeze(0) # (Sequence_Length,)
            
            # 예측된 프레임 레이블을 numpy 배열로 변환
            estimated_frame_labels = predictions.cpu().numpy()

            # 원본 데이터 항목에서 경계 및 레이블 정보 가져오기
            item = test_dataset.data_items[idx]
            audio_id = item['audio_id']
            original_boundaries = item['boundaries']
            original_labels_ids = item['original_labels']
            
            # mir_eval을 위한 경계 및 레이블 생성
            # 예측된 프레임 레이블을 기반으로 경계와 레이블 시퀀스를 생성합니다.
            
            # Ensure there are predicted frames to process
            if len(estimated_frame_labels) == 0:
                print(f"경고: 오디오 ID {audio_id}에 대한 예측 프레임이 비어 있습니다. 평가 건너뛰기.")
                continue

            # 특징 프레임당 시간 계산
            # CNN의 풀링 레이어 때문에 특징 프레임당 시간이 늘어남 (두 번의 2배 풀링 -> 4배)
            effective_hop_length = config.HOP_LENGTH * 4
            frame_times = np.arange(len(estimated_frame_labels)) * effective_hop_length / config.SR
            
            # 예측된 레이블 시퀀스에서 경계점 찾기
            # 첫 번째 세그먼트의 시작 경계와 레이블
            estimated_boundaries_list = [0.0]
            estimated_labels_list = [estimated_frame_labels[0]]

            # 레이블이 변경되는 지점을 경계로 추가
            for i in range(1, len(estimated_frame_labels)):
                if estimated_frame_labels[i] != estimated_frame_labels[i-1]:
                    estimated_boundaries_list.append(frame_times[i])
                    estimated_labels_list.append(estimated_frame_labels[i])
            
            # 마지막 세그먼트의 끝 경계점 추가
            # 이 경계는 마지막 예측 프레임의 끝 시간으로 설정
            estimated_boundaries_list.append(frame_times[-1] + effective_hop_length / config.SR)
            
            estimated_boundaries = np.array(estimated_boundaries_list)
            estimated_labels_for_eval = np.array(estimated_labels_list) # ID 형태

            # mir_eval.segment.evaluate는 2차원 구간 배열을 기대합니다.
            # 1차원 경계 배열을 2차원 구간 배열로 변환합니다.
            # 원본 경계 배열에서 중복을 제거하여 mir_eval의 요구사항을 충족시킵니다.
            unique_original_boundaries = np.unique(original_boundaries)

            if len(unique_original_boundaries) > 1: # 최소한 두 개의 고유 경계가 있어야 구간이 생성됨
                ref_intervals = mir_eval.util.boundaries_to_intervals(unique_original_boundaries)
                
                # original_labels_ids는 SALAMI 데이터셋의 원본 어노테이션에서 파싱된 것으로,
                # 일반적으로 마지막 경계에 대한 레이블도 포함되어 있습니다 (N 경계, N 레이블).
                # mir_eval.util.boundaries_to_intervals로 생성된 ref_intervals의 개수(len(unique_original_boundaries) - 1)에
                # 맞추기 위해 original_labels_ids의 마지막 레이블을 제외합니다.
                if len(original_labels_ids) >= len(ref_intervals) + 1: # N labels for N boundaries -> N-1 intervals
                    ref_labels_for_eval = [id_to_label[int(l)] for l in original_labels_ids[:len(ref_intervals)]] 
                else:
                    print(f"경고: 오디오 ID {audio_id} - 원본 레이블 수가 참조 구간 수와 일치하지 않습니다. (원본 레이블: {len(original_labels_ids)}, 구간: {len(ref_intervals)})")
                    ref_intervals = np.array([])
                    ref_labels_for_eval = []
                    continue
            else:
                ref_intervals = np.array([])
                ref_labels_for_eval = []
                print(f"경고: 오디오 ID {audio_id}에 대한 원본 경계가 부족하거나 유효하지 않아 평가할 수 없습니다.")
                continue # 평가 건너뛰기

            # Check if estimated_boundaries has enough points to form intervals
            if len(estimated_boundaries) > 1: 
                est_intervals = mir_eval.util.boundaries_to_intervals(estimated_boundaries)
                # estimated_labels_for_eval은 이미 구간의 개수(len(estimated_boundaries) - 1)에 맞춰 생성되었으므로,
                # 추가적인 슬라이싱이 필요하지 않습니다.
                if len(estimated_labels_for_eval) == len(est_intervals):
                    est_labels_for_eval_str = [id_to_label[int(l)] for l in estimated_labels_for_eval]
                else:
                    print(f"경고: 오디오 ID {audio_id} - 예측 레이블 수가 예측 구간 수와 일치하지 않습니다. (예측 레이블: {len(estimated_labels_for_eval)}, 구간: {len(est_intervals)})")
                    est_intervals = np.array([])
                    est_labels_for_eval_str = []
                    continue
            else:
                est_intervals = np.array([])
                est_labels_for_eval_str = []
                print(f"경고: 오디오 ID {audio_id}에 대한 예측 경계가 부족하여 평가할 수 없습니다 (예측 프레임은 있었으나 단일 세그먼트).")
                continue # 평가 건너뛰기

            if len(ref_intervals) > 0 and len(est_intervals) > 0:
                scores = mir_eval.segment.evaluate(
                    ref_intervals,
                    ref_labels_for_eval,
                    est_intervals,
                    est_labels_for_eval_str
                )
                all_scores.append(scores)
            else:
                print(f"경고: 오디오 ID {audio_id}에 대한 유효한 원본 또는 예측 구간/레이블이 없습니다.")

    if all_scores:
        # 모든 평가 결과의 평균 계산
        avg_scores = {key: np.mean([s[key] for s in all_scores]) for key in all_scores[0]}
        print("\n--- 전체 테스트 데이터셋 평균 평가 결과 ---")
        for key, value in avg_scores.items():
            print(f"{key}: {value:.4f}")
    else:
        print("평가할 데이터가 없습니다.")

if __name__ == "__main__":
    infer_and_evaluate(config)

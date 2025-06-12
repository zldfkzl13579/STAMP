# src/tagger.py

import torch
import numpy as np
import os
from tqdm import tqdm
from scipy.signal import medfilt # 예측 결과 스무딩을 위해 추가

from config import config
from model import CNNGRUHybridModel
from feature_extractor import extract_features # 단일 오디오 특징 추출 함수
from utils import get_label_mapping # get_label_mapping 함수 임포트 추가

def load_model_for_inference(config, input_feature_dim):
    """
    학습된 모델을 로드하여 추론 준비를 합니다.
    """
    model = CNNGRUHybridModel(
        input_feature_dim=input_feature_dim,
        num_classes=config.NUM_CLASSES,
        cnn_out_channels=config.CNN_OUT_CHANNELS,
        gru_hidden_size=config.GRU_HIDDEN_SIZE,
        gru_num_layers=config.GRU_NUM_LAYERS,
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)

    if os.path.exists(config.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=config.DEVICE))
        print(f"모델 가중치 로드 완료: {config.CHECKPOINT_PATH}")
    else:
        raise FileNotFoundError(f"모델 가중치 파일이 없습니다: {config.CHECKPOINT_PATH}. 먼저 모델을 학습시켜야 합니다.")
    
    model.eval() # 평가 모드로 설정
    return model

def tag_audio_and_generate_annotation(audio_path, output_dir, config):
    """
    단일 오디오 파일에 대해 송폼 태깅을 수행하고 어노테이션 파일을 생성합니다.
    """
    print(f"\n--- 오디오 파일 태깅 시작: {os.path.basename(audio_path)} ---")

    # 1. 레이블 매핑 가져오기
    label_to_id, id_to_label = get_label_mapping()

    # 2. 오디오 특징 추출
    print("오디오 특징 추출 중...")
    # extract_features 함수는 딕셔너리를 반환하므로, 올바르게 처리해야 합니다.
    extracted_data = extract_features(
        audio_path, 
        config.SR, 
        config.N_FFT, 
        config.HOP_LENGTH, 
        config.N_MELS, 
        config.N_MFCC, 
        config.SSM_SIZE
    )
    
    if extracted_data is None: # 특징 추출 실패 시 None 반환
        print(f"특징 추출 실패: {audio_path}")
        return

    features = extracted_data['features'] # (프레임 수, 특징 차원)
    ssm = extracted_data['ssm'] # (SSM_SIZE, SSM_SIZE)
    
    if len(features) == 0: # 특징이 비어있는 경우
        print(f"특징이 비어 있습니다: {audio_path}")
        return

    input_feature_dim = features.shape[1]

    # 3. 모델 로드
    try:
        model = load_model_for_inference(config, input_feature_dim)
    except FileNotFoundError as e:
        print(e)
        return

    # 4. 추론
    print("모델 추론 중...")
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)
    
    with torch.no_grad():
        outputs = model(features_tensor) # (1, Sequence_Length_after_CNN, Num_Classes)
        predictions = torch.argmax(outputs, dim=-1).squeeze(0) # (Sequence_Length_after_CNN,)
    
    predicted_frame_labels = predictions.cpu().numpy()

    # 예측 결과 스무딩 (선택 사항) - 짧은 전환 오류 등을 줄이는 데 도움이 될 수 있습니다.
    # config에 스무딩 윈도우 크기를 추가할 수 있습니다 (예: config.SMOOTHING_WINDOW_SIZE)
    # 여기서는 임의로 5를 사용합니다. 홀수여야 합니다.
    # if len(predicted_frame_labels) > 5:
    #    predicted_frame_labels = medfilt(predicted_frame_labels, kernel_size=5).astype(int)

    # 5. 프레임 예측 -> 구간 어노테이션 변환
    print("프레임 예측을 어노테이션으로 변환 중...")
    # CNN의 풀링 레이어 때문에 특징 프레임당 시간이 늘어남 (두 번의 2배 풀링 -> 4배)
    # config.HOP_LENGTH는 원본 오디오 프레임당 시간이고, 모델 출력은 이보다 4배 긴 시간 간격으로 예측합니다.
    effective_hop_length = config.HOP_LENGTH * 4
    
    # 시간 스케일 계산
    frame_times = np.arange(len(predicted_frame_labels)) * effective_hop_length / config.SR

    # 어노테이션 리스트 초기화
    annotation_lines = []
    
    # 예측된 프레임 레이블이 없는 경우 처리
    if len(predicted_frame_labels) == 0:
        print(f"경고: 오디오 ID {os.path.basename(audio_path)}에 대한 예측된 프레임 레이블이 없습니다.")
        return

    # 첫 번째 세그먼트 시작
    current_start_time = 0.0
    current_label_id = predicted_frame_labels[0]

    for i in range(1, len(predicted_frame_labels)):
        if predicted_frame_labels[i] != current_label_id:
            end_time = frame_times[i]
            annotation_lines.append(f"{current_start_time:.6f}\t{id_to_label[int(current_label_id)]}")
            current_start_time = end_time
            current_label_id = predicted_frame_labels[i]
    
    # 마지막 세그먼트 추가 (오디오의 끝 시간까지)
    # 오디오의 실제 끝 시간을 사용하는 것이 좋습니다.
    # features.shape[0]은 원본 특징의 프레임 수입니다.
    final_audio_duration = (features.shape[0] * config.HOP_LENGTH) / config.SR # 대략적인 원본 오디오 길이
    
    # 모델의 마지막 예측 프레임의 끝 시간을 사용하거나, 원본 오디오 길이에 맞춤
    # 여기서 `final_audio_duration`은 원본 특징 기반이므로, 모델의 `predicted_frame_labels`의 마지막 프레임의 시간 이후로 잡아주는 것이 일반적입니다.
    last_predicted_frame_end_time = frame_times[-1] + effective_hop_length / config.SR if len(frame_times) > 0 else 0.0

    # 둘 중 더 긴 시간으로 최종 어노테이션 경계를 설정
    final_segment_end_time = max(final_audio_duration, last_predicted_frame_end_time)
    
    annotation_lines.append(f"{final_segment_end_time:.6f}\t{id_to_label[int(current_label_id)]}")

    # 6. 어노테이션 파일 저장
    os.makedirs(output_dir, exist_ok=True)
    audio_filename_base = os.path.splitext(os.path.basename(audio_path))[0]
    output_annotation_path = os.path.join(output_dir, f"{audio_filename_base}_predicted_annotation.txt")

    with open(output_annotation_path, 'w', encoding='utf-8') as f:
        for line in annotation_lines:
            f.write(line + '\n')

    print(f"어노테이션 파일 생성 완료: {output_annotation_path}")
    print("--- 태깅 완료 ---")

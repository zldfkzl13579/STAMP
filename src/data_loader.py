# src/data_loader.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import load_annotations_for_id, get_label_mapping, pad_sequence, pad_labels
from config import config
from tqdm import tqdm

class SongFormDataset(Dataset):
    """
    송폼 인식을 위한 PyTorch Dataset 클래스입니다.
    특징 데이터와 해당 어노테이션을 로드합니다.
    """
    def __init__(self, feature_dir, annotation_base_dir, label_to_id, id_to_label, max_sequence_length=None):
        self.feature_dir = feature_dir
        self.annotation_base_dir = annotation_base_dir
        self.label_to_id = label_to_id
        self.id_to_label = id_to_label
        self.max_sequence_length = max_sequence_length
        self.data_items = []

        # 특징 파일 목록 로드
        feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.npz')]
        
        print(f"데이터셋 로드 중: {feature_dir}")
        for feature_file in tqdm(feature_files, desc="Loading dataset"):
            audio_id = os.path.splitext(feature_file)[0].replace('_features', '') # _features 제거
            feature_path = os.path.join(feature_dir, feature_file)

            # 특징 로드
            try:
                loaded_data = np.load(feature_path)
                features = loaded_data['features'] # (프레임 수, 특징 차원)
                ssm = loaded_data['ssm'] # (SSM_SIZE, SSM_SIZE)
            except Exception as e:
                print(f"특징 파일 로드 중 오류 발생: {feature_path} - {e}")
                continue

            # 어노테이션 로드
            # SALAMI 데이터셋은 ID당 여러 어노테이션이 있을 수 있습니다.
            # 여기서는 첫 번째 어노테이션만 사용하거나, 모든 어노테이션을 학습에 활용할 수 있습니다.
            # 예시에서는 일단 첫 번째 유효한 어노테이션을 사용합니다.
            all_boundaries, all_labels_ids = load_annotations_for_id(audio_id, annotation_base_dir, label_to_id)
            
            if not all_boundaries:
                print(f"경고: 어노테이션을 찾을 수 없습니다: {audio_id}")
                continue
            
            # 여러 어노테이션 중 첫 번째 어노테이션을 기본으로 사용 (또는 모든 어노테이션을 처리하는 로직 추가)
            # 여기서는 모든 어노테이션을 개별 데이터 포인트로 추가합니다.
            for i in range(len(all_boundaries)):
                boundaries = all_boundaries[i]
                labels_ids = all_labels_ids[i]

                # 특징 프레임에 맞춰 레이블 시퀀스 생성
                # 각 특징 프레임이 어떤 송폼 레이블에 속하는지 매핑합니다.
                # 특징의 시간 길이를 기반으로 레이블을 확장합니다.
                
                # 특징의 총 시간 = (프레임 수 - 1) * hop_length / sr
                feature_duration = (features.shape[0] - 1) * config.HOP_LENGTH / config.SR
                
                # 마지막 어노테이션 경계가 특징의 총 시간보다 작으면 마지막 경계를 특징의 총 시간으로 설정
                if len(boundaries) > 0 and boundaries[-1] < feature_duration:
                    boundaries = np.append(boundaries, feature_duration)
                    # 마지막 레이블은 이전 레이블과 동일하게 설정하거나, 'no_function' 등으로 설정
                    labels_ids = np.append(labels_ids, labels_ids[-1] if len(labels_ids) > 0 else label_to_id['no_function'])

                # 각 특징 프레임에 해당하는 레이블을 결정
                frame_labels = np.full(features.shape[0], label_to_id['no_function'], dtype=int)
                for j in range(len(boundaries) - 1):
                    start_time = boundaries[j]
                    end_time = boundaries[j+1]
                    label_id = labels_ids[j]

                    # 해당 시간 구간에 속하는 프레임 인덱스 계산
                    start_frame = int(start_time * config.SR / config.HOP_LENGTH)
                    end_frame = int(end_time * config.SR / config.HOP_LENGTH)
                    
                    # 인덱스 범위 조정
                    start_frame = max(0, min(start_frame, features.shape[0] - 1))
                    end_frame = max(0, min(end_frame, features.shape[0])) # end_frame은 exclusive

                    frame_labels[start_frame:end_frame] = label_id
                
                self.data_items.append({
                    'audio_id': audio_id,
                    'features': features, # (프레임 수, 특징 차원)
                    'ssm': ssm,           # (SSM_SIZE, SSM_SIZE)
                    'labels': frame_labels, # (프레임 수,)
                    'boundaries': boundaries, # 원본 경계
                    'original_labels': labels_ids # 원본 레이블 ID
                })
        print(f"총 {len(self.data_items)}개의 데이터 항목 로드 완료.")

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        
        features = item['features']
        ssm = item['ssm']
        labels = item['labels']

        # PyTorch Tensor로 변환
        features_tensor = torch.tensor(features, dtype=torch.float32)
        ssm_tensor = torch.tensor(ssm, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return features_tensor, ssm_tensor, labels_tensor

def collate_fn(batch):
    """
    DataLoader를 위한 사용자 정의 collate_fn.
    배치 내의 시퀀스 길이를 동적으로 패딩하여 가장 긴 시퀀스에 맞춥니다.
    """
    features, ssm, labels = zip(*batch)

    # 가장 긴 시퀀스 길이 찾기
    # features는 이미 텐서이므로 .shape[0] 사용
    max_len = max(f.shape[0] for f in features)

    # 패딩 적용
    padded_features = []
    padded_labels = []
    for f, l in zip(features, labels):
        # f와 l은 이미 PyTorch 텐서이므로, pad_sequence와 pad_labels는 텐서를 반환할 것임
        padded_features.append(pad_sequence(f, max_len, padding_value=0.0))
        padded_labels.append(pad_labels(l, max_len, padding_value=-1)) # -1은 패딩 레이블 ID로 사용

    # 텐서로 변환 (이미 텐서이므로 stack만 하면 됨)
    features_batch = torch.stack(padded_features)
    ssm_batch = torch.stack(ssm) # SSM은 이미 고정 크기이므로 패딩 필요 없음
    labels_batch = torch.stack(padded_labels)

    return features_batch, ssm_batch, labels_batch

def get_data_loaders(config):
    """
    학습, 검증, 테스트 데이터 로더를 생성하여 반환합니다.
    """
    label_to_id, id_to_label = get_label_mapping()
    
    # 데이터셋 인스턴스 생성
    train_dataset = SongFormDataset(
        feature_dir=os.path.join(config.FEATURE_SAVE_DIR, 'Train'),
        annotation_base_dir=config.ANNOTATIONS_PATH,
        label_to_id=label_to_id,
        id_to_label=id_to_label
    )
    val_dataset = SongFormDataset(
        feature_dir=os.path.join(config.FEATURE_SAVE_DIR, 'Validation'),
        annotation_base_dir=config.ANNOTATIONS_PATH,
        label_to_id=label_to_id,
        id_to_label=id_to_label
    )
    test_dataset = SongFormDataset(
        feature_dir=os.path.join(config.FEATURE_SAVE_DIR, 'Test'),
        annotation_base_dir=config.ANNOTATIONS_PATH,
        label_to_id=label_to_id,
        id_to_label=id_to_label
    )

    # DataLoader 인스턴스 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0 # Windows에서는 num_workers를 0으로 설정하는 것이 안전합니다.
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    return train_loader, val_loader, test_loader, label_to_id, id_to_label


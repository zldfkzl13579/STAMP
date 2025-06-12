# src/utils.py

import os
import numpy as np
import mir_eval
from collections import OrderedDict
import torch # torch 임포트 추가

def get_label_mapping():
    """
    송폼 레이블을 정수 ID로 매핑하는 딕셔너리를 반환합니다.
    'no_function'을 포함하여 모든 가능한 레이블을 정의합니다.
    실제 SALAMI 데이터셋의 모든 고유 레이블을 확인하여 업데이트해야 합니다.
    """
    labels = [
        'Silence', 'Intro', 'Verse', 'Pre-Chorus', 'Chorus', 'Bridge',
        'Outro', 'Fade-out', 'Transition', 'Solo', 'A', 'B', 'C', 'no_function'
    ]
    # OrderedDict를 사용하여 순서 보장
    label_to_id = OrderedDict({label: i for i, label in enumerate(labels)})
    id_to_label = OrderedDict({i: label for i, label in enumerate(labels)})
    return label_to_id, id_to_label

def parse_annotation_file(annotation_path, label_to_id):
    """
    SALAMI 어노테이션 파일을 파싱하여 (시간, 레이블 ID) 쌍의 리스트를 반환합니다.
    하나 또는 두 개의 어노테이션 파일을 처리합니다.
    """
    boundaries = []
    labels = []

    # 단일 어노테이션 파일 처리
    if os.path.exists(annotation_path):
        with open(annotation_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    try:
                        time = float(parts[0])
                        label = parts[1]
                        boundaries.append(time)
                        labels.append(label_to_id.get(label, label_to_id['no_function'])) # 없는 레이블은 no_function으로 처리
                    except ValueError:
                        print(f"경고: 유효하지 않은 라인 형식 - {line.strip()} (파일: {annotation_path})")
                        continue
    else:
        print(f"경고: 어노테이션 파일이 존재하지 않습니다: {annotation_path}")
        return np.array([]), np.array([])

    return np.array(boundaries), np.array(labels)

def load_annotations_for_id(audio_id, annotations_base_path, label_to_id):
    """
    주어진 오디오 ID에 대해 모든 어노테이션 파일을 로드합니다.
    SALAMI 데이터셋은 ID당 여러 어노테이션이 있을 수 있습니다 (textfile1, textfile2).
    모든 어노테이션을 로드하여 반환합니다.
    """
    audio_id_str = str(audio_id)
    parsed_dir = os.path.join(annotations_base_path, audio_id_str, 'parsed')
    
    all_boundaries = []
    all_labels = []

    if not os.path.exists(parsed_dir):
        print(f"경고: 어노테이션 디렉토리가 존재하지 않습니다: {parsed_dir}")
        return [], []

    # 'textfile1_functions.txt'와 'textfile2_functions.txt'를 모두 확인
    for i in range(1, 3): # textfile1, textfile2
        annotation_file_name = f'textfile{i}_functions.txt'
        annotation_path = os.path.join(parsed_dir, annotation_file_name)
        
        if os.path.exists(annotation_path):
            boundaries, labels = parse_annotation_file(annotation_path, label_to_id)
            if len(boundaries) > 0:
                all_boundaries.append(boundaries)
                all_labels.append(labels)
        # else:
        #     print(f"정보: 어노테이션 파일이 존재하지 않습니다: {annotation_path}")

    return all_boundaries, all_labels

def evaluate_segmentation(reference_boundaries, reference_labels, estimated_boundaries, estimated_labels, id_to_label):
    """
    mir_eval을 사용하여 세그멘테이션 성능을 평가합니다.
    reference_labels와 estimated_labels는 ID가 아닌 실제 레이블 문자열이어야 합니다.
    """
    # ID를 실제 레이블 문자열로 변환
    ref_labels_str = [id_to_label[int(l)] for l in reference_labels]
    est_labels_str = [id_to_label[int(l)] for l in estimated_labels]

    # mir_eval은 마지막 경계점을 포함하지 않으므로, 마지막 레이블을 제거합니다.
    # 하지만 mir_eval.segment.evaluate는 자동으로 처리하므로, 여기서는 그대로 전달합니다.

    # mir_eval.segment.evaluate는 reference와 estimated의 길이가 같아야 한다고 가정합니다.
    # 보통 estimated_boundaries의 마지막 경계는 reference의 마지막 경계와 일치해야 합니다.
    # 만약 estimated_boundaries가 reference_boundaries보다 짧다면, 마지막 경계점을 추가해줍니다.
    if len(estimated_boundaries) > 0 and len(reference_boundaries) > 0:
        if estimated_boundaries[-1] < reference_boundaries[-1]:
            estimated_boundaries = np.append(estimated_boundaries, reference_boundaries[-1])
            # 마지막 레이블은 이전 레이블과 동일하게 설정하거나, 'no_function' 등으로 설정
            estimated_labels = np.append(estimated_labels, estimated_labels[-1] if len(estimated_labels) > 0 else label_to_id['no_function'])
            est_labels_str = [id_to_label[int(l)] for l in estimated_labels] # 다시 변환

    scores = mir_eval.segment.evaluate(
        reference_boundaries,
        ref_labels_str,
        estimated_boundaries,
        est_labels_str
    )
    return scores

def pad_sequence(sequence, max_len, padding_value=0):
    """
    시퀀스를 지정된 최대 길이로 패딩합니다.
    NumPy 배열 또는 PyTorch 텐서를 처리할 수 있습니다.
    """
    is_torch_tensor = isinstance(sequence, torch.Tensor)

    current_len = sequence.shape[0] if is_torch_tensor else len(sequence)

    if current_len >= max_len:
        return sequence[:max_len]
    
    if is_torch_tensor:
        # PyTorch 텐서 패딩
        padding_shape = (max_len - current_len, sequence.shape[1])
        # padding_value의 dtype을 sequence.dtype과 일치시키고, device도 일치시킵니다.
        padding = torch.full(padding_shape, padding_value, dtype=sequence.dtype, device=sequence.device)
        return torch.cat((sequence, padding), dim=0)
    else:
        # NumPy 배열 패딩
        padding = np.full((max_len - current_len, sequence.shape[1]), padding_value, dtype=sequence.dtype)
        return np.vstack((sequence, padding))

def pad_labels(labels, max_len, padding_value=-1):
    """
    레이블 시퀀스를 지정된 최대 길이로 패딩합니다.
    NumPy 배열 또는 PyTorch 텐서를 처리할 수 있습니다.
    """
    is_torch_tensor = isinstance(labels, torch.Tensor)

    current_len = labels.shape[0] if is_torch_tensor else len(labels)

    if current_len >= max_len:
        return labels[:max_len]
    
    if is_torch_tensor:
        # PyTorch 텐서 패딩
        padding_shape = (max_len - current_len,)
        # padding_value의 dtype을 labels.dtype과 일치시키고, device도 일치시킵니다.
        padding = torch.full(padding_shape, padding_value, dtype=labels.dtype, device=labels.device)
        return torch.cat((labels, padding), dim=0)
    else:
        # NumPy 배열 패딩
        padding = np.full(max_len - current_len, padding_value, dtype=labels.dtype)
        return np.concatenate((labels, padding))


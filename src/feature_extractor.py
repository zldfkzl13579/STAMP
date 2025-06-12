# src/feature_extractor.py

import os
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
from config import config
from scipy.spatial.distance import pdist, squareform

def extract_features(audio_path, sr, n_fft, hop_length, n_mels, n_mfcc, ssm_size):
    """
    주어진 오디오 파일에서 MFCCs, Chromagrams, RMS Energy, Novelty Function,
    Rhythmic Novelty Function, Self-Similarity Matrix (SSM)를 추출합니다.
    """
    try:
        # 오디오 로드
        y, sr = librosa.load(audio_path, sr=sr, mono=True) # 스테레오 -> 모노 변환하여 특징 추출

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        # Chromagrams
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

        # RMS Energy
        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)

        # Novelty Function (Onset Detection Function)
        onset_env = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, units='frames')
        # 온셋 엔벨로프는 스칼라 값이므로, 특징 차원에 맞게 변환
        onset_features = np.zeros(mfccs.shape[1])
        onset_features[onset_env] = 1 # 온셋이 감지된 프레임에 1 표시

        # Rhythmic Novelty Function (Tempogram)
        # 템포그램은 리듬 정보를 포함하며, 프레임 수가 다를 수 있으므로 MFCCs 프레임 수에 맞게 조정
        onset_env_full = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, units='samples')
        # 템포그램은 일반적으로 2D 배열 (빈도 x 시간)
        tempogram = librosa.feature.tempogram(y=y, sr=sr, onset_envelope=librosa.onset.onset_strength(y=y, sr=sr),
                                              hop_length=hop_length)
        
        # Self-Similarity Matrix (SSM)
        # 코사인 유사도 기반의 SSM
        # 특징 벡터를 결합하여 SSM을 생성
        combined_features = np.vstack([mfccs, chroma, rms])
        
        # SSM 계산 (유클리드 거리 기반)
        # 프레임 수가 너무 많으면 메모리 문제가 발생할 수 있으므로, 특징을 다운샘플링하여 계산
        # 여기서는 특징 벡터 자체의 유사도를 계산하여 SSM을 만듭니다.
        # 각 프레임 벡터 간의 코사인 유사도를 계산하여 SSM을 만듭니다.
        # librosa.segment.recurrence_matrix를 사용할 수도 있습니다.
        
        # 특징 벡터 간의 코사인 유사도 계산
        # 프레임 수가 너무 많으면 계산량이 많아지므로, 여기서는 MFCCs를 예시로 사용
        # 실제 구현에서는 모든 특징을 합친 벡터를 사용할 수 있습니다.
        
        # 정규화된 특징 벡터
        norm_features = combined_features / np.linalg.norm(combined_features, axis=0)
        ssm = np.dot(norm_features.T, norm_features) # 코사인 유사도 기반 SSM

        # SSM 다운샘플링 (1024x1024)
        # SSM이 1024x1024보다 크면 다운샘플링
        if ssm.shape[0] > ssm_size:
            # 간단한 선형 보간 또는 평균 풀링으로 다운샘플링
            # 여기서는 간단하게 리사이징을 위해 scipy.ndimage.zoom을 사용할 수 있지만,
            # numpy만 사용한다면, 블록 평균 풀링을 구현할 수 있습니다.
            block_size_row = ssm.shape[0] // ssm_size
            block_size_col = ssm.shape[1] // ssm_size
            
            # 블록 평균 풀링 구현
            downsampled_ssm = np.zeros((ssm_size, ssm_size))
            for i in range(ssm_size):
                for j in range(ssm_size):
                    row_start = i * block_size_row
                    row_end = (i + 1) * block_size_row
                    col_start = j * block_size_col
                    col_end = (j + 1) * block_size_col
                    downsampled_ssm[i, j] = np.mean(ssm[row_start:row_end, col_start:col_end])
            ssm = downsampled_ssm
        
        # 모든 특징을 하나의 배열로 결합
        # 특징들의 시퀀스 길이를 맞춰야 합니다.
        # mfccs.shape[1]이 프레임 수
        
        # RMS는 (1, N_frames) -> (N_frames,)
        rms = rms.squeeze()

        # onset_features는 (N_frames,)
        
        # tempogram은 (N_freq, N_frames)
        # tempogram의 시간 축 길이를 다른 특징들과 맞춥니다.
        # tempogram을 평균하여 1차원 벡터로 만들거나, 특정 밴드를 선택할 수 있습니다.
        # 여기서는 평균하여 1차원 벡터로 만듭니다.
        tempogram_mean = np.mean(tempogram, axis=0)

        # 모든 특징을 (특징 차원, 시간 프레임) 형태로 맞춥니다.
        # 각 특징의 프레임 수가 다를 수 있으므로, 가장 짧은 프레임 수에 맞춥니다.
        min_frames = min(mfccs.shape[1], chroma.shape[1], rms.shape[0], onset_features.shape[0], tempogram_mean.shape[0])

        # 특징들을 결합 (각 특징의 프레임 수를 최소 프레임 수로 자르기)
        features_combined = np.vstack([
            mfccs[:, :min_frames],
            chroma[:, :min_frames],
            rms[:min_frames].reshape(1, -1), # 1차원 배열을 2차원으로 변환
            onset_features[:min_frames].reshape(1, -1),
            tempogram_mean[:min_frames].reshape(1, -1)
        ])

        # SSM은 별도로 저장하거나, 특징 벡터에 추가할 수 있습니다.
        # 여기서는 특징 벡터와 SSM을 딕셔너리 형태로 반환합니다.
        
        return {
            'features': features_combined.T, # (시간 프레임, 특징 차원) 형태로 반환
            'ssm': ssm
        }

    except Exception as e:
        print(f"오디오 파일 {audio_path} 특징 추출 중 오류 발생: {e}")
        return None

def process_and_save_features(config):
    """
    SALAMI 데이터셋의 오디오 파일에서 특징을 추출하고 .npy 파일로 저장합니다.
    """
    datasets = ['Train', 'Validation', 'Test']
    
    for dataset_type in datasets:
        audio_dir = os.path.join(config.AUDIO_PATH, dataset_type)
        feature_save_dir = os.path.join(config.FEATURE_SAVE_DIR, dataset_type)
        
        print(f"\n--- {dataset_type} 데이터셋 특징 추출 시작 ---")
        
        audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.flac')]
        
        for audio_file in tqdm(audio_files, desc=f"Extracting {dataset_type} features"):
            audio_id = os.path.splitext(audio_file)[0]
            audio_path = os.path.join(audio_dir, audio_file)
            
            # 이미 특징 파일이 존재하면 건너뛰기
            feature_output_path = os.path.join(feature_save_dir, f"{audio_id}_features.npz")
            if os.path.exists(feature_output_path):
                # print(f"특징 파일이 이미 존재합니다. 건너뛰기: {feature_output_path}")
                continue

            extracted_data = extract_features(
                audio_path,
                config.SR,
                config.N_FFT,
                config.HOP_LENGTH,
                config.N_MELS,
                config.N_MFCC,
                config.SSM_SIZE
            )

            if extracted_data:
                # 특징과 SSM을 별도의 키로 저장
                np.savez_compressed(feature_output_path, features=extracted_data['features'], ssm=extracted_data['ssm'])
            else:
                print(f"특징 추출 실패: {audio_file}")

    print("\n--- 모든 특징 추출 완료 ---")

if __name__ == "__main__":
    process_and_save_features(config)

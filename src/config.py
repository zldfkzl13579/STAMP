# src/config.py

import os
import torch

class Config:
    """
    프로젝트의 모든 설정 및 하이퍼파라미터를 정의합니다.
    """
    def __init__(self):
        # 데이터셋 경로 설정
        # config.py는 STAMP/src 폴더에 있으므로, MYPROJECT 폴더를 BASE_DIR로 설정하려면
        # 두 단계 상위 디렉토리로 이동해야 합니다.
        self.BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..') # 현재 파일 기준 두 단계 상위 폴더 (MYPROJECT)
        self.SALAMI_DATASETS_PATH = os.path.join(self.BASE_DIR, 'SALAMI_datasets')
        self.FEATURES_PATH = os.path.join(self.SALAMI_DATASETS_PATH, 'features')
        self.AUDIO_PATH = os.path.join(self.SALAMI_DATASETS_PATH, 'SALAMI_audio')
        self.ANNOTATIONS_PATH = os.path.join(self.SALAMI_DATASETS_PATH, 'salami-data-public-master', 'annotations')

        # 특징 추출 설정
        self.SR = 48000  # 샘플링 레이트 (Hz)
        self.N_FFT = 2048  # FFT 윈도우 크기
        self.HOP_LENGTH = 512  # 홉 길이
        self.N_MELS = 128  # 멜 스펙트로그램 필터 뱅크 개수
        self.N_MFCC = 20  # MFCC 개수
        self.SSM_SIZE = 1024 # SSM 다운샘플링 크기

        # 모델 설정
        # NUM_CLASSES를 utils.py의 get_label_mapping에 정의된 실제 레이블 개수와 일치시킵니다.
        # 현재 get_label_mapping에는 14개의 고유 레이블이 정의되어 있습니다.
        self.NUM_CLASSES = 14  
        self.CNN_OUT_CHANNELS = 64 # CNN 출력 채널 수
        self.GRU_HIDDEN_SIZE = 128 # GRU 히든 스테이트 크기
        self.GRU_NUM_LAYERS = 2    # GRU 레이어 개수
        self.DROPOUT_RATE = 0.3    # 드롭아웃 비율

        # 학습 설정
        self.BATCH_SIZE = 16
        self.NUM_EPOCHS = 50
        self.LEARNING_RATE = 0.001
        self.WEIGHT_DECAY = 1e-5 # L2 정규화
        self.PATIENCE = 10       # Early Stopping을 위한 patience (검증 손실이 개선되지 않을 때까지 기다리는 에폭 수)

        # 저장 경로
        self.MODEL_SAVE_DIR = os.path.join(self.BASE_DIR, 'models')
        self.CHECKPOINT_PATH = os.path.join(self.MODEL_SAVE_DIR, 'best_model.pth')
        self.FEATURE_SAVE_DIR = os.path.join(self.SALAMI_DATASETS_PATH, 'features_processed') # 추출된 특징 저장 경로

        # 장치 설정 (GPU 사용 가능 여부 확인)
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"사용 가능한 장치: {self.DEVICE}")

        # 폴더 생성
        os.makedirs(self.MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(self.FEATURE_SAVE_DIR, exist_ok=True)
        os.makedirs(os.path.join(self.FEATURE_SAVE_DIR, 'Train'), exist_ok=True)
        os.makedirs(os.path.join(self.FEATURE_SAVE_DIR, 'Validation'), exist_ok=True)
        os.makedirs(os.path.join(self.FEATURE_SAVE_DIR, 'Test'), exist_ok=True)

# 설정 객체 인스턴스 생성
config = Config()

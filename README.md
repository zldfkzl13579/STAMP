송폼 자동 인식 및 태깅 AI 파이프라인
이 프로젝트는 딥러닝(CNN-GRU 하이브리드 모델)을 사용하여 음악 오디오에서 송폼(Song Form)을 자동으로 인식하고 태깅하는 AI 파이프라인입니다. 특징 추출부터 모델 학습, 추론, 평가 및 단일 오디오 파일 태깅까지 전 과정을 지원합니다.

1. 프로젝트 디렉토리 구조
프로젝트의 전체 디렉토리 구조는 다음과 같습니다.
```
MYPROJECT/
├── main.py                     # 메인 실행 스크립트
├── README.md                   # 프로젝트 설명 파일
├── src/
│   ├── config.py               # 전역 설정 및 하이퍼파라미터 정의
│   ├── data_loader.py          # 데이터셋 로딩 및 전처리 (PyTorch Dataset, DataLoader)
│   ├── evaluate.py             # 모델 평가 스크립트 (mir_eval 사용)
│   ├── feature_extractor.py    # 오디오 특징 추출 스크립트 (MFCC, Chroma, RMS, Novelty, SSM 등)
│   ├── inference.py            # 학습된 모델을 사용한 추론 및 평가
│   ├── model.py                # CNN-GRU 하이브리드 모델 정의
│   ├── tagger.py               # 단일 오디오 파일 태깅 및 어노테이션 파일 생성
│   ├── train.py                # 모델 학습 스크립트
│   └── utils.py                # 유틸리티 함수 (레이블 매핑, 어노테이션 파싱, 패딩 등)
├── SALAMI_datasets/            # SALAMI 데이터셋 저장 디렉토리 (사용자가 직접 구성)
│   ├── SALAMI_audio/           # 원본 오디오 파일 (FLAC 형식)
│   │   ├── Train/
│   │   ├── Validation/
│   │   └── Test/
│   ├── features/               # librosa를 통해 추출된 원본 특징 (현재 사용되지 않음, `features_processed` 사용)
│   ├── features_processed/     # 처리된 특징 파일 (Numpy .npz 형식)
│   │   ├── Train/              # 학습 데이터셋의 특징
│   │   ├── Validation/         # 검증 데이터셋의 특징
│   │   └── Test/               # 테스트 데이터셋의 특징
│   └── salami-data-public-master/ # SALAMI 원본 어노테이션 파일
│       └── annotations/
│           └── <audio_id>/
│               └── parsed/
│                   └── <audio_id>.txt
├── models/                     # 학습된 모델 가중치 저장 디렉토리
│   └── best_model.pth          # 최적의 모델 가중치 파일
└── tagged_annotations/         # 단일 오디오 태깅 결과 저장 디렉토리 (tag_audio 모드 시 자동 생성)
    └── <audio_filename>_annotation.txt
```
    
참고: SALAMI_datasets 폴더는 사용자가 SALAMI 데이터셋을 다운로드하여 위의 구조에 맞게 배치해야 합니다. 특히 SALAMI_audio 내부에는 Train, Validation, Test 폴더를 생성하고 각 오디오 파일을 적절히 배치해야 합니다. salami-data-public-master 폴더는 SALAMI 데이터셋의 어노테이션 파일이 위치하는 곳입니다.

2. 설치 및 필수 라이브러리
이 프로젝트를 실행하기 위해 다음 라이브러리들이 필요합니다. pip를 사용하여 설치할 수 있습니다.

pip install torch torchvision torchaudio numpy librosa scikit-learn mir_eval tqdm tensorboard scipy
```
주요 라이브러리:

torch: 딥러닝 모델 구축 및 학습을 위한 PyTorch 프레임워크

numpy: 수치 계산을 위한 기본 라이브러리

librosa: 오디오 분석 및 특징 추출 라이브러리

scikit-learn: 머신러닝 유틸리티 (여기서는 직접적인 사용은 없으나, 데이터 전처리 등에 사용될 수 있음)

mir_eval: 음악 정보 검색(MIR) 평가 지표 계산 라이브러리 (송폼 평가에 필수)

tqdm: 루프 진행 상황을 시각화하는 데 사용

tensorboard: 학습 과정 시각화 도구

scipy: 과학 계산 라이브러리 (medfilt 등)
```
3. 사용법
main.py 스크립트를 통해 파이프라인의 다양한 단계를 실행할 수 있습니다. --mode 인자를 사용하여 원하는 동작을 지정합니다.

3.1. 모든 단계 실행 (특징 추출, 학습, 평가)
모든 파이프라인 단계를 순차적으로 실행합니다.
```
python main.py --mode all
```
3.2. 특징 추출만 실행
오디오 파일에서 특징을 추출하여 SALAMI_datasets/features_processed 디렉토리에 저장합니다.
```
python main.py --mode features
```
3.3. 모델 학습만 실행
추출된 특징을 사용하여 모델을 학습하고 models/best_model.pth에 최적 모델 가중치를 저장합니다. TensorBoard 로그는 runs/ 디렉토리에 저장됩니다.
```
python main.py --mode train
```
3.4. 모델 추론 및 평가만 실행
학습된 모델을 사용하여 테스트 데이터셋에 대해 추론을 수행하고 성능을 평가합니다.
```
python main.py --mode evaluate
# 또는
python main.py --mode infer
```
3.5. 단일 오디오 파일 태깅
특정 오디오 파일에 대해 송폼을 예측하고, 예측된 송폼 어노테이션을 텍스트 파일로 저장합니다.
--audio_path 인자로 태깅할 오디오 파일의 경로를 지정해야 합니다. --output_dir을 통해 저장될 디렉토리를 지정할 수 있습니다 (기본값: 현재 작업 디렉토리).
```
python main.py --mode tag_audio --audio_path "path/to/your/audio_file.flac" --output_dir "path/to/save/annotation"
```
예시:
```
python main.py --mode tag_audio --audio_path "SALAMI_datasets/SALAMI_audio/Test/0000.flac" --output_dir "./my_tagged_results"
```
참고: tag_audio 모드를 사용하기 전에 모델이 학습되어 models/best_model.pth 파일이 존재해야 합니다.

4. 설정 (config.py)
src/config.py 파일에서 데이터셋 경로, 특징 추출 파라미터, 모델 하이퍼파라미터, 학습 설정 등을 세부적으로 조정할 수 있습니다. 프로젝트의 모든 전역 설정은 이 파일에서 관리됩니다.

# src/config.py
```
class Config:
    def __init__(self):
        # ... (경로 설정) ...
        self.SR = 48000           # 샘플링 레이트
        self.N_FFT = 2048         # FFT 윈도우 크기
        self.HOP_LENGTH = 512     # 홉 길이
        self.NUM_CLASSES = 14     # 송폼 클래스 개수
        self.BATCH_SIZE = 16      # 배치 크기
        self.NUM_EPOCHS = 50      # 학습 에폭 수
        self.LEARNING_RATE = 0.001
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 학습 장치 설정
        # ... (그 외 설정) ...
```
GPU 사용: config.py의 DEVICE 설정에 따라 GPU(CUDA)를 사용할 수 있습니다. torch.cuda.is_available()이 True를 반환하면 자동으로 GPU를 사용합니다.

5. 학습 진행 상황 모니터링 (TensorBoard)
모델 학습 중 TensorBoard를 사용하여 학습 손실, 검증 손실, 프레임 정확도 등을 실시간으로 모니터링할 수 있습니다. 학습 스크립트 실행 후 다음 명령어를 터미널에 입력하세요.
```
tensorboard --logdir=runs
```
그리고 웹 브라우저에서 표시되는 URL(일반적으로 http://localhost:6006/)에 접속합니다.

6. 개발자 정보
저자: 정현섭

라이센스: MIT License

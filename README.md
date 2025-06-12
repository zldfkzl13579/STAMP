# STAMP
STAMP(Structure Tagging and Analysis Module for Pieces) (음악)작품 구조 태깅 및 분석 모듈

사용법
전체 실행:
& C:/Users/bossc/miniconda3/envs/song_form_env/python.exe d:/ProjectFiles/MyProject/STAMP/main.py 또는 --mode all
특징 추출:
& C:/Users/bossc/miniconda3/envs/song_form_env/python.exe d:/ProjectFiles/MyProject/STAMP/main.py --mode features
학습 모드:
& C:/Users/bossc/miniconda3/envs/song_form_env/python.exe d:/ProjectFiles/MyProject/STAMP/main.py --mode train
추론 및 평가 모드:
& C:/Users/bossc/miniconda3/envs/song_form_env/python.exe d:/ProjectFiles/MyProject/STAMP/main.py --mode evaluate 또는 --mode infer
태깅 모드:
& C:/Users/bossc/miniconda3/envs/song_form_env/python.exe d:/ProjectFiles/MyProject/STAMP/main.py --mode tag_audio --audio_path "E:\Music\헬크\HELP.flac" --output_dir "C:\Users\bossc\Downloads"

MYPROJECT/
├── SALAMI_datasets/
│   ├── SALAMI_audio/
│   │   ├── Train/
│   │   │   ├── 1.flac
│   │   │   ├── ...
│   │   └── Validation/
│   │   │   ├── 34.flac
│   │   │   ├── ...
│   │   └── Test/
│   │   │   ├── 432.flac
│   │   │   ├── ...
│   └── salami-data-public-master/
│       └── annotations/
│           ├── 1/
│           │   └── parsed/
│           │       └── 1.txt
│           ├── 34/
│           │   └── parsed/
│           │       └── 34.txt
│           └── 432/
│               └── parsed/
│                   └── 432.txt
│
└── STAMP
    ├── src/
    │   ├── config.py
    │   ├── data_loader.py       
    │   ├── feature_extractor.py 
    │   ├── model.py             
    │   ├── train.py             
    │   ├── inference.py         
    │   └── utils.py             
    ├── main.py                  
    └── README.md

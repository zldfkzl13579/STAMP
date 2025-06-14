# STAMP
STAMP(Structure Tagging and Analysis Module for Pieces) (음악)작품 구조 태깅 및 분석 모듈

필요 라이브러리
PyTorch, Librosa, NumPy, SciPy, mir_eval, Tqdm, TensorBoard, skimage

사용법  
전체 실행:  
python main.py --mode all  
특징 추출:  
python main.py --mode features  
학습 모드:  
python main.py --mode train  
추론 및 평가 모드:  
python main.py --mode evaluate  
# 또는  
python main.py --mode infer  
태깅 모드:  
python main.py --mode tag_audio --audio_path "E:\Music\abcd.flac" --output_dir "C:\Users\bossc\Downloads"

환경설정방법: config.json 직접수정
ssm_downsample: SSM 다운샘플링 여부 (true 또는 false)
ssm_downsample_factor: SSM 다운샘플링 비율 (예: 2로 설정하면 크기가 절반으로 줄어듬)
paths.audio_base_dir: 음원 파일의 상대 경로 (예: "datasets/audio")
paths.annotations_base_dir: 어노테이션 파일의 상대 경로 (예: "datasets/annotations")
paths.features_save_dir: 특징 파일 저장 경로의 상대 경로 (예: "datasets/features_processed")
paths.model_save_dir: 모델 저장 경로의 상대 경로 (예: "models")

아노테이션 파일 형태 및 사용되는 송폼태그(Intro, Verse, Pre-chorus, Chorus, Post-chorus, Bridge, Outro, Interlude, Solo, No_function(Silence 포함), END)
0.000000000	No_function
0.394739229	Intro
47.292630385	Verse
109.646802721	Chorus
117.382585034	Verse
193.878435374	Chorus
201.417551020	Verse
261.942879818	Outro
276.339554563	End

구성
<pre>
STAMP/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── feature_extractor.py
│   ├── inference.py
│   ├── model.py
│   ├── tagger.py
│   ├── train.py
│   └── utils.py
├── main.py
├── README.md
├── config.json          
├── requirements.txt  
├── models/      
│   └── best_model.pth  
├── datasets/ 
│   ├── audio/
│   │   ├── Train/
│   │   │   ├── 1.flac
│   │   │   ├── ...
│   │   └── Validation/
│   │   │   ├── 34.flac
│   │   │   ├── ...
│   │   └── Test/
│   │   │   ├── 432.flac
│   │   │   ├── ...
│   ├── annotations/
│   │   ├── 1/
│   │   │   ├── 1_1.txt
│   │   │   └── 1_2.txt
│   │   ├── 34/
│   │   │   ├── 34_1.txt
│   │   │   └── 34_2.txt
│   │   └── 432/
│   │       ├── 432_1.txt
│   │       └── 432_2.txt
│   └── features_processed/ 
│       ├── Train/
│       ├── Validation/
│       └── Test/
└── tagged_outputs/
    └── your_audio_form_annotation.txt
</pre>

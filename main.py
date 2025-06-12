# main.py

import sys
import os

# src 폴더를 Python 경로에 추가하여 모듈을 임포트할 수 있도록 합니다.
# 이 라인은 다른 모듈 임포트보다 먼저 실행되어야 합니다.
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import argparse # argparse 모듈 임포트
from config import config
from feature_extractor import process_and_save_features
from train import train_model
from inference import infer_and_evaluate
from tagger import tag_audio_and_generate_annotation # tagger 모듈 임포트

def main():
    print("--- 송폼 자동 인식 및 태깅 AI 개발 파이프라인 시작 ---")
    print(f"사용 가능한 장치: {config.DEVICE}")

    parser = argparse.ArgumentParser(description="송폼 자동 인식 및 태깅 AI 파이프라인")
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'features', 'train', 'evaluate', 'infer', 'tag_audio'],
                        help="실행할 파이프라인 모드: 'all', 'features', 'train', 'evaluate', 'infer', 'tag_audio'")
    parser.add_argument('--audio_path', type=str,
                        help="단일 오디오 태깅 모드에서 사용할 오디오 파일 경로")
    parser.add_argument('--output_dir', type=str, default='.',
                        help="태깅된 어노테이션 파일을 저장할 디렉토리 (기본값: 현재 디렉토리)")
    args = parser.parse_args()

    # 1. 특징 추출 및 저장
    if args.mode in ['all', 'features']:
        print("\n[단계 1/3] 특징 추출 및 저장...")
        process_and_save_features(config)
        print("[단계 1/3] 특징 추출 및 저장 완료.")
        if args.mode == 'features':
            return

    # 2. 모델 학습
    if args.mode in ['all', 'train']:
        print("\n[단계 2/3] 모델 학습 시작...")
        train_model(config)
        print("[단계 2/3] 모델 학습 완료.")
        if args.mode == 'train':
            return

    # 3. 모델 추론 및 평가 (데이터셋 전체)
    if args.mode in ['all', 'evaluate', 'infer']:
        print("\n[단계 3/3] 모델 추론 및 평가 시작...")
        infer_and_evaluate(config)
        print("[단계 3/3] 모델 추론 및 평가 완료.")
        if args.mode in ['evaluate', 'infer']:
            return

    # 4. 단일 오디오 태깅 (새로운 모드)
    if args.mode == 'tag_audio':
        if not args.audio_path:
            print("오류: 'tag_audio' 모드에는 '--audio_path' 인자가 필수입니다.")
            return
        
        # 기본 출력 디렉토리가 현재 작업 디렉토리이므로, 원하는 다른 경로로 변경 가능
        # config.BASE_DIR은 MYPROJECT 폴더를 가리키므로, 그 안에 'tagged_annotations' 폴더를 생성
        output_annotation_dir = args.output_dir if args.output_dir != '.' else os.path.join(config.BASE_DIR, 'tagged_annotations')
        os.makedirs(output_annotation_dir, exist_ok=True) # 출력 디렉토리 생성

        tag_audio_and_generate_annotation(args.audio_path, output_annotation_dir, config)
        print("\n--- 단일 오디오 태깅 완료 ---")
        return

    print("\n--- 송폼 자동 인식 및 태깅 AI 개발 파이프라인 종료 ---")

if __name__ == "__main__":
    main()

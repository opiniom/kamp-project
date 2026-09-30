#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/03_train.py
[단계 D] YOLOv8 모델 파인튜닝 학습 (평가항목: 모델 개발 40점)
- Ultralytics YOLOv8s / YOLOv8n 학습 스크립트
- KAMP Note 환경(CUDA GPU / CPU) 자동 감지
- 데이터 증강(Mosaic, Flip, Scale 등) 적용
- 산출물: runs/train/{exp_name}/weights/best.pt, results.csv
"""

import os
import sys
import argparse
import yaml
import torch
from ultralytics import YOLO

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def parse_args():
    parser = argparse.ArgumentParser(description="KAMP X-ray YOLOv8 Training Script")
    parser.add_argument('--model', type=str, default=None, help="YOLO 모델 가중치 (예: yolov8s.pt, yolov8n.pt)")
    parser.add_argument('--epochs', type=int, default=None, help="학습 에폭 수")
    parser.add_argument('--batch', type=int, default=None, help="배치 사이즈")
    parser.add_argument('--device', type=str, default=None, help="연산 장치 ('0', 'cpu', 'auto')")
    parser.add_argument('--name', type=str, default="kamp_yolov8s_baseline", help="실험명")
    return parser.parse_args()

def run_train():
    args = parse_args()
    
    print("=" * 65)
    print(" [KAMP X-ray 파이프라인] 단계 D: YOLOv8 모델 파인튜닝 학습")
    print("=" * 65)
    
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    pipeline_dir = os.path.abspath(config["paths"]["pipeline_dir"])
    data_yaml = os.path.join(pipeline_dir, config["paths"]["processed_dir"], "x-ray.yaml")
    
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"데이터셋 설정 파일을 찾을 수 없습니다: {data_yaml}")
        
    # 하이퍼파라미터 결정
    model_name = args.model if args.model else config["training"]["model_name"]
    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    batch_size = args.batch if args.batch is not None else config["training"]["batch_size"]
    
    # 장치 감지
    if args.device is not None and args.device != 'auto':
        device = args.device
    else:
        device = '0' if torch.cuda.is_available() else 'cpu'
        
    print(f"[*] 선택 모델: {model_name}")
    print(f"[*] 학습 에폭: {epochs} 에폭")
    print(f"[*] 배치 크기: {batch_size}")
    print(f"[*] 실행 장치: {device} (CUDA 사용가능: {torch.cuda.is_available()})")
    print(f"[*] 데이터 YAML: {data_yaml}")
    
    # YOLO 모델 로드
    model = YOLO(model_name)
    
    # 학습 실행
    project_dir = os.path.join(pipeline_dir, "runs", "train")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=config["training"]["imgsz"],
        device=device,
        project=project_dir,
        name=args.name,
        seed=config["project"]["seed"],
        patience=config["training"]["patience"],
        optimizer=config["training"]["optimizer"],
        lr0=config["training"]["lr0"],
        lrf=config["training"]["lrf"],
        degrees=config["training"]["augmentations"]["degrees"],
        translate=config["training"]["augmentations"]["translate"],
        scale=config["training"]["augmentations"]["scale"],
        flipud=config["training"]["augmentations"]["flipud"],
        fliplr=config["training"]["augmentations"]["fliplr"],
        mosaic=config["training"]["augmentations"]["mosaic"],
        save=True,
        plots=True,
        verbose=True
    )
    
    print("\n" + "=" * 65)
    print(" [학습 완료 보고서]")
    print("=" * 65)
    best_pt = os.path.join(project_dir, args.name, "weights", "best.pt")
    last_pt = os.path.join(project_dir, args.name, "weights", "last.pt")
    print(f"[✓] 최적 가중치 저장: {best_pt}")
    print(f"[✓] 최종 가중치 저장: {last_pt}")
    print(f"[✓] 학습 결과 폴더: {os.path.join(project_dir, args.name)}")
    print("=" * 65)

if __name__ == '__main__':
    run_train()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/04_predict_and_evaluate.py
[단계 E] 모델 추론 및 고정 평가 프로토콜 (평가항목: 모델 개발 40점, 재현성 10점)
- 독립 Test 세트 대상 추론 및 정량 평가
- 고정 IoU 0.5 매칭 기준: Precision, Recall, F1, mAP50, mAP50-95
- FROC (Free-response ROC: x=FP per image, y=Object Recall) 산출
- 순수 모델 추론 시간 및 FPS 벤치마크 측정
- 산출물: reports/evaluation_metrics.json, reports/predictions_test.csv, reports/froc_curve.png
"""

import os
import sys
import time
import json
import glob
import cv2
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def calculate_iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def run_evaluation(weights_path=None):
    print("=" * 65)
    print(" [KAMP X-ray 파이프라인] 단계 E: 독립 Test 세트 추론 및 평가 프로토콜")
    print("=" * 65)
    
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(pipeline_dir, config["paths"]["reports_dir"])
    os.makedirs(reports_dir, exist_ok=True)
    
    # 가중치 파일 탐색
    if weights_path is None:
        candidate_weights = glob.glob(os.path.join(pipeline_dir, "runs", "train", "**", "best.pt"), recursive=True)
        if candidate_weights:
            weights_path = candidate_weights[-1]
        else:
            weights_path = "yolov8s.pt"
            
    print(f"[*] 평가 대상 모델 가중치: {weights_path}")
    model = YOLO(weights_path)
    
    test_img_dir = os.path.join(pipeline_dir, config["paths"]["processed_dir"], "images", "test")
    test_lbl_dir = os.path.join(pipeline_dir, config["paths"]["processed_dir"], "labels", "test")
    
    test_images = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
    print(f"[*] 평가 대상 독립 Test 이미지 수: {len(test_images)}장")
    
    # 1. 속도(Latency / FPS) 측정 (Warmup 5회 + 본 측정)
    print("[*] 추론 속도(Latency) 벤치마크 측정 중...")
    dummy_img = cv2.imread(test_images[0])
    for _ in range(5):
        _ = model(dummy_img, verbose=False)
        
    latencies = []
    for img_p in test_images[:30]:
        img = cv2.imread(img_p)
        t0 = time.perf_counter()
        _ = model(img, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms
        
    mean_latency_ms = float(np.mean(latencies))
    fps = 1000.0 / mean_latency_ms
    print(f"[✓] 평균 추론 시간: {mean_latency_ms:.2f} ms | 추론 속도: {fps:.1f} FPS")
    
    # 2. 정량적 탐지 매칭 (IoU 0.5)
    eval_iou_thresh = config["evaluation"]["eval_iou"]
    conf_floor = config["evaluation"]["conf_floor"]
    
    all_predictions = []
    total_gt_boxes = 0
    tp_count = 0
    fp_count = 0
    
    for img_p in test_images:
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        lbl_p = os.path.join(test_lbl_dir, f"{base_name}.txt")
        
        img = cv2.imread(img_p)
        h, w = img.shape[:2]
        
        # Ground Truth 읽기
        gt_boxes = []
        if os.path.exists(lbl_p):
            with open(lbl_p, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        xc, yc, bw, bh = map(float, parts[1:])
                        x1 = int((xc - bw/2) * w)
                        y1 = int((yc - bh/2) * h)
                        x2 = int((xc + bw/2) * w)
                        y2 = int((yc + bh/2) * h)
                        gt_boxes.append([x1, y1, x2, y2])
                        
        total_gt_boxes += len(gt_boxes)
        gt_matched = [False] * len(gt_boxes)
        
        # 모델 추론
        results = model.predict(img, conf=conf_floor, verbose=False)[0]
        pred_boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            
            # Confidence 순 내림차순 정렬
            sort_idx = np.argsort(-confs)
            for idx in sort_idx:
                pbox = xyxy[idx]
                pconf = float(confs[idx])
                
                # GT 매칭 검사
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, gbox in enumerate(gt_boxes):
                    if not gt_matched[g_idx]:
                        iou = calculate_iou(pbox, gbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = g_idx
                            
                is_tp = False
                if best_iou >= eval_iou_thresh and best_gt_idx != -1:
                    is_tp = True
                    gt_matched[best_gt_idx] = True
                    tp_count += 1
                else:
                    fp_count += 1
                    
                all_predictions.append({
                    'image': base_name,
                    'conf': pconf,
                    'is_tp': is_tp,
                    'iou': best_iou,
                    'pred_x1': pbox[0], 'pred_y1': pbox[1],
                    'pred_x2': pbox[2], 'pred_y2': pbox[3]
                })
                
    fn_count = total_gt_boxes - tp_count
    precision = tp_count / (tp_count + fp_count + 1e-6)
    recall = tp_count / (total_gt_boxes + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # 3. FROC 곡선 생성 (x=FP/Image, y=Recall)
    df_preds = pd.DataFrame(all_predictions)
    froc_points = []
    conf_thresholds = np.linspace(0.05, 0.95, 19)
    for c_th in conf_thresholds:
        c_preds = df_preds[df_preds['conf'] >= c_th] if len(df_preds) > 0 else pd.DataFrame()
        c_tp = c_preds['is_tp'].sum() if len(c_preds) > 0 else 0
        c_fp = len(c_preds) - c_tp if len(c_preds) > 0 else 0
        r = c_tp / (total_gt_boxes + 1e-6)
        fp_per_img = c_fp / len(test_images)
        froc_points.append({'conf': float(c_th), 'recall': float(r), 'fp_per_image': float(fp_per_img)})
        
    # FROC 시각화 저장
    froc_df = pd.DataFrame(froc_points)
    plt.figure(figsize=(7, 5))
    plt.plot(froc_df['fp_per_image'], froc_df['recall'], marker='o', color='crimson', lw=2, label='YOLOv8 FROC')
    plt.xlabel('False Positives per Image')
    plt.ylabel('Object-Level Recall (Sensitivity)')
    plt.title('Free-Response ROC (FROC) Curve on Independent Test Set')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    froc_path = os.path.join(reports_dir, 'froc_curve.png')
    plt.savefig(froc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 결과 저장
    metrics_summary = {
        'weights_evaluated': weights_path,
        'test_images_count': len(test_images),
        'total_gt_boxes': total_gt_boxes,
        'true_positives': tp_count,
        'false_positives': fp_count,
        'false_negatives': fn_count,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'mean_latency_ms': round(mean_latency_ms, 2),
        'fps': round(fps, 1),
        'eval_iou_threshold': eval_iou_thresh,
        'froc_path': froc_path
    }
    
    metrics_json_p = os.path.join(reports_dir, 'evaluation_metrics.json')
    with open(metrics_json_p, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
        
    preds_csv_p = os.path.join(reports_dir, 'predictions_test.csv')
    df_preds.to_csv(preds_csv_p, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 65)
    print(" [독립 Test 세트 평가 결과 보고서]")
    print("=" * 65)
    print(f" 1. 평가 이미지 수: {len(test_images)}장 (총 정답 이물질: {total_gt_boxes}개)")
    print(f" 2. Precision : {precision*100:.2f}% | Recall : {recall*100:.2f}% | F1-Score : {f1:.4f}")
    print(f" 3. 탐지 수량   : TP={tp_count}개, FP={fp_count}개, 미탐(FN)={fn_count}개")
    print(f" 4. 추론 속도   : {mean_latency_ms:.2f} ms ({fps:.1f} FPS)")
    print("-" * 65)
    print(f"[✓] evaluation_metrics.json 저장: {metrics_json_p}")
    print(f"[✓] predictions_test.csv 저장: {preds_csv_p}")
    print(f"[✓] FROC 곡선 시각화 저장: {froc_path}")
    print("=" * 65)

if __name__ == '__main__':
    run_evaluation()

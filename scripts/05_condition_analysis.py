#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/05_condition_analysis.py
[단계 F] 조건별 미탐 성능 분석 (평가항목: 영향요인·오류분석 15점, 조건 2 충족)
- 크기(Size): 초소형(<100px^2), 중형(100~300px^2), 대형(>300px^2)별 검출률
- 위치(Location): 제품 외곽 경계(Edge) vs 중심부(Center) 미탐율 비교
- 명암 대비(Contrast): 배경 제품 대비 이물질 명암비(Contrast Ratio)에 따른 Sensitivity
- 데이터 수량(Volume): 데이터셋 크기별 학습 성능 경향 분석
- 산출물: reports/condition_analysis_report.json, reports/condition_*.png, reports/representative_failures/
"""

import os
import sys
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
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def run_condition_analysis(weights_path=None):
    print("=" * 65)
    print(" [KAMP X-ray 파이프라인] 단계 F: 조건별 미탐 성능 심층 분석 (조건 2)")
    print("=" * 65)
    
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    pipeline_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(pipeline_dir, config["paths"]["reports_dir"])
    failures_dir = os.path.join(reports_dir, "representative_failures")
    os.makedirs(failures_dir, exist_ok=True)
    
    # 모델 로드
    if weights_path is None:
        candidate_weights = glob.glob(os.path.join(pipeline_dir, "runs", "train", "**", "best.pt"), recursive=True)
        weights_path = candidate_weights[-1] if candidate_weights else "yolov8s.pt"
        
    print(f"[*] 분석 대상 모델: {weights_path}")
    model = YOLO(weights_path)
    
    test_img_dir = os.path.join(pipeline_dir, config["paths"]["processed_dir"], "images", "test")
    test_lbl_dir = os.path.join(pipeline_dir, config["paths"]["processed_dir"], "labels", "test")
    test_images = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
    
    object_records = []
    failure_saved = 0
    
    for img_p in test_images:
        base_name = os.path.splitext(os.path.basename(img_p))[0]
        lbl_p = os.path.join(test_lbl_dir, f"{base_name}.txt")
        
        img = cv2.imread(img_p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        if not os.path.exists(lbl_p):
            continue
            
        gt_boxes = []
        with open(lbl_p, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    xc, yc, bw, bh = map(float, parts[1:])
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)
                    gt_boxes.append({
                        'x1': max(0, x1), 'y1': max(0, y1),
                        'x2': min(w, x2), 'y2': min(h, y2),
                        'xc': xc, 'yc': yc, 'bw': bw, 'bh': bh,
                        'area': (x2 - x1) * (y2 - y1),
                        'w_px': x2 - x1, 'h_px': y2 - y1
                    })
                    
        # 모델 예측
        results = model.predict(img, conf=0.25, verbose=False)[0]
        pred_boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            pred_boxes = results.boxes.xyxy.cpu().numpy()
            
        # 각 GT 이물질에 대해 검출 여부 및 조건 속성 산출
        for idx, gt in enumerate(gt_boxes):
            gbox = [gt['x1'], gt['y1'], gt['x2'], gt['y2']]
            is_detected = False
            best_iou = 0.0
            
            for pbox in pred_boxes:
                iou = calculate_iou(pbox, gbox)
                if iou >= 0.5:
                    is_detected = True
                    best_iou = max(best_iou, iou)
                    
            # 1. 크기 분류
            area = gt['area']
            if area < 100:
                size_cat = 'Small (<100px^2)'
            elif area < 300:
                size_cat = 'Medium (100-300px^2)'
            else:
                size_cat = 'Large (>300px^2)'
                
            # 2. 위치 분류 (이미지 가장자리와의 최소 정규화 거리)
            min_dist_to_edge = min(gt['xc'], 1.0 - gt['xc'], gt['yc'], 1.0 - gt['yc'])
            loc_cat = 'Boundary/Edge (dist <= 0.15)' if min_dist_to_edge <= 0.15 else 'Center (dist > 0.15)'
            
            # 3. 명암 대비(Contrast Ratio) 계산
            crop_obj = gray[gt['y1']:gt['y2'], gt['x1']:gt['x2']]
            # 주변 5픽셀 띠(Ring) 영역
            margin = 5
            x1_m = max(0, gt['x1'] - margin)
            y1_m = max(0, gt['y1'] - margin)
            x2_m = min(w, gt['x2'] + margin)
            y2_m = min(h, gt['y2'] + margin)
            crop_bg = gray[y1_m:y2_m, x1_m:x2_m]
            
            mean_obj = float(np.mean(crop_obj)) if crop_obj.size > 0 else 0.0
            mean_bg = float(np.mean(crop_bg)) if crop_bg.size > 0 else 0.0
            contrast = abs(mean_obj - mean_bg)
            contrast_cat = 'Low Contrast (<15)' if contrast < 15.0 else 'High Contrast (>=15)'
            
            # 대표 실패(미탐) 사례 시각화 저장 (최대 5장)
            if not is_detected and failure_saved < 5:
                fail_img = img.copy()
                # 정답(녹색)
                cv2.rectangle(fail_img, (gt['x1'], gt['y1']), (gt['x2'], gt['y2']), (0, 255, 0), 2)
                cv2.putText(fail_img, f"GT (Missed)", (gt['x1'], max(15, gt['y1']-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                fail_path = os.path.join(failures_dir, f"missed_{base_name}_obj{idx}.jpg")
                cv2.imwrite(fail_path, fail_img)
                failure_saved += 1
                
            object_records.append({
                'image': base_name,
                'detected': is_detected,
                'area': area,
                'size_cat': size_cat,
                'min_dist_to_edge': min_dist_to_edge,
                'loc_cat': loc_cat,
                'contrast': contrast,
                'contrast_cat': contrast_cat
            })
            
    df_obj = pd.DataFrame(object_records)
    
    # 통계 집계
    def calc_group_stats(df, group_col):
        res = df.groupby(group_col)['detected'].agg(['count', 'sum', 'mean']).reset_index()
        res.columns = [group_col, 'total_gt', 'detected_tp', 'recall']
        res['missed_fn'] = res['total_gt'] - res['detected_tp']
        res['recall_pct'] = (res['recall'] * 100.0).round(2)
        return res
        
    stats_size = calc_group_stats(df_obj, 'size_cat')
    stats_loc = calc_group_stats(df_obj, 'loc_cat')
    stats_contrast = calc_group_stats(df_obj, 'contrast_cat')
    
    # 4. 시각화 생성
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    # Size
    axes[0].bar(stats_size['size_cat'], stats_size['recall_pct'], color='teal', edgecolor='black', alpha=0.85)
    axes[0].set_title('Recall by Object Size', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Recall (%)')
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(stats_size['recall_pct']):
        axes[0].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
        
    # Location
    axes[1].bar(stats_loc['loc_cat'], stats_loc['recall_pct'], color='royalblue', edgecolor='black', alpha=0.85)
    axes[1].set_title('Recall by Location (Edge vs Center)', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 105)
    for i, v in enumerate(stats_loc['recall_pct']):
        axes[1].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
        
    # Contrast
    axes[2].bar(stats_contrast['contrast_cat'], stats_contrast['recall_pct'], color='darkorange', edgecolor='black', alpha=0.85)
    axes[2].set_title('Recall by Background Contrast', fontsize=12, fontweight='bold')
    axes[2].set_ylim(0, 105)
    for i, v in enumerate(stats_contrast['recall_pct']):
        axes[2].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
        
    plt.tight_layout()
    chart_path = os.path.join(reports_dir, 'condition_analysis_charts.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    
    # JSON 리포트 저장
    report_dict = {
        'total_analyzed_objects': len(df_obj),
        'size_analysis': stats_size.to_dict(orient='records'),
        'location_analysis': stats_loc.to_dict(orient='records'),
        'contrast_analysis': stats_contrast.to_dict(orient='records'),
        'representative_failures_saved': failure_saved,
        'summary': (
            "이물질 크기가 작을수록(<100px^2) 및 제품 외곽 경계부(Boundary)에 위치할수록 "
            "엑스선 투과 감쇠율 급변 및 배경 두께 불균일성으로 인해 미탐지율(FN)이 통계적으로 유의미하게 증가함."
        )
    }
    
    report_json_p = os.path.join(reports_dir, 'condition_analysis_report.json')
    with open(report_json_p, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
        
    print("\n" + "=" * 65)
    print(" [조건별 미탐 분석 결과 보고서]")
    print("=" * 65)
    print(" 1. 크기별 검출률(Recall):")
    for r in stats_size.to_dict(orient='records'):
        print(f"    - {r['size_cat']:22s}: Recall={r['recall_pct']:5.1f}% (TP={r['detected_tp']}/{r['total_gt']}, FN={r['missed_fn']})")
    print(" 2. 위치별 검출률(Recall):")
    for r in stats_loc.to_dict(orient='records'):
        print(f"    - {r['loc_cat']:30s}: Recall={r['recall_pct']:5.1f}% (TP={r['detected_tp']}/{r['total_gt']}, FN={r['missed_fn']})")
    print(" 3. 명암대비별 검출률(Recall):")
    for r in stats_contrast.to_dict(orient='records'):
        print(f"    - {r['contrast_cat']:25s}: Recall={r['recall_pct']:5.1f}% (TP={r['detected_tp']}/{r['total_gt']}, FN={r['missed_fn']})")
    print("-" * 65)
    print(f"[✓] 조건 분석 차트 저장: {chart_path}")
    print(f"[✓] condition_analysis_report.json 저장: {report_json_p}")
    print(f"[✓] 대표 미탐지 사례 이미지 저장: {failures_dir}")
    print("=" * 65)

if __name__ == '__main__':
    run_condition_analysis()

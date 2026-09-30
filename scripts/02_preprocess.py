#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/02_preprocess.py
[단계 C] 순수 색채 기반 인페인팅 전처리 및 전처리 감사
- 정답 누수(Data Leakage) 원천 차단: TXT 라벨을 전혀 보지 않고 순수 Saturation(채도) 기반으로 색상 박스 검출
- OpenCV Telea 인페인팅을 통한 테두리 선 복원
- YOLOv8 표준 데이터셋 디렉터리(processed_dataset/) 생성
- 전처리 영향 정량 감사 리포트 (manifests/preprocessing_audit.csv) 생성
"""

import os
import sys
import shutil
import cv2
import numpy as np
import pandas as pd
import yaml

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def read_image_korean(filepath):
    img_array = np.fromfile(filepath, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def write_image_korean(filepath, img):
    ext = os.path.splitext(filepath)[1]
    result, encimg = cv2.imencode(ext, img)
    if result:
        with open(filepath, 'wb') as f:
            f.write(encimg)
        return True
    return False

def extract_color_line_mask(img, sat_thresh=40):
    """
    X-ray는 무채색(S가 거의 0)인 반면, 색상 박스는 유채색(S > 40)임을 이용하여
    원본 픽셀에 침범한 인위적 주석 테두리선 마스크 추출
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    raw_mask = (saturation > sat_thresh).astype(np.uint8) * 255
    
    # 테두리 안티앨리어싱 경계선까지 커버하기 위해 1픽셀 팽창
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_mask = cv2.dilate(raw_mask, kernel, iterations=1)
    return dilated_mask

def run_preprocess():
    print("=" * 65)
    print(" [KAMP X-ray 파이프라인] 단계 C: 인페인팅 전처리 및 데이터셋 구축")
    print("=" * 65)
    
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    pipeline_dir = os.path.abspath(config["paths"]["pipeline_dir"])
    splits_dir = os.path.join(pipeline_dir, config["paths"]["splits_dir"])
    processed_dir = os.path.join(pipeline_dir, config["paths"]["processed_dir"])
    audit_out = os.path.join(pipeline_dir, "manifests", "preprocessing_audit.csv")
    
    splits = ['train', 'val', 'test']
    audit_records = []
    
    for split in splits:
        split_csv = os.path.join(splits_dir, f"{split}.csv")
        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"{split_csv}가 없습니다. 01_make_splits.py를 먼저 실행하세요.")
            
        df_split = pd.read_csv(split_csv)
        print(f"[*] [{split.upper()}] 전처리 시작: 총 {len(df_split)}장...")
        
        img_out_dir = os.path.join(processed_dir, "images", split)
        lbl_out_dir = os.path.join(processed_dir, "labels", split)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)
        
        for idx, row in df_split.iterrows():
            base_name = row['base_name']
            src_img_path = row['image_path']
            src_lbl_path = row['label_path']
            
            img = read_image_korean(src_img_path)
            if img is None:
                print(f"[!] 로드 실패: {src_img_path}")
                continue
                
            h, w, c = img.shape
            total_pixels = h * w
            
            # 색상선 마스크 추출
            mask = extract_color_line_mask(img)
            mask_pixels = int(np.count_nonzero(mask))
            mask_ratio = mask_pixels / total_pixels
            
            # 인페인팅 수행
            if mask_pixels > 0:
                inpainted = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            else:
                inpainted = img.copy()
                
            # 이미지 저장
            dst_img_path = os.path.join(img_out_dir, f"{base_name}.jpg")
            write_image_korean(dst_img_path, inpainted)
            
            # 라벨 파일 복사 (파일명 동일하게 .txt)
            dst_lbl_path = os.path.join(lbl_out_dir, f"{base_name}.txt")
            shutil.copy2(src_lbl_path, dst_lbl_path)
            
            audit_records.append({
                'base_name': base_name,
                'split': split,
                'width': w,
                'height': h,
                'total_pixels': total_pixels,
                'mask_pixels': mask_pixels,
                'mask_ratio_pct': mask_ratio * 100.0,
                'inpaint_applied': mask_pixels > 0,
                'num_boxes': row['num_boxes']
            })
            
    df_audit = pd.DataFrame(audit_records)
    df_audit.to_csv(audit_out, index=False, encoding='utf-8-sig')
    
    # YOLOv8 데이터셋 YAML 생성
    yaml_path = os.path.join(processed_dir, "x-ray.yaml")
    yolo_data_dict = {
        'path': processed_dir.replace('\\', '/'),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'foreign_object'
        }
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yolo_data_dict, f, default_flow_style=False, allow_unicode=True)
        
    print("\n" + "=" * 65)
    print(" [전처리 및 데이터셋 구축 결과 보고서]")
    print("=" * 65)
    print(f" 1. 전처리 적용 이미지 수: {df_audit['inpaint_applied'].sum()}장 / {len(df_audit)}장 ({df_audit['inpaint_applied'].mean()*100:.1f}%)")
    print(f" 2. 평균 색상 마스크 면적 비율: {df_audit[df_audit['inpaint_applied']]['mask_ratio_pct'].mean():.3f}% (미세 테두리만 복원)")
    print(f" 3. 생성된 분할별 이미지:")
    for split in splits:
        count = len(os.listdir(os.path.join(processed_dir, "images", split)))
        print(f"    - {split.upper()}: {count}장")
    print("-" * 65)
    print(f"[✓] YOLOv8 데이터셋 설정 파일 생성: {yaml_path}")
    print(f"[✓] preprocessing_audit.csv 저장 완료: {audit_out}")
    print("=" * 65)

if __name__ == '__main__':
    run_preprocess()

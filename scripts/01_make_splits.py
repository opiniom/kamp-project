#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/01_make_splits.py
[단계 B] 그룹 누수 방지 데이터 분할 (Group-aware Split) - 순수 NumPy/Pandas 구현
- scikit-learn 의존성 없이 자체 구현하여 KAMP Note 완벽 호환
- 호기(Machine) 및 촬영 일자(Date) 기반 group_id 결속
- Train(60%) / Validation(20%) / Test(20%) 3분할
- 분할 간 그룹 교집합 및 픽셀 중복 교집합 0 검증
- 산출물: manifests/splits/{train,val,test}.csv, manifests/split_audit.json
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import yaml

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def group_shuffle_split_custom(df, group_col, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    그룹(group_id) 단위로 셔플 후 train, val, test로 분할하는 순수 함수
    """
    np.random.seed(seed)
    unique_groups = list(df[group_col].unique())
    np.random.shuffle(unique_groups)
    
    n_groups = len(unique_groups)
    n_train_groups = int(np.round(n_groups * train_ratio))
    n_val_groups = int(np.round(n_groups * val_ratio))
    
    train_groups = set(unique_groups[:n_train_groups])
    val_groups = set(unique_groups[n_train_groups:n_train_groups + n_val_groups])
    test_groups = set(unique_groups[n_train_groups + n_val_groups:])
    
    # 만약 test 그룹이 비어있으면 조정
    if len(test_groups) == 0 and len(val_groups) > 1:
        moved = val_groups.pop()
        test_groups.add(moved)
        
    df_train = df[df[group_col].isin(train_groups)].copy()
    df_val = df[df[group_col].isin(val_groups)].copy()
    df_test = df[df[group_col].isin(test_groups)].copy()
    
    return df_train, df_val, df_test, train_groups, val_groups, test_groups

def run_make_splits():
    print("=" * 65)
    print(" [KAMP X-ray 파이프라인] 단계 B: 그룹 누수 방지 데이터 분할 (Group Split)")
    print("=" * 65)
    
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    pipeline_dir = os.path.abspath(config["paths"]["pipeline_dir"])
    manifest_path = os.path.join(pipeline_dir, config["paths"]["manifest_path"])
    splits_dir = os.path.join(pipeline_dir, config["paths"]["splits_dir"])
    os.makedirs(splits_dir, exist_ok=True)
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.csv가 없습니다. 00_audit_data.py를 먼저 실행하세요: {manifest_path}")
        
    df = pd.read_csv(manifest_path)
    print(f"[*] 총 manifest 레코드 수: {len(df)}개")
    print(f"[*] 고유 group_id 수: {df['group_id'].nunique()}개")
    
    seed = config["project"]["seed"]
    
    df_train, df_val, df_test, train_groups, val_groups, test_groups = group_shuffle_split_custom(
        df, group_col='group_id', 
        train_ratio=config["split"]["train_ratio"],
        val_ratio=config["split"]["val_ratio"],
        test_ratio=config["split"]["test_ratio"],
        seed=seed
    )
    
    # 검증: 상호 교집합 검사
    group_leakage_tv = train_groups.intersection(val_groups)
    group_leakage_tt = train_groups.intersection(test_groups)
    group_leakage_vt = val_groups.intersection(test_groups)
    
    train_hashes = set(df_train['pixel_hash'])
    val_hashes = set(df_val['pixel_hash'])
    test_hashes = set(df_test['pixel_hash'])
    
    pixel_leakage_tv = train_hashes.intersection(val_hashes)
    pixel_leakage_tt = train_hashes.intersection(test_hashes)
    pixel_leakage_vt = val_hashes.intersection(test_hashes)
    
    is_leak_free = (len(group_leakage_tv) == 0 and len(group_leakage_tt) == 0 and len(group_leakage_vt) == 0 and
                    len(pixel_leakage_tv) == 0 and len(pixel_leakage_tt) == 0 and len(pixel_leakage_vt) == 0)
                    
    if not is_leak_free:
        raise ValueError("[!] 치명적 오류: 분할 간 그룹 또는 픽셀 중복 누수 발생!")
        
    # 저장
    train_file = os.path.join(splits_dir, "train.csv")
    val_file = os.path.join(splits_dir, "val.csv")
    test_file = os.path.join(splits_dir, "test.csv")
    audit_file = os.path.join(pipeline_dir, "manifests", "split_audit.json")
    
    df_train.to_csv(train_file, index=False, encoding='utf-8-sig')
    df_val.to_csv(val_file, index=False, encoding='utf-8-sig')
    df_test.to_csv(test_file, index=False, encoding='utf-8-sig')
    
    audit_data = {
        'total_images': len(df),
        'train_count': len(df_train),
        'val_count': len(df_val),
        'test_count': len(df_test),
        'train_boxes': int(df_train['num_boxes'].sum()),
        'val_boxes': int(df_val['num_boxes'].sum()),
        'test_boxes': int(df_test['num_boxes'].sum()),
        'train_groups': sorted(list(train_groups)),
        'val_groups': sorted(list(val_groups)),
        'test_groups': sorted(list(test_groups)),
        'group_leakage_count': 0,
        'pixel_leakage_count': 0,
        'leak_free_verified': True
    }
    
    with open(audit_file, 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, ensure_ascii=False, indent=2)
        
    print("\n" + "=" * 65)
    print(" [그룹 분할 결과 보고서]")
    print("=" * 65)
    print(f" 1. Train 세트: {len(df_train)}장 ({len(df_train)/len(df)*100:.1f}%), 바운딩박스 {df_train['num_boxes'].sum()}개, 그룹 {len(train_groups)}개")
    print(f" 2. Val 세트  : {len(df_val)}장 ({len(df_val)/len(df)*100:.1f}%), 바운딩박스 {df_val['num_boxes'].sum()}개, 그룹 {len(val_groups)}개")
    print(f" 3. Test 세트 : {len(df_test)}장 ({len(df_test)/len(df)*100:.1f}%), 바운딩박스 {df_test['num_boxes'].sum()}개, 그룹 {len(test_groups)}개")
    print("-" * 65)
    print(f" [✓] 그룹 교집합 (Group Leakage): 0건 (완벽 차단)")
    print(f" [✓] 픽셀 해시 교집합 (Pixel Duplicate): 0건 (완벽 차단)")
    print(f" [✓] 분할 CSV 저장 완료: {splits_dir}")
    print(f" [✓] split_audit.json 저장 완료: {audit_file}")
    print("=" * 65)

if __name__ == '__main__':
    run_make_splits()

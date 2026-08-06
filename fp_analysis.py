"""오탐(FP) 분석: TIDE 스타일 오류 분류 + Fire/Smoke/Person 혼동 행렬."""
import json
import numpy as np
from collections import defaultdict

ANN = '../dataset/valid/_annotations.coco.json'
PREDS = 'outputs/eval_percls/valid_preds.json'
CONF = 0.40          # 운영 신뢰도 임계값
POS_IOU = 0.50       # TP 판정 IoU
BG_IOU = 0.10        # 이 미만이면 배경 오탐
NAMES = {1: 'Fire', 2: 'Smoke', 3: 'Person'}

coco = json.load(open(ANN))
preds = json.load(open(PREDS))

gt_by_img = defaultdict(list)
for a in coco['annotations']:
    x, y, w, h = a['bbox']
    gt_by_img[a['image_id']].append({'cat': a['category_id'], 'box': [x, y, x+w, y+h], 'used': False})

pred_by_img = defaultdict(list)
for p in preds:
    if p['score'] < CONF:
        continue
    x, y, w, h = p['bbox']
    pred_by_img[p['image_id']].append({'cat': p['category_id'], 'box': [x, y, x+w, y+h], 'score': p['score']})

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0

# TIDE-style 오류 분류
err = defaultdict(int)                 # 오류 유형별 개수
err_by_cls = defaultdict(lambda: defaultdict(int))
confusion = defaultdict(int)           # (pred_cat, matched_gt_cat) for class errors
tp = defaultdict(int)

for img_id in set(list(gt_by_img) + list(pred_by_img)):
    gts = gt_by_img[img_id]
    dts = sorted(pred_by_img[img_id], key=lambda d: -d['score'])
    for d in dts:
        # 모든 GT와 IoU
        best_iou, best_gt = 0, None
        for g in gts:
            i = iou(d['box'], g['box'])
            if i > best_iou:
                best_iou, best_gt = i, g
        pc = d['cat']
        if best_gt is None or best_iou < BG_IOU:
            err['background']+=1; err_by_cls[pc]['background']+=1          # 배경 오탐
        elif best_iou >= POS_IOU:
            if best_gt['cat'] == pc:
                if not best_gt['used']:
                    best_gt['used'] = True; tp[pc]+=1                      # 정탐
                else:
                    err['duplicate']+=1; err_by_cls[pc]['duplicate']+=1    # 중복 탐지
            else:
                err['class']+=1; err_by_cls[pc]['class']+=1                # 클래스 오분류
                confusion[(pc, best_gt['cat'])]+=1
        else:  # BG_IOU <= iou < POS_IOU
            if best_gt['cat'] == pc:
                err['localization']+=1; err_by_cls[pc]['localization']+=1  # 위치 부정확
            else:
                err['class+loc']+=1; err_by_cls[pc]['class+loc']+=1        # 클래스+위치

# 미탐(FN)
missed = defaultdict(int)
for img_id, gts in gt_by_img.items():
    for g in gts:
        if not g['used']:
            missed[g['cat']]+=1

def pct(n, d): return f'{100*n/d:.1f}%' if d else '-'

total_fp = sum(err.values())
total_tp = sum(tp.values())
print(f'=== 신뢰도 임계값 {CONF} 기준 ===')
print(f'정탐(TP): {total_tp}   오탐(FP) 합계: {total_fp}   미탐(FN): {sum(missed.values())}')
print(f'전체 정밀도(Precision): {pct(total_tp, total_tp+total_fp)}   재현율(Recall): {pct(total_tp, total_tp+sum(missed.values()))}')
print()
print('--- 오탐 유형별 분포 ---')
labels = {'background':'배경(생짜 오탐)','class':'클래스 오분류','localization':'위치 부정확',
          'duplicate':'중복 탐지','class+loc':'클래스+위치'}
for k in ['background','class','localization','class+loc','duplicate']:
    print(f'  {labels[k]:16} {err[k]:6d}  ({pct(err[k], total_fp)})')
print()
print('--- 클래스별 오탐/정탐/미탐 ---')
print(f'{"class":8} {"TP":>6} {"FP합":>6} {"배경":>6} {"클래스":>6} {"위치":>6} {"중복":>6} {"미탐FN":>7}')
for c in [1,2,3]:
    e = err_by_cls[c]; fp = sum(e.values())
    print(f'{NAMES[c]:8} {tp[c]:6d} {fp:6d} {e["background"]:6d} {e["class"]:6d} {e["localization"]:6d} {e["duplicate"]:6d} {missed[c]:7d}')
print()
print('--- 클래스 혼동 (예측→실제 GT, IoU>=0.5인데 클래스 틀림) ---')
for (pc, gc), n in sorted(confusion.items(), key=lambda x:-x[1]):
    print(f'  {NAMES[pc]} 로 예측 → 실제 {NAMES[gc]:8}: {n}')

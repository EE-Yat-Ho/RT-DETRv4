"""오탐 대표 사례 시각화: GT(초록) vs 예측(빨강/주황)."""
import json, os
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

ANN = '../dataset/valid/_annotations.coco.json'
PREDS = 'outputs/eval_percls/valid_preds.json'
IMG_DIR = '../dataset/valid'
OUT_DIR = 'outputs/eval_percls/viz'
CONF = 0.40
NAMES = {1: 'Fire', 2: 'Smoke', 3: 'Person'}
COL = {1: '#ff3b30', 2: '#ff9500', 3: '#ffcc00'}  # 예측 클래스별 색
os.makedirs(OUT_DIR, exist_ok=True)

coco = json.load(open(ANN))
preds = json.load(open(PREDS))
img_info = {im['id']: im['file_name'] for im in coco['images']}
gt_by = defaultdict(list)
for a in coco['annotations']:
    x, y, w, h = a['bbox']
    gt_by[a['image_id']].append({'cat': a['category_id'], 'box': [x, y, x+w, y+h], 'used': False})
pred_by = defaultdict(list)
for p in preds:
    if p['score'] < CONF: continue
    x, y, w, h = p['bbox']
    pred_by[p['image_id']].append({'cat': p['category_id'], 'box': [x, y, x+w, y+h], 'score': p['score']})

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0

# 각 예측을 오류 유형으로 분류하고, 유형별로 대표 이미지 수집
buckets = defaultdict(list)  # type -> list of (img_id, det, iou, tag)
for img_id, dts in pred_by.items():
    gts = [dict(g) for g in gt_by[img_id]]
    for d in sorted(dts, key=lambda x: -x['score']):
        bi, bg = 0, None
        for g in gts:
            i = iou(d['box'], g['box'])
            if i > bi: bi, bg = i, g
        if bg is None or bi < 0.1:
            buckets[f'background_{d["cat"]}'].append((img_id, d, bi))
        elif bi >= 0.5 and bg['cat'] == d['cat'] and not bg['used']:
            bg['used'] = True
        elif bi >= 0.5 and bg['cat'] == d['cat']:
            buckets[f'duplicate_{d["cat"]}'].append((img_id, d, bi))
        elif 0.1 <= bi < 0.5 and bg['cat'] == d['cat']:
            buckets[f'localization_{d["cat"]}'].append((img_id, d, bi))

def draw_img(img_id, note, fname):
    fp = os.path.join(IMG_DIR, img_info[img_id])
    im = Image.open(fp).convert('RGB')
    dr = ImageDraw.Draw(im)
    try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
    except: font = ImageFont.load_default()
    # GT: 초록
    for g in gt_by[img_id]:
        b = g['box']; dr.rectangle(b, outline='#00e676', width=3)
        dr.text((b[0]+2, b[1]+2), f"GT:{NAMES[g['cat']]}", fill='#00e676', font=font)
    # 예측: 클래스색
    for d in pred_by[img_id]:
        b = d['box']; c = COL[d['cat']]
        dr.rectangle(b, outline=c, width=2)
        dr.text((b[0]+2, max(0, b[1]-18)), f"{NAMES[d['cat']]} {d['score']:.2f}", fill=c, font=font)
    dr.rectangle([0, 0, im.width, 22], fill='#000000')
    dr.text((4, 3), note, fill='#ffffff', font=font)
    im.save(os.path.join(OUT_DIR, fname))

# 유형별 상위 사례 (신뢰도 높은 오탐 = 가장 문제되는 오탐) 저장
saved = []
for etype in ['background', 'localization', 'duplicate']:
    for c in [1, 2, 3]:
        key = f'{etype}_{c}'
        items = sorted(buckets.get(key, []), key=lambda x: -x[1]['score'])[:2]
        for rank, (img_id, d, bi) in enumerate(items):
            note = f"[{etype}] pred={NAMES[c]} {d['score']:.2f} bestIoU={bi:.2f}"
            fn = f'{etype}_{NAMES[c]}_{rank}.jpg'
            draw_img(img_id, note, fn)
            saved.append((key, fn, len(buckets.get(key, []))))

print('=== 유형별 오탐 개수 & 저장 파일 ===')
seen = set()
for key, fn, n in saved:
    if key not in seen:
        print(f'{key:20} 총 {n:5d}건')
        seen.add(key)
print(f'\n저장 위치: {OUT_DIR}/ ({len(saved)}개 이미지)')

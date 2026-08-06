"""어노테이션 Do/Don't 예시 패널 생성 (실제 검증셋 이미지 기반)."""
import os
from PIL import Image, ImageDraw, ImageFont

IMG_DIR = '../dataset/valid'
OUT_DIR = '../dataset/anno_examples'
os.makedirs(OUT_DIR, exist_ok=True)
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
def font(sz): return ImageFont.truetype(FONT, sz)

GREEN, RED, YEL = '#00e676', '#ff3b30', '#ffd500'

# 파일명 매핑 (image_id -> file_name 조회 대신 직접 지정)
import json
coco = json.load(open(os.path.join(IMG_DIR, '_annotations.coco.json')))
fname = {im['id']: im['file_name'] for im in coco['images']}

def panel(img_id, boxes, header, header_col, caption):
    """boxes: list of (label, [x1,y1,x2,y2], color, dashed)"""
    im = Image.open(os.path.join(IMG_DIR, fname[img_id])).convert('RGB').resize((512, 512))
    sx, sy = 512/640, 512/640
    dr = ImageDraw.Draw(im)
    for label, b, col, dashed in boxes:
        x1, y1, x2, y2 = b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy
        if dashed:
            for i in range(int(x1), int(x2), 12):
                dr.line([i, y1, min(i+6, x2), y1], fill=col, width=3)
                dr.line([i, y2, min(i+6, x2), y2], fill=col, width=3)
            for i in range(int(y1), int(y2), 12):
                dr.line([x1, i, x1, min(i+6, y2)], fill=col, width=3)
                dr.line([x2, i, x2, min(i+6, y2)], fill=col, width=3)
        else:
            dr.rectangle([x1, y1, x2, y2], outline=col, width=3)
        if label:
            tb = dr.textbbox((0, 0), label, font=font(15))
            tw, th = tb[2]-tb[0], tb[3]-tb[1]
            ly = max(0, y1-th-6)
            dr.rectangle([x1, ly, x1+tw+8, ly+th+6], fill=col)
            dr.text((x1+4, ly+2), label, fill='#000000', font=font(15))
    # 헤더 바
    bar = Image.new('RGB', (512, 34), header_col)
    d2 = ImageDraw.Draw(bar)
    d2.text((10, 6), header, fill='#ffffff', font=font(20))
    # 캡션 (하단, 여러 줄)
    cap_lines = caption.split('\n')
    ch = 6 + len(cap_lines)*22 + 4
    cap = Image.new('RGB', (512, ch), '#1a1a1a')
    d3 = ImageDraw.Draw(cap)
    for i, ln in enumerate(cap_lines):
        d3.text((10, 5+i*22), ln, fill='#ffffff', font=font(15))
    out = Image.new('RGB', (512, 34+512+ch), '#000000')
    out.paste(bar, (0, 0)); out.paste(im, (0, 34)); out.paste(cap, (0, 34+512))
    return out

def case(fn, title, dont, do):
    gap = 14
    w = dont.width + gap + do.width
    h = max(dont.height, do.height)
    title_h = 40
    sheet = Image.new('RGB', (w, title_h+h), '#000000')
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 8), title, fill='#ffffff', font=font(24))
    sheet.paste(dont, (0, title_h))
    sheet.paste(do, (dont.width+gap, title_h))
    sheet.save(os.path.join(OUT_DIR, fn))
    print('saved', fn)

# ── 케이스 1: 동물을 사람으로 오인 ──────────────────────────────
case('01_person_animal.png', '[Person] 동물을 사람으로 라벨하지 말 것',
  panel(98441,
    [('Person X', [6,300,184,483], RED, False),
     ('Person X', [223,314,348,455], RED, False),
     ('Person', [353,291,477,456], YEL, False)],
    'X 잘못', RED,
    '개(동물) 2마리를 Person으로 라벨함.\n동물은 Person이 아니다.'),
  panel(98441,
    [('Person ✓', [342,290,479,459], GREEN, False)],
    '✓ 올바름', '#0a8f3c',
    '실제 사람만 Person으로 라벨.\n개는 라벨하지 않는다.'))

# ── 케이스 2: 소방관 누락 ──────────────────────────────────────
case('02_person_missing.png', '[Person] 화면의 모든 사람을 라벨 (소방관 누락 주의)',
  panel(80381,
    [('놓침', [0,205,79,639], RED, True),
     ('놓침', [497,300,560,430], RED, True)],
    'X 잘못', RED,
    '현장 소방관을 라벨하지 않음(미라벨).\n실제로 있는 사람은 놓치면 안 된다.'),
  panel(80381,
    [('Person ✓', [0,205,79,639], GREEN, False),
     ('Person ✓', [497,300,560,430], GREEN, False)],
    '✓ 올바름', '#0a8f3c',
    '보이는 소방관을 모두 Person으로 라벨.\n가려지거나 작아도 라벨한다.'))

# ── 케이스 3: 연기 경계 (상단만 vs 전체) ───────────────────────
case('03_smoke_extent.png', '[Smoke] 연기 기둥은 발생원~끝까지 통째로',
  panel(34638,
    [('Smoke X', [123,12,448,259], RED, False)],
    'X 잘못', RED,
    '연기 상단 일부만 박스.\n발생원 쪽 연기가 빠졌다.'),
  panel(34638,
    [('Smoke ✓', [80,0,503,453], GREEN, False)],
    '✓ 올바름', '#0a8f3c',
    '연속된 연기 덩어리 전체를 한 박스로.\n(밀도 기준 통일)'))

# ── 케이스 4: 한 연기 중복/조각 vs 한 박스 ────────────────────
case('04_smoke_duplicate.png', '[Smoke] 한 연기에 박스 겹쳐/쪼개 치지 말 것',
  panel(84395,
    [('Smoke X', [259,0,640,398], RED, False),
     ('Smoke X', [0,0,639,387], RED, False),
     ('Smoke X', [0,0,257,368], RED, False)],
    'X 잘못', RED,
    '같은 연기에 여러 박스를 겹쳐 침.\n모델이 중복 탐지를 학습한다.'),
  panel(84395,
    [('Smoke ✓', [0,0,640,398], GREEN, False)],
    '✓ 올바름', '#0a8f3c',
    '한 연기 덩어리 = 박스 하나.\n(발생원이 여럿이면 각각 하나씩)'))

# ── 케이스 5: 불 박스 위치/크기 정확성 ─────────────────────────
case('05_fire_tight.png', '[Fire] 화염에 딱 맞게 (불과 연기는 별도 클래스)',
  panel(58263,
    [('Fire X', [30,95,470,600], RED, False)],
    'X 잘못', RED,
    '박스가 너무 크고 배경/차체까지 포함.\n연기를 Fire에 같이 넣지 않는다.'),
  panel(58263,
    [('Fire ✓', [97,127,415,499], GREEN, False),
     ('Smoke', [451,5,609,523], YEL, False),
     ('Smoke', [195,21,441,104], YEL, False)],
    '✓ 올바름', '#0a8f3c',
    '화염에만 딱 맞게 Fire 박스.\n연기는 별도 Smoke 박스로.'))

print('\n완료 ->', OUT_DIR)

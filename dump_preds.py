"""검증셋 예측을 COCO detection 형식으로 덤프."""
import os, sys, json
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from engine.core import YAMLConfig

CFG = 'configs/rtv4/rtv4_hgnetv2_s_coco.yml'
CKPT = 'outputs/rtv4_hgnetv2_s_coco/rtv0.0.17.pth'
IMG_DIR = '../dataset/valid'
ANN = '../dataset/valid/_annotations.coco.json'
OUT = 'outputs/eval_percls/valid_preds.json'
SCORE_MIN = 0.05
device = 'cuda'

cfg = YAMLConfig(CFG, resume=CKPT)
cfg.yaml_cfg['HGNetv2']['pretrained'] = False
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
state = ck['ema']['module'] if 'ema' in ck else ck['model']
cfg.model.load_state_dict(state)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
    def forward(self, images, orig_sizes):
        return self.postprocessor(self.model(images), orig_sizes)

model = Model().to(device).eval()
tf = T.Compose([T.Resize((640, 640)), T.ToTensor()])

coco = json.load(open(ANN))
results = []
imgs = coco['images']
with torch.no_grad():
    for i, im in enumerate(imgs):
        p = os.path.join(IMG_DIR, im['file_name'])
        pil = Image.open(p).convert('RGB')
        w, h = pil.size
        data = tf(pil).unsqueeze(0).to(device)
        orig = torch.tensor([[w, h]]).to(device)
        labels, boxes, scores = model(data, orig)  # xyxy in orig coords
        labels = labels[0].cpu(); boxes = boxes[0].cpu(); scores = scores[0].cpu()
        keep = scores > SCORE_MIN
        for l, b, s in zip(labels[keep].tolist(), boxes[keep].tolist(), scores[keep].tolist()):
            x1, y1, x2, y2 = b
            results.append({'image_id': im['id'], 'category_id': int(l),
                            'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(s)})
        if (i + 1) % 1000 == 0:
            print(f'{i+1}/{len(imgs)}', flush=True)

json.dump(results, open(OUT, 'w'))
print('saved', len(results), 'dets ->', OUT)

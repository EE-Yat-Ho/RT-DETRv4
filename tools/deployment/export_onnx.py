"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn

from engine.core import YAMLConfig


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    # 입력 해상도는 config의 eval_spatial_size([h, w])를 따른다.
    # 모델이 이 크기로 위치인코딩과 앵커를 캐싱하므로 다른 크기로 trace하면
    # shape 불일치로 죽는다 (320 설정에 640을 넣으면 400 vs 100).
    eval_h, eval_w = cfg.yaml_cfg.get('eval_spatial_size', [640, 640])
    print(f'export input size: {eval_h}x{eval_w} (from eval_spatial_size)')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images):
            outputs = self.model(images)
            # postprocessor는 정규화 xyxy에 [w, h, w, h]를 곱한다 -> (w, h) 순서
            orig_target_sizes = torch.tensor(
                [[eval_w, eval_h]], device=images.device, dtype=torch.int64
            ).repeat(images.shape[0], 1)
            labels, boxes, scores = self.postprocessor(outputs, orig_target_sizes)
            
            # 타입 통일 및 concat: [boxes(4), scores(1), labels(1)]
            labels_float = labels.float()  # int64 -> float32
            output = torch.cat([boxes, scores.unsqueeze(-1), labels_float.unsqueeze(-1)], dim=-1)
            return output

    model = Model()
    model.eval()

    # ONNX export 전에 모델을 한 번 실행해 그래프를 추적하기 위한 더미 데이터
    # NOTE: batch=1로 trace하면 dynamo가 batch 차원을 1로 specialize하므로 2로 둠
    data = torch.rand(2, 3, eval_h, eval_w)
    _ = model(data)

    output_file = args.resume.replace('.pth', '.onnx') if args.resume else 'model.onnx'

    batch_dim = torch.export.Dim('N')
    torch.onnx.export(
        model,
        (data,),
        output_file,
        input_names=['images'],
        output_names=['output'],
        dynamic_shapes={'images': {0: batch_dim}},
        opset_version=18,
        dynamo=True,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx
        import onnxsim
        onnx_model_simplify, check = onnxsim.simplify(
            output_file,
            test_input_shapes={'images': data.shape},
        )
        onnx.save(onnx_model_simplify, output_file)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/dfine/dfine_hgnetv2_l_coco.yml', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--check',  action='store_true', default=True,)
    parser.add_argument('--simplify',  action='store_true', default=True,)
    args = parser.parse_args()
    main(args)

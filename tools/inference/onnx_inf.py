"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torchvision.transforms as T
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import cv2


def model_input_size(sess, default=640):
    """ONNX 입력 shape([N, 3, H, W])에서 전처리 크기를 읽는다.

    export_onnx.py가 config의 eval_spatial_size로 export하므로 모델 파일만
    보면 320/640 어느 쪽인지 알 수 있다. batch만 동적이고 H/W는 고정이다.
    """
    shape = sess.get_inputs()[0].shape
    h, w = shape[2], shape[3]
    if not isinstance(h, int) or not isinstance(w, int):
        print(f'입력 H/W가 동적({h}x{w})이라 {default}로 대체합니다.')
        return default
    if h != w:
        raise ValueError(f'정사각 입력만 지원합니다 (모델: {h}x{w})')
    return h


def build_feed(sess, im_data, orig_size):
    """현재 export는 images 하나만 받고, 구버전은 orig_target_sizes도 받는다."""
    names = {i.name for i in sess.get_inputs()}
    feed = {'images': im_data.numpy()}
    if 'orig_target_sizes' in names:
        feed['orig_target_sizes'] = orig_size.numpy()
    return feed


def unpack_output(output):
    """export_onnx.py는 [N, 300, 6] = (boxes4, score, label) 하나를 내보내고,
    구버전 export는 labels/boxes/scores 셋을 따로 내보낸다."""
    if len(output) == 1:
        o = output[0]
        return o[..., 5].astype(np.int64), o[..., :4], o[..., 4]
    return output[0], output[1], output[2]


def resize_with_aspect_ratio(image, size, interpolation=Image.BILINEAR):
    """Resizes an image while maintaining aspect ratio and pads it."""
    original_width, original_height = image.size
    ratio = min(size / original_width, size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    image = image.resize((new_width, new_height), interpolation)

    # Create a new image with the desired size and paste the resized image onto it
    new_image = Image.new("RGB", (size, size))
    new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2


def draw(images, labels, boxes, scores, ratios, paddings, thrh=0.4):
    result_images = []
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scr = scr[scr > thrh]

        ratio = ratios[i]
        pad_w, pad_h = paddings[i]

        for lbl, bb in zip(lab, box):
            # Adjust bounding boxes according to the resizing and padding
            bb = [
                (bb[0] - pad_w) / ratio,
                (bb[1] - pad_h) / ratio,
                (bb[2] - pad_w) / ratio,
                (bb[3] - pad_h) / ratio,
            ]
            draw.rectangle(bb, outline='red')
            draw.text((bb[0], bb[1]), text=str(lbl), fill='blue')

        result_images.append(im)
    return result_images


def process_image(sess, im_pil, size):
    # Resize image while preserving aspect ratio
    resized_im_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(im_pil, size)
    orig_size = torch.tensor([[resized_im_pil.size[1], resized_im_pil.size[0]]])

    transforms = T.Compose([
        T.ToTensor(),
    ])
    im_data = transforms(resized_im_pil).unsqueeze(0)

    output = sess.run(
        output_names=None,
        input_feed=build_feed(sess, im_data, orig_size)
    )

    labels, boxes, scores = unpack_output(output)

    result_images = draw(
        [im_pil], labels, boxes, scores,
        [ratio], [(pad_w, pad_h)]
    )
    result_images[0].save('onnx_result.jpg')
    print("Image processing complete. Result saved as 'result.jpg'.")


def process_video(sess, video_path, size):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('onnx_result.mp4', fourcc, fps, (orig_w, orig_h))

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize frame while preserving aspect ratio
        resized_frame_pil, ratio, pad_w, pad_h = resize_with_aspect_ratio(frame_pil, size)
        orig_size = torch.tensor([[resized_frame_pil.size[1], resized_frame_pil.size[0]]])

        transforms = T.Compose([
            T.ToTensor(),
        ])
        im_data = transforms(resized_frame_pil).unsqueeze(0)

        output = sess.run(
            output_names=None,
            input_feed=build_feed(sess, im_data, orig_size)
        )

        labels, boxes, scores = unpack_output(output)

        # Draw detections on the original frame
        result_images = draw(
            [frame_pil], labels, boxes, scores,
            [ratio], [(pad_w, pad_h)]
        )
        frame_with_detections = result_images[0]

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_with_detections), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'result.mp4'.")


def main(args):
    """Main function."""
    # Load the ONNX model
    sess = ort.InferenceSession(args.onnx)
    print(f"Using device: {ort.get_device()}")

    # 전처리 크기를 ONNX 입력 shape에서 읽는다 (하드코딩하지 않음)
    size = model_input_size(sess)
    print(f'inference input size: {size}x{size} (from ONNX input shape)')

    input_path = args.input

    try:
        # Try to open the input as an image
        im_pil = Image.open(input_path).convert('RGB')
        process_image(sess, im_pil, size)
    except IOError:
        # Not an image, process as video
        process_video(sess, input_path, size)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True, help='Path to the ONNX model file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image or video file.')
    args = parser.parse_args()
    main(args)

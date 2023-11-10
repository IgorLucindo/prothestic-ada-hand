import numpy as np
#import time

#import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())
sys.path.append(os.environ['HOME'] + "/Desktop/carlosIgor/computer vision/rotationDetection/yolo_obb_sim/src/yolov5_obb/scripts")

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_obb, print_args, scale_coords, scale_polys, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly

weights=os.environ['HOME'] + "/Desktop/carlosIgor/computer vision/rotationDetection/yolo_obb_sim/src/yolov5_obb/scripts/best.pt"
imgsz=[640, 480]  # inference size (pixels)
conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
save_conf=False,  # save confidences in --save-txt labels
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
augment=False  # augmented inference
visualize=False  # visualize features
line_thickness=3  # bounding box thickness (pixels)
hide_labels=False  # hide labels
hide_conf=False  # hide confidences
half=False  # use FP16 half-precision inference
stride = 32
device_num=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
dnn = False

# Initialize
device = select_device(device_num)

# Load model
model = DetectMultiBackend(weights, device=device, dnn=dnn)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size(imgsz, s=stride)  # check image size

# Half
half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
if pt or jit:
    model.model.half() if half else model.model.float()

# Run inference
model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup

cap = cv2.VideoCapture('cars.mp4')

def loop():
    while True:
        _, img = cap.read()
        # Letterbox
        img0 = img.copy()
        img = img[np.newaxis, :, :, :]        

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        pred = model(img, augment=augment, visualize=visualize)

        # Apply NMS
        pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True, max_det=max_det)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])

            s = f'{i}: '
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(img0, line_width=line_thickness, example=str(names))
            if len(det):

                # Rescale polys from img_size to im0 size
                pred_poly = scale_polys(img.shape[2:], pred_poly, img0.shape)
                det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])

                # Print results
                #for c in det[:, -1].unique():
                    #n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                for *poly, conf, cls in reversed(det):
                    #line = (cls, *poly, conf) if save_conf else (cls, *poly)  # label format
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.poly_label(poly, label, color=colors(c, True))

        cv2.imshow("IMAGE", img0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    loop()
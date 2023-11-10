import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from torchvision.models.resnet import resnet50
from torchvision.models.mobilenet import mobilenet_v2
import torchvision.transforms as T
from PIL import Image
import json
import time
import multiprocessing

# Create a shared variable that can be accessed by both processes
# shared_var = multiprocessing.Array('c', 100) 
# model = resnet50(pretrained=True).eval()
model = mobilenet_v2(pretrained=True).eval()
# model = torch.hub.load("ultralytics/yolov5", "yolov5s")

total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params}")


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

confidence_threshold=0.5

with open('/home/bioinlab/Desktop/carlosIgor/computer vision/imagenet_class_index.json') as f:
    raw_class = json.load(f)
with open('/home/bioinlab/Desktop/carlosIgor/computer vision/imagenet_subclass.json') as f:
    new_class = json.load(f)
# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def which_class(raw_class, new_class, idxs, scores):
    idxs = idxs.reshape(-1)
    scores = scores.reshape(-1)
    
    for idx, score in zip(idxs, scores):
        # if score.item() >= confidence_threshold:
        if str(idx.item()) in list(new_class.keys()):
            return new_class[str(idx.item())][1], str(score.item())[:5]

    return 'nothing', 0




window_size = 30
stride = 10
detection_counts = {}
grasp_type="None"


grasp_mapping = {
    "fountain_pen": 'Pinch',
    "ballpoint": 'Pinch',
    "beer_bottle": 'Cylinder',
    "water_bottle":'Cylinder',
    "remote_control":'Point',
    "wine_bottle":'Cylinder',
    "mouse":'Mouse Grip',
    "cup":'Power',
    "coffee_mug":'Cylinder',
    "Granny_Smith":'Power',
    "banana":'Power'
}





try:

    totalframe_count = 0
    start_time = time.time()
    frame_count = 0
    object_positions = {}
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        color_image2 = Image.fromarray(color_image)
        image = transform(color_image2)
        out = model(torch.Tensor(image).unsqueeze(0))
        out = torch.nn.functional.softmax(out).reshape(-1)
        score, predicted_idx = torch.topk(out, 5)
        name_final, score_final = which_class(raw_class, new_class, predicted_idx, score)
        if score_final == 0:
            score_final = ''

        # Increment frame count
        frame_count += 1
        totalframe_count += 1
        # Update detection counts and positions for the current object
        if name_final != 'nothing':
            if name_final in detection_counts:
                detection_counts[name_final] += 1
                object_positions[name_final].append(frame_count)
            else:
                detection_counts[name_final] = 1
                object_positions[name_final] = [frame_count]

        # Check if the window is complete
        if frame_count >= window_size:
            print (grasp_type)
            # Process the detection counts and send grasp types
            for obj, count in detection_counts.items():
                # print("object:", obj)
                # print("count:", count)
                if count >= 25:
                    # Send grasp type related to the object 'obj' to the hand
                    # print(f"Sending grasp type for object '{obj}' to the hand")
                    grasp_type=grasp_mapping[obj]
                    # print (grasp_type)

                    # Update the shared variable
                    # shared_var.value = b"grasp_type"

            # Slide the window by 'stride' frames
            for obj, positions in object_positions.items():
                positions_to_remove = [p for p in positions if p <= stride]
                count_removed = len(positions_to_remove)
                for p in positions_to_remove:
                    positions.remove(p)
                for i in range(len(positions)):
                    positions[i] -= stride
                detection_counts[obj] -= count_removed

            frame_count -= stride

            for obj in list(detection_counts.keys()):
                if obj not in object_positions:
                    del detection_counts[obj]
                # if len(positions) < stride:
                #     del detection_counts[obj]
                #     del object_positions[obj]




        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        text = str(score_final) + ' ' + name_final
        text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness=2)[0]
        text_offset_x = 50
        text_offset_y = 50
        cv2.putText(color_image, text, (text_offset_x, text_offset_y), font, font_scale,(0,255,0), thickness=2)
        cv2.putText(color_image, f'Grasp Type: {grasp_type}', (50, 100), font, font_scale, (0, 0, 255), thickness=2)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()


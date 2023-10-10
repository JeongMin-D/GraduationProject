---
title: "YOLOv5와 Unreal Engine4를 활용한 Sensorless AMR 충돌방지"
date: 2023-10-09
categories: ROS Robot operation system YOLOv5 camera unrealengine4 sensorless tcp amr turtlebot collision detect
---

# MySQL Server

## Database configuration

![sql_one_1](https://github.com/JeongMin-D/GraduationProject/assets/38097923/86efe777-adf1-41bd-a535-553de7f6f38b)

- 각 x, y, theta의 값을 DOUBLE로 설정하여 데이터를 공유할 수 있도록 설정

![sql_one](https://github.com/JeongMin-D/GraduationProject/assets/38097923/c644c501-445d-40e9-ac40-ceb51e545bec)

- one 테이블: Turtlebot3의 Odometry를 실시간으로 업로드하여 실시간 터틀봇 위치 공유
- 통신 속도를 위해 지속적으로 insert 하는 것이 아니라 upload하는 방식 사용

![sql_two_1](https://github.com/JeongMin-D/GraduationProject/assets/38097923/736afb53-d019-46e4-a6a5-d151148f11e3)

- YOLOv5에서 객체를 찾은 정보 중 ID는 INT 형, Label은 Varchar형, Confidence는 Float형, x y w h는 Double형, timestamp는 Time 형으로 지정

![sql_two](https://github.com/JeongMin-D/GraduationProject/assets/38097923/2129bc16-8bff-48ee-a47a-0e0c5daec9a0)

- two 테이블: YOLOv5에서 객체를 찾은 정보를 실시간으로 업로드하여 객체 정보와 위치 공유
- 통신 속도를 위해 실시간으로 객체를 탐지하면 insert가 아닌 update로 업로드하고 5초간 새로운 업데이트가 되지 않으면 삭제

![sql_three_1](https://github.com/JeongMin-D/GraduationProject/assets/38097923/045112e1-6167-4eba-afed-b7829fec1a6b)

- stop 데이터를 INT형으로 설정

![sql_three](https://github.com/JeongMin-D/GraduationProject/assets/38097923/ba1c44ff-d1b5-492c-b95a-fa2d095fbab8)

- three 테이블: 충돌 신호에 따라 stop 값이 True or False와 같이 1과 0으로 업데이트되어 신호

# YOLOv5

## Custom data training
[Roboflow](https://universe.roboflow.com/)

- Roboflow의 박스 데이터를 다운 받아서 박스 데이터 학습 진행

![custom_data](https://github.com/JeongMin-D/GraduationProject/assets/38097923/08c949a0-d5b7-4aaa-9b47-64aaba59bc6f)

- 같은 공간 내에 모든 파일을 위치시켜야 학습 진행 원활

```commandline
python train.py --img 416 --batch 16 --epochs 50 --data ./package/data.yaml --cfg ./models/yolov5s.yaml --weights yolov5s.pt --name box_yolov5s_results
```

- 각 값들은 사용자 지정에 따라 설정하고 --data 값은 해당 경로 내의 yaml 파일로 설정

## Training file validation

```commandline
python val.py --data 데이터 학습할 때의 yaml 파일.yaml --weights 학습 후 생성된 best.pt 파일 best.pt --img 640 --batch-size 32
```

- 학습할 때 입력하였던 yaml 파일과 학습 후 생성된 best.pt 파일을 사용하여 트레이닝 데이터 확인

![PR_curve](https://github.com/JeongMin-D/GraduationProject/assets/38097923/be0a6f19-41cc-4fb9-8389-0fbd39c0bb45)

![val_batch2_pred](https://github.com/JeongMin-D/GraduationProject/assets/38097923/7192db4f-fe56-49f9-b2bb-eb4be5b31d4b)

- validation 진행 후 다음과 같은 그래프 및 파일이 생성되고, 이때 IoU가 0.5일 때, mAP가 98.3%가 나오는 것을 확인할 수 있음

## Object detection, Location estimation and Upload data to MySQL

```python
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

import pymysql

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    frame_count = 0  # 프레임 카운트 변수 추가
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # 커넥션 초기화
        def initialize_mysql_connection():
            conn = pymysql.connect(
                host='192.168.1.155',
                user='turtlebot',
                password='0000',
                db='test',
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            return conn

        # SQL 내 객체 존재 시 업데이트, 없을 시 추가
        def insert_data_into_mysql(object_id, label, confidence, x, y, w, h):
            conn = initialize_mysql_connection()
            try:
                with conn.cursor() as cursor:
                    sql_check = "SELECT * FROM two WHERE object_id=%s AND label=%s"
                    cursor.execute(sql_check, (object_id, label))
                    existing_record = cursor.fetchone()

                    if existing_record:
                        sql_update = "UPDATE two SET confidence=%s, x_cm=%s, y_cm=%s, w=%s, h=%s, timestamp=NOW() WHERE object_id=%s AND label=%s"
                        cursor.execute(sql_update, (confidence, x, y, w, h, object_id, label))
                    else:
                        sql_insert = "INSERT INTO two (object_id, label, confidence, x_cm, y_cm, w, h, timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())"
                        cursor.execute(sql_insert, (object_id, label, confidence, x, y, w, h))

                conn.commit()
            finally:
                conn.close()

        # timestamp의 시간이 5초가 흐르면 데이터 삭제
        def delete_inactive_objects():
            conn = initialize_mysql_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT NOW() - INTERVAL 5 SECOND as threshold_time")
                    threshold_time = cursor.fetchone()['threshold_time']

                    sql_delete_inactive = "DELETE FROM two WHERE timestamp < %s"
                    cursor.execute(sql_delete_inactive, (threshold_time,))

                conn.commit()
            finally:
                conn.close()

        # Initialize dictionary to store object IDs
        object_ids = {}

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            frame_count += 1
            if frame_count % 5 == 0:
                delete_inactive_objects()
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    # Get object ID (or assign a new one)
                    if c not in object_ids:
                        object_ids[c] = 1
                    else:
                        object_ids[c] += 1
                    object_id = object_ids[c]

                    # Print coordinates
                    x, y, w, h = map(int, xyxy)  # Convert to integers
                    x_cm = x * 0.0264583333
                    y_cm = y * 0.0264583333
                    w_cm = w * 0.0264583333
                    h_cm = h * 0.0264583333
                    print(
                        f'Object ID: {object_id}, Label: {label}, Confidence: {confidence_str}, Coordinates: x={x_cm}, y={y_cm}, w={w_cm}, h={h_cm}')

                    insert_data_into_mysql(object_id, label, confidence, x_cm, y_cm, w_cm, h_cm)

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}{object_id} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1000], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
```

```commandline
python test4.py --weights best.pt --source 0 --device 0 --conf 0.6 --line-thickness 2
```
- 커스텀 데이터를 학습한 best.pt와 웹캠, GPU를 사용하고 confidence 0.6이상만 검출

### 동일 객체에 대한 ID 부여

```python
# Initialize dictionary to store object IDs
        object_ids = {}

# Get object ID (or assign a new one)
                    if c not in object_ids:
                        object_ids[c] = 1
                    else:
                        object_ids[c] += 1
                    object_id = object_ids[c]
```

- 탐지 객체가 object_ids(딕셔너리) 안에 없을 때 1번 ID를 부여
- 이미 object_ids(딕셔너리) 안에 존재할 경우 ID에 1씩 더하며 순차적으로 ID 부여

### 픽셀을 사용한 위치 추정

```python
# Print coordinates
                    x, y, w, h = map(int, xyxy)  # Convert to integers
                    x_cm = x * 0.0264583333
                    y_cm = y * 0.0264583333
                    w_cm = w * 0.0264583333
                    h_cm = h * 0.0264583333
```

- DPI: 픽셀 밀도 혹은 인치당 도트 수
- 96 DPI = 96PX/1INCH
- 1 INCH = 2.54CM
- 96PX = 2.54CM
- 1PX = 2.54CM / 96 = 0.026458333CM
- 왼쪽 상단을 기준으로 x는 왼쪽에서 오른쪽, y는 위에서 아래로 증가시켜 픽셀 값을 생성하고 cm로 변환
- w는 감지된 객체의 너비를 픽셀로 나타내고 h는 높이를 나타냄

### MySQL에 탐지 객체 업로드 및 삭제

```python
        def initialize_mysql_connection():
            conn = pymysql.connect(
                host='192.168.1.155',
                user='turtlebot',
                password='0000',
                db='test',
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            return conn

        # SQL 내 객체 존재 시 업데이트, 없을 시 추가
        def insert_data_into_mysql(object_id, label, confidence, x, y, w, h):
            conn = initialize_mysql_connection()
            try:
                with conn.cursor() as cursor:
                    sql_check = "SELECT * FROM two WHERE object_id=%s AND label=%s"
                    cursor.execute(sql_check, (object_id, label))
                    existing_record = cursor.fetchone()

                    if existing_record:
                        sql_update = "UPDATE two SET confidence=%s, x=%s, y=%s, w=%s, h=%s, timestamp=NOW() WHERE object_id=%s AND label=%s"
                        cursor.execute(sql_update, (confidence, x, y, w, h, object_id, label))
                    else:
                        sql_insert = "INSERT INTO two (object_id, label, confidence, x, y, w, h, timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())"
                        cursor.execute(sql_insert, (object_id, label, confidence, x, y, w, h))

                conn.commit()
            finally:
                conn.close()

        # timestamp의 시간이 5초가 흐르면 데이터 삭제
        def delete_inactive_objects():
            conn = initialize_mysql_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT NOW() - INTERVAL 5 SECOND as threshold_time")
                    threshold_time = cursor.fetchone()['threshold_time']

                    sql_delete_inactive = "DELETE FROM two WHERE timestamp < %s"
                    cursor.execute(sql_delete_inactive, (threshold_time,))

                conn.commit()
            finally:
                conn.close()
            frame_count += 1
            if frame_count % 5 == 0:
                delete_inactive_objects()
                insert_data_into_mysql(object_id, label, confidence, x_cm, y_cm, w_cm, h_cm)
```

- initialize_mysql_connection(): pymysql을 사용하여 host, user, password, db 등 통신에 필요한 기본 정보 설정
- insert_data_into_mysql(object_id, label, confidence, x, y, w, h): 객체 탐지 중 생성한 ID, 객체 라벨, conf, 좌표 및 너비, 높이 등을 Mysql 데이터베이스에 업로드
- delete_inactive_objects(): 업로드 된 객체 중 5초간 업데이트 되지 않은 객체는 삭제

# AMR

## Custom DWA Algorithm

```python
#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import random
from tf.transformations import euler_from_quaternion
import pymysql

class TurtleBot:
    def __init__(self):
        rospy.init_node('turtlebot_controller', anonymous=False)
        rospy.on_shutdown(self.shutdown)

        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback)

        self.rate = rospy.Rate(10)  # 10 Hz
        self.target_coordinates = []  # List to store target coordinates
        self.current_target_index = 0  # Index of the current target coordinate

        self.max_linear_velocity = 0.2
        self.max_angular_velocity = 1.0
        self.max_linear_acceleration = 0.1
        self.max_angular_acceleration = 1.0

        self.scan_ranges = None
        self.scan_angle_min = None
        self.scan_angle_increment = None

        self.current_pose = None  # Store current pose

        # Connect to the database
        self.conn = pymysql.connect(host='192.168.1.155', user='turtlebot', password='0000', database='test')
        self.cursor = self.conn.cursor()

    def insert_coordinates(self, x, y, theta):
        try:
            # Connect to the database
            conn = pymysql.connect(host='192.168.1.155', user='turtlebot', password='0000', database='test')
            cursor = conn.cursor()

            query = "UPDATE one SET x = %s, y = %s, theta = %s WHERE idone = 1"
            self.cursor.execute(query, (x, y, theta))
            self.conn.commit()

        except pymysql.Error as e:
            print(f"Error in update_position: {e}")

    def check_stop(self):
        try:
            # Retrieve the collision value from the three table
            query = "SELECT stop FROM three order by idthree desc limit 1"
            self.cursor.execute(query)
            result = self.cursor.fetchone()

            if result is None:
                return 0
            else:
                return result[0]

        except pymysql.Error as e:
            print(f"Error in check_collision: {e}")
            return 0

    def odom_callback(self, data):
        self.current_pose = data.pose.pose  # Update current pose

        # Check if there are any target coordinates
        if len(self.target_coordinates) == 0:
            return

        # Calculate the Euclidean distance between the current position and the current target position
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y

        # Find current yaw
        # current_orientation = (
        #     self.current_pose.orientation.x,
        #     self.current_pose.orientation.y,
        #     self.current_pose.orientation.z,
        #     self.current_pose.orientation.w
        # )
        # _, _, current_yaw = euler_from_quaternion(current_orientation)

        distance_to_target = math.sqrt((self.target_coordinates[self.current_target_index][0] - current_x) ** 2 +
                                       (self.target_coordinates[self.current_target_index][1] - current_y) ** 2)

        # Check if the TurtleBot has reached the current target position
        if distance_to_target <= 0.1:
            rospy.loginfo("Target reached: {}".format(self.target_coordinates[self.current_target_index]))
            self.current_target_index += 1  # Move to the next target coordinate

            if self.current_target_index >= len(self.target_coordinates):
                self.cmd_vel.publish(Twist())  # Stop the TurtleBot
                rospy.loginfo("All target positions reached!")
                rospy.signal_shutdown("All target positions reached!")  # Shutdown the node

                # Insert current coordinates into the database
                # self.insert_coordinates(current_x, current_y, current_yaw)

    def move_to_target(self, target_coordinates):
        self.target_coordinates = target_coordinates

        for i in range(len(self.target_coordinates)):
            target_x, target_y = self.target_coordinates[i]
            rospy.loginfo("Moving to target: ({}, {})".format(target_x, target_y))

            while not rospy.is_shutdown():
                # Get the current target coordinates
                target_x, target_y = self.target_coordinates[i]

                # Calculate the distance to the current target
                current_x = self.current_pose.position.x
                current_y = self.current_pose.position.y
                distance_to_target = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)

                # Calculate the angle to the target
                target_angle = math.atan2(target_y - current_y, target_x - current_x)

                # Calculate the angular velocity to align with the target angle
                current_orientation = (
                    self.current_pose.orientation.x,
                    self.current_pose.orientation.y,
                    self.current_pose.orientation.z,
                    self.current_pose.orientation.w
                )
                _, _, current_yaw = euler_from_quaternion(current_orientation)
                angle_difference = target_angle - current_yaw

                # Insert current coordinates into the database
                self.insert_coordinates(current_x, current_y, current_yaw)

                # Normalize the angle difference to the range [-pi, pi]
                angle_difference = math.atan2(math.sin(angle_difference), math.cos(angle_difference))

                # Calculate the linear velocity based on the distance to the target
                linear_velocity = self.max_linear_velocity * distance_to_target

                # Limit the linear velocity within the maximum limits
                linear_velocity = max(0.0, min(self.max_linear_velocity, linear_velocity))

                # Calculate the angular velocity to align with the target angle
                angular_velocity = self.max_angular_velocity * angle_difference

                # Limit the angular velocity within the maximum limits
                angular_velocity = max(-self.max_angular_velocity, min(self.max_angular_velocity, angular_velocity))

                # Publish the velocities to move towards the target while aligning with the target angle
                twist_cmd = Twist()
                twist_cmd.linear.x = linear_velocity
                twist_cmd.angular.z = angular_velocity
                self.cmd_vel.publish(twist_cmd)

                # If the distance to the target is less than a threshold, stop and break the loop
                if distance_to_target <= 0.1:
                    rospy.loginfo("Target reached: ({}, {})".format(target_x, target_y))
                    twist_cmd = Twist()
                    twist_cmd.angular.z = 0.0
                    twist_cmd.linear.x = 0.0
                    self.cmd_vel.publish(twist_cmd)
                    break

                # Check for stop signal from the database
                if self.check_stop() == 1:
                    twist_cmd = Twist()
                    twist_cmd.angular.z = 0.0
                    twist_cmd.linear.x = 0.0
                    self.cmd_vel.publish(twist_cmd)
                    rospy.loginfo("Stopped by external signal.")
                    return

                self.rate.sleep()

            # Stop the TurtleBot before moving to the next target
            twist_cmd = Twist()
            twist_cmd.angular.z = 0.0
            twist_cmd.linear.x = 0.0
            self.cmd_vel.publish(twist_cmd)
            self.rate.sleep()

        rospy.loginfo("All target positions reached!")
        rospy.signal_shutdown("All target positions reached!")  # Shutdown the node

    def calculate_dwa(self):
        if self.scan_ranges is None or self.scan_angle_min is None or self.scan_angle_increment is None:
            return 0.0, 0.0

        # DWA algorithm implementation
        min_cost = float('inf')
        best_linear_velocity = 0.0
        best_angular_velocity = 0.0

        for linear_velocity in self.generate_linear_velocities():
            for angular_velocity in self.generate_angular_velocities():
                # Simulate the robot's trajectory to calculate the cost of the trajectory
                simulated_trajectory_cost = self.simulate_trajectory(linear_velocity, angular_velocity)

                # Choose the trajectory with the lowest cost
                if simulated_trajectory_cost < min_cost:
                    min_cost = simulated_trajectory_cost
                    best_linear_velocity = linear_velocity
                    best_angular_velocity = angular_velocity

        return best_linear_velocity, best_angular_velocity

    def simulate_trajectory(self, linear_velocity, angular_velocity):
        # Simulate the robot's trajectory using the provided linear and angular velocities
        # and calculate the cost of the trajectory

        # Define some simulation parameters
        num_steps = 50
        dt = 0.1
        trajectory_cost = 0.0

        # Simulate the trajectory and accumulate cost
        for step in range(num_steps):
            # Simulate the robot's motion using the given velocities
            linear_distance = linear_velocity * dt
            angular_distance = angular_velocity * dt

            # Update the robot's position and orientation
            current_x = self.current_pose.position.x
            current_y = self.current_pose.position.y
            current_orientation = (
                self.current_pose.orientation.x,
                self.current_pose.orientation.y,
                self.current_pose.orientation.z,
                self.current_pose.orientation.w
            )
            _, _, current_yaw = euler_from_quaternion(current_orientation)

            new_x = current_x + linear_distance * math.cos(current_yaw)
            new_y = current_y + linear_distance * math.sin(current_yaw)
            new_yaw = current_yaw + angular_distance

            # Calculate the cost based on the distance to the target and collision risk
            distance_to_target = math.sqrt((self.target_coordinates[self.current_target_index][0] - new_x) ** 2 +
                                           (self.target_coordinates[self.current_target_index][1] - new_y) ** 2)
            collision_risk = self.calculate_collision_risk(new_x, new_y, new_yaw)

            # Update the trajectory cost with a combination of distance to target and collision risk
            trajectory_cost += distance_to_target + collision_risk

        return trajectory_cost

    def calculate_collision_risk(self, x, y, yaw):
        # Calculate the collision risk for the given pose (x, y, yaw) using laser scan data

        # Define some collision risk parameters
        collision_threshold = 0.3
        half_scan_range = len(self.scan_ranges) // 2

        # Get the index of the scan range closest to the robot's forward direction
        robot_heading_index = half_scan_range

        # Calculate the collision risk by checking scan ranges in the robot's heading direction
        collision_risk = 0.0
        for i in range(half_scan_range - 45, half_scan_range + 45):
            scan_range = self.scan_ranges[i]

            # Convert the scan index to the corresponding angle
            angle = self.scan_angle_min + self.scan_angle_increment * i

            # Calculate the position of the obstacle in the scan range
            obstacle_x = x + scan_range * math.cos(yaw + angle)
            obstacle_y = y + scan_range * math.sin(yaw + angle)

            # Calculate the distance to the obstacle
            distance_to_obstacle = math.sqrt((obstacle_x - x) ** 2 + (obstacle_y - y) ** 2)

            # Increase collision risk if obstacle is close
            if distance_to_obstacle < collision_threshold:
                collision_risk += 1.0

        return collision_risk

    def generate_linear_velocities(self):
        # Generate a range of linear velocities
        num_steps = 5
        step_size = self.max_linear_velocity / num_steps
        velocities = [i * step_size for i in range(num_steps + 1)]
        return velocities

    def generate_angular_velocities(self):
        # Generate a range of angular velocities
        num_steps = 10
        step_size = self.max_angular_velocity / num_steps
        velocities = [-self.max_angular_velocity + i * step_size for i in range(num_steps + 1)]
        return velocities

    def shutdown(self):
        rospy.loginfo("Stopping the TurtleBot...")
        self.cmd_vel.publish(Twist())  # Stop the TurtleBot
        rospy.sleep(1)
        self.conn.close()  # Close the database connection

if __name__ == '__main__':
    try:
        turtlebot = TurtleBot()
        target_coordinates = []
        while True:
            target_input = input("Enter target coordinates (x,y) or 'done' to finish: ")
            if target_input.lower() == 'done':
                break
            else:
                x, y = target_input.split(",")
                target_coordinates.append((float(x), float(y)))
        turtlebot.move_to_target(target_coordinates)
    except rospy.ROSInterruptException:
        pass
```

### Upload Odometry to MySQL

```python
    def insert_coordinates(self, x, y, theta):
        try:
            # Connect to the database
            conn = pymysql.connect(host='192.168.1.155', user='turtlebot', password='0000', database='test')
            cursor = conn.cursor()

            query = "UPDATE one SET x = %s, y = %s, theta = %s WHERE idone = 1"
            self.cursor.execute(query, (x, y, theta))
            self.conn.commit()

        except pymysql.Error as e:
            print(f"Error in update_position: {e}")
    def move_to_target(self, target_coordinates):
        self.target_coordinates = target_coordinates

        for i in range(len(self.target_coordinates)):
            target_x, target_y = self.target_coordinates[i]
            rospy.loginfo("Moving to target: ({}, {})".format(target_x, target_y))

            while not rospy.is_shutdown():
                # Get the current target coordinates
                target_x, target_y = self.target_coordinates[i]

                # Calculate the distance to the current target
                current_x = self.current_pose.position.x
                current_y = self.current_pose.position.y
                distance_to_target = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)

                # Calculate the angle to the target
                target_angle = math.atan2(target_y - current_y, target_x - current_x)

                # Calculate the angular velocity to align with the target angle
                current_orientation = (
                    self.current_pose.orientation.x,
                    self.current_pose.orientation.y,
                    self.current_pose.orientation.z,
                    self.current_pose.orientation.w
                )
                _, _, current_yaw = euler_from_quaternion(current_orientation)
                angle_difference = target_angle - current_yaw

                # Insert current coordinates into the database
                self.insert_coordinates(current_x, current_y, current_yaw)
```

- pymysql을 사용하여 테이블 one에 실시간 x, y, theta 입력

### Stop sign detection and stopping

```python
    def check_stop(self):
        try:
            # Retrieve the collision value from the three table
            query = "SELECT stop FROM three order by idthree desc limit 1"
            self.cursor.execute(query)
            result = self.cursor.fetchone()

            if result is None:
                return 0
            else:
                return result[0]
                
    def move_to_target(self, target_coordinates):

                # Check for stop signal from the database
                if self.check_stop() == 1:
                    twist_cmd = Twist()
                    twist_cmd.angular.z = 0.0
                    twist_cmd.linear.x = 0.0
                    self.cmd_vel.publish(twist_cmd)
                    rospy.loginfo("Stopped by external signal.")
                    return
```

- 테이블 three의 stop 데이터를 지속적으로 읽고 데이터가 1이 되면 충돌을 감지하고 긴급 정지

# Unreal Engine 4

![ue4_overall](https://github.com/JeongMin-D/GraduationProject/assets/38097923/91f316f8-19c1-408e-a878-283c5af98b98)

- Level blueprint

## Import AMR data from MySQL

![plugin](https://github.com/JeongMin-D/GraduationProject/assets/38097923/343bbf5e-3553-4697-bf19-844183757338)

- MySQL과 연동가능한 플러그인 사용

![sql_open](https://github.com/JeongMin-D/GraduationProject/assets/38097923/a4447c37-4aa3-41c1-8475-50209b4f3615)

- MySQLDBConnection 액터 블루프린트 생성 후 사용할 MySQL Database와 연결 설정

![sql_data_input_output](https://github.com/JeongMin-D/GraduationProject/assets/38097923/6404f99d-bae4-4c55-8726-475b18ba5a29)

- Tick 이벤트로 실시간 two, one 테이블 데이터 Select 

![sql_data_input](https://github.com/JeongMin-D/GraduationProject/assets/38097923/29d933cd-c01c-4e2a-b637-deb3f08adaa2)

- 실시간으로 Select한 데이터를 Row별로 잘라서 ObstacleData 이벤트 디스패처로 전송
- 레벨 블루프린트에서 데이터를 사용하기 위함

![ue4_turtlebot](https://github.com/JeongMin-D/GraduationProject/assets/38097923/0f885660-a6ae-4f2e-b334-f0e65f42f213)

- 레벨 블루프린트에서 ObstacleData를 할당하고 x, y, theta를 위치로 데이터를 불러서 Turtlebot 액터에 데이터 전송하여 위치 이동

## Receive object information from YOLOv5 and spawn object

![sql_open](https://github.com/JeongMin-D/GraduationProject/assets/38097923/5bab6233-47f1-4c20-8a92-976b2f230d49)

- 위 사진과 똑같은 MySQLDBConnection 액터 블루프린트 사용

![sql_data_input_output](https://github.com/JeongMin-D/GraduationProject/assets/38097923/f3ad0c01-228b-47e6-856e-ac82a0646002)

- Sequence, 동일한 MySQLDBConnection 액터 생성을 해보았지만 오류로 인해 update 불가
- 한번에 2개의 쿼리를 실행해야하는 것이 문제인 것으로 생각하나 검증 불가
- 따라서 A와 B를 한번씩 번갈아가며 실행하는 Flip Flop으로 해결

![ue4_object](https://github.com/JeongMin-D/GraduationProject/assets/38097923/27fa9d2a-04b7-4618-bbc7-e8d2c150b1d6)

- 감지된 객체가 0이 아닌 경우 중 스폰되지 않은 객체는 스폰, 스폰된 경우는 이동되도록 코드 구성
- 5초간 객체가 감지되지 않아서 데이터베이스가 0이 되면 스폰된 객체는 삭제
- 객체의 x, y 위치에서 데이터를 뽑아 스폰 및 위치 이동에 사용

## Detect conflicting signals and upload to MySQL

![overlap_pic](https://github.com/JeongMin-D/GraduationProject/assets/38097923/9d5ea4a7-228d-4aa6-acf6-7ec6e2e2e620)

- 장애물 감지를 위해 Turtlebot 액터에 Box Collision 생성

![overlap](https://github.com/JeongMin-D/GraduationProject/assets/38097923/956a3841-1fa7-438b-8d2a-cbd40c5c4a2a)

- Box Collision에 다른 물체가 겹쳐지는 순간 장애물로 감지하고 Sendsignal에 True 값을 실어서 레벨 블루프린트로 전송

![ue4_collision_signal](https://github.com/JeongMin-D/GraduationProject/assets/38097923/5c92e0b4-85d6-4586-8891-9c784daf36aa)

- 레벨 블루프린트에서 Sendsignal을 할당하고 True 데이터를 받은 경우 변수에 stop, True를 받지 못한 경우 move로 저장

![sql_data_input_output](https://github.com/JeongMin-D/GraduationProject/assets/38097923/58c9d7f5-8885-4465-975a-1ede4c6d5fcc)

- 위와 같은 MySQLDBConnection에서 변수가 stop일 경우 데이터베이스 three 테이블 stop 값을 1, move일 경우 0으로 업데이트하여 충돌 신호 전송

# Result

![1](https://github.com/JeongMin-D/GraduationProject/assets/38097923/5d4d268c-0b57-4b67-8282-a5992de7a0ba)

![4](https://github.com/JeongMin-D/GraduationProject/assets/38097923/ab02c612-30da-4f9b-a1d9-d1c21fc50bff)

- 0.6 이상의 box를 YOLOv5로 감지하고, 그 데이터를 데이터베이스 two 테이블에 전송하여 업로드
- Turtlebot의 Odometry의 x, y, theta 값을 데이터베이스 one 테이블에 전송하여 업로드
- Unreal Engine 4에서 one과 two 테이블의 데이터를 종합하여 가상환경에서 구현
- 가상환경에 구현된 box와 Turtlebot이 이동 중 충돌하게 되면 three 테이블에 충돌 신호 전송
- Turtlebot 코드에서 three 테이블의 충돌신호를 select하다가 stop의 값이 1로 업데이트 되면 Turtlebot에 정지신호를 보내 정지
- Turtlebot이 정지하면 Odometry 정보가 업데이트 되지 않아 가상환경 내에서 같이 정지
- 따라서, Sensorless Turtlebot이 객체 탐지 및 위치 정보 기반으로 가상환경을 만들어 장애물을 감지하고 정지하는 기술 제작
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, xywhTheda2Points
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, non_max_suppression_obb
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import numpy as np
import math


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    t_inference_all = []
    t_nms_all = []
    cnt = 0

    for eval_clear_i in range(16):
        eval_clear = 'evaluation/' + str(eval_clear_i) + '.txt'
        with open(eval_clear, 'a') as f:
            f.truncate(0)
            f.close()

    for path, img, im0s, vid_cap in dataset:
        # print('path: ', path)
        cnt += 1
        if cnt % 20 == 0:
            print(cnt, ' / ', dataset.nf, (cnt/dataset.nf * 100), '%', '************************************************************************************************************************************')

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        half = False
        if half:
            img = img.half()
            model = model.half()

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS

        # Remove prediction boxes under extreme conditions
        pred = pred.view(-1, 22)
        h_limit = (pred[:, 3] < 8)
        aspect_ratio_limit = (pred[:, 2] / abs(pred[:, 3] + 1e-6) > 30)
        out_limit = (h_limit | aspect_ratio_limit)
        pred = pred[~out_limit]
        pred = pred.view(1, -1, 22)

        pred = non_max_suppression_obb(pred, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, labels=opt.classes, multi_label=False)
        t3 = time_synchronized()

        # Save output to txt
        eval_path = path.split("/")[-1].split(".")[0]

        for eval_out_i in range(pred[0].shape[0]):
            eval_conf = pred[0][eval_out_i, 5].cpu().numpy()

            eval_xywhTheta = pred[0][eval_out_i, :5].cpu().numpy()
            eval_xywhTheta[-1] = (eval_xywhTheta[-1] + math.pi / 2) / math.pi * 180
            eval_Points = xywhTheda2Points(eval_xywhTheta)

            eval_cls = pred[0][eval_out_i, 6].cpu().numpy()
            from decimal import Decimal
            eval_write = Decimal(str(eval_cls)).normalize()
            eval_write = int(eval_write)
            eval_txt = 'evaluation/' + str(eval_write) + '.txt'
            with open(eval_txt, 'a') as f:
                f.write(eval_path + " ")
                f.write(str(eval_conf) + " ")
                for eval_Points_i in range(eval_Points.shape[0]):
                    f.write(str(eval_Points[eval_Points_i][0]) + " ")
                    f.write(str(eval_Points[eval_Points_i][1]) + " ")
                f.write('\n')
                f.close()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 6].unique():
                    n = (det[:, 6] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for x, y, w, h, angle, conf, cls in reversed(det):
                    xywh = np.array([x.cpu(), y.cpu(), w.cpu(), h.cpu()])
                    if save_txt:  # Write to file
                        x, y, w, h = x/gn[0], y/gn[1], w/gn[0], h/gn[1]  # 0~1
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 7 + '\n') % (conf, cls, x, y, w, h, int((math.pi/2 + angle) / math.pi * 180)))  # label format

                    if save_img or view_img:  # Add bbox to image
                        angle = int((math.pi/2 + angle) / math.pi * 180)
                        label = '%s %.2f %s' % (names[int(cls)], conf, angle)
                        xyxy = xywh2xyxy(xywh)
                        plot_one_box(xyxy, angle, im0, label=label, color=colors[int(cls)])

            # Print time (inference + NMS)
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            t_inference_all.append((t2 - t1)*1000)
            t_nms_all.append((t3 - t2) * 1000)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    print('inference average time: ', sum(t_inference_all)/len(t_inference_all),
          'nms average time: ', sum(t_nms_all)/len(t_nms_all))
    if save_txt or save_img:
        print('Results saved to %s' % Path(save_dir))

    print(f'Done. ({time.time() - t0:.3f}s)')


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = x[0] - x[2] / 2  # top left x
    y[1] = x[1] - x[3] / 2  # top left y
    y[2] = x[0] + x[2] / 2  # bottom right x
    y[3] = x[1] + x[3] / 2  # bottom right y
    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', default='weights/*.pt')
    parser.add_argument('--source', type=str, default='split/DOTA_test/', help='source')
    parser.add_argument('--project', default='evaluation/predictions', help='save results to project/name')
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

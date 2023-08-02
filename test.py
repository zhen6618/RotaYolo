import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm
import math
from copy import deepcopy

from models.experimental import attempt_load
from utils.datasets import create_dataloader, xywhTheda2Points
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, non_max_suppression_obb, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, \
    increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel
from EigenTheda import Eigen2xywhTheda, xywhTheda2Eigen, x2Eigen, Eigen2xywhTheda_numpy
from decimal import Decimal
from mmcv.ops import diff_iou_rotated_2d
import copy


def compute_iou_test(tcx, pcx, scale):
    """
    input: tcx:(m, 5)  pcx:(n, 5)
    output: loss_iou:(m, n)
    """
    m, n = tcx.shape[0], pcx.shape[0]
    device = pcx.device
    loss_iou = torch.zeros([tcx.shape[0], pcx.shape[0]])

    t = copy.deepcopy(tcx).unsqueeze(1).repeat(1, n, 1)  # (m, n, 5)
    p = copy.deepcopy(pcx).unsqueeze(0).repeat(m, 1, 1)  # (m, n, 5)

    p[..., 4] = p[..., 4] / 180 * math.pi
    t[..., 4] = t[..., 4] / 180 * math.pi
    p[..., :4] = p[..., :4] / scale
    t[..., :4] = t[..., :4] / scale
    p = p.to(dtype=torch.float32)
    t = t.to(dtype=torch.float32)

    iou_mmcv_multi = torch.zeros((0, n)).to(device)
    for j in range(m):
        t_j, p_j = t[j].unsqueeze(0), p[j].unsqueeze(0)

        iou_mmcv_j = diff_iou_rotated_2d(t_j, p_j)  # x y w h angle(rad)
        iou_mmcv_multi = torch.cat([iou_mmcv_multi, iou_mmcv_j], dim=0)

    iou_mmcv = iou_mmcv_multi

    return iou_mmcv

def test(data,
         weights=None,
         batch_size=4,
         imgsz=1024,
         conf_thres=0.3,
         iou_thres=0.1,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         half_precision=False,
         trace=False,
         is_coco=False,
         v5_metric=False):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size
        
        if trace:
            model = TracedModel(model, device, imgsz)

    # Configure
    model.eval()
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []

    'test'
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):

        loss_targets = copy.deepcopy(targets).to(device)

        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        "bs_index, cls, cx, cy(0-1), x1, x2, x3 to bs_index, cls, x, y, w, h, angle[0, 180)"
        uu = deepcopy(targets[:, 2:7]).detach().cpu().numpy()
        uu = x2Eigen(uu)
        uu, _ = Eigen2xywhTheda_numpy(uu)
        targets[:, 2:7] = torch.from_numpy(uu)
        targets = targets.to(device)

        nb, _, height, width = img.shape  # batch size, channels, height, width

        # if half:
        #     img = img.half()
        #     model = model.half()

        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = model(img, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # Compute loss
            if compute_loss:
                loss += compute_loss([x.float() for x in deepcopy(train_out)], deepcopy(loss_targets), img)[1][:3]  # box, obj, cls

            # Run NMS
            "after NMS: x y w h angle[-pi/2, pi/2) conf cls"
            targets[:, 2:6] *= torch.Tensor([width, height, width, height]).to(device)
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling

            # Remove prediction boxes under extreme conditions
            out = out.view(-1, 22)
            off = (out[:, 0] >= img.shape[2]) | (out[:, 0] <= 0) | (out[:, 1] >= img.shape[2]) | (out[:, 1] <= 0)
            h_limit = (out[:, 3] < 4)
            w_limit = (out[:, 2] > 800)
            aspect_ratio_limit = (out[:, 2] / abs(out[:, 3] + 1e-6) > 30)
            out_limit = (w_limit | h_limit | aspect_ratio_limit | off)
            out = out[~out_limit]
            out = out.view(1, -1, 22)

            out = non_max_suppression_obb(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=False)
            t1 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            # predn = pred.clone()
            # scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # # Append to text file
            # if save_txt:
            #     gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
            #     for *xyxy, conf, cls in predn.tolist():
            #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            #         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            #         with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
            #             f.write(('%g ' * len(line)).rstrip() % line + '\n')
            #
            # # W&B logging - Media Panel Plots
            # if len(wandb_images) < log_imgs and wandb_logger.current_epoch > 0:  # Check for test operation
            #     if wandb_logger.current_epoch % wandb_logger.bbox_interval == 0:
            #         box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
            #                      "class_id": int(cls),
            #                      "box_caption": "%s %.3f" % (names[cls], conf),
            #                      "scores": {"class_score": conf},
            #                      "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
            #         boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
            #         wandb_images.append(wandb_logger.wandb.Image(img[si], boxes=boxes, caption=path.name))
            # wandb_logger.log_training_progress(predn, path, names) if wandb_logger and wandb_logger.wandb_run else None
            #
            # # Append to pycocotools JSON dictionary
            # if save_json:
            #     # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
            #     image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            #     box = xyxy2xywh(predn[:, :4])  # xywh
            #     box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            #     for p, b in zip(pred.tolist(), box.tolist()):
            #         jdict.append({'image_id': image_id,
            #                       'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
            #                       'bbox': [round(x, 3) for x in b],
            #                       'score': round(p[4], 5)})

            'Assign all predictions as incorrect'
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                tbox = labels[:, 1:6]
                ppred = deepcopy(pred)
                ppred[:, 4] = (ppred[:, 4] + math.pi / 2) / math.pi * 180  # pred: (n, [xylsθ, conf, cls]) θ[0, 180)

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == ppred[:, 6]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        compute_ious = compute_iou_test(tbox[ti], ppred[pi, :5], height)
                        compute_ious = compute_ious.T
                        ious, i = compute_ious.max(1)  # best ious, indices
                        ious = ious.cpu()

                        # mAP
                        detected_set = set()
                        for j in (torch.Tensor(ious) > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv
                                if len(detected) == nl:
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 6].cpu(), tcls))

        # Plot images
        plots = True
        if plots and batch_i < 20:
            f = save_dir / f'test{batch_i}_labels.png'  # labels
            targets[:, 2:6] /= torch.Tensor([width, height, width, height]).to(device)
            "targets: [img_index, clsid cx cy l s(0-1) theta]) θ[0, 180)  [n, 7]"
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()

            f = save_dir / f'test{batch_i}_pred.png'  # predictions
            "out: x y w h angle(-pi/2~pi/2) conf cls  list{[m, 7]}"
            for out_i in range(len(out)):
                out[out_i][:, 0] = out[out_i][:, 0] / width
                out[out_i][:, 1] = out[out_i][:, 1] / height
                out[out_i][:, 2] = out[out_i][:, 2] / width
                out[out_i][:, 3] = out[out_i][:, 3] / height
                out[out_i][:, 4] = (out[out_i][:, 4] + math.pi/2) / math.pi * 180
            "out: x y w h(0-1) angle[0, 180) conf cls"

            "output_to_target： [img_index, class_id, x, y, w, h(0-1), conf, angle[0, 180)]"
            Thread(target=plot_images, args=(img, output_to_target(out, width, height), paths, f, names), daemon=True).start()
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=v5_metric, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='weights/*.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/dota.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=4, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric
             )

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights:
            test(opt.data, w, opt.batch_size, opt.img_size, 0.25, 0.45, save_json=False, plots=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov7.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, w, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False, v5_metric=opt.v5_metric)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot

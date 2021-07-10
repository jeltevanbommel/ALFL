import pickle
import gc
import argparse
import json
import os
import glob
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.loss import ComputeLoss
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized

batch_size = 64
imgsz = 640
conf_thres = 0.001
iou_thres = 0.6
gs = 32  # gridsize (max stride of model)


class opt: pass


opt.single_cls = False


def get_dataloader(nc):
    if nc == 2:
        return create_dataloader(f'KITTI/yolo2/val.txt', imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                 prefix=colorstr('val2: '))[0]
    else:
        return create_dataloader(f'KITTI/yolo2/val.txt', imgsz, batch_size, gs, opt, pad=0.5, rect=True,
                                 prefix=colorstr('val2: '))[0]


def evaluate(weights, model=None, save_dir=None, dataloader=None, nc=2, compute_loss=None, plots=True):
    from_scratch = model is None
    if from_scratch:
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        if w == 'best' or w == 'last':
            w = Path(weights[0] if isinstance(weights, list) else weights).parent.parent.stem.replace("run-26-", "run-15-")
        save_dir = increment_path(Path('runs/val/') / w, exist_ok=False)  # increment run
        set_logging()
        device = select_device('', batch_size=batch_size)
        # Half
        model = attempt_load(weights, map_location=device)  # load FP32 model
    else:
        device = next(model.parameters()).device  # get model device

    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make dir
    # PREP MODEL
    half = device.type != 'cpu'# half precision only supported on CUDA
    if half:
        model.half()
    # Configure
    model.eval()
    if from_scratch:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    # PREP COMPUTE LOSS
    compute_loss = ComputeLoss(model)

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        # Run model
        t = time_synchronized()
        out, train_out = model(img, augment=False)  # inference and training outputs
        t0 += time_synchronized() - t

        # Compute loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        out = non_max_suppression(out, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=False)

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
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            box = xyxy2xywh(predn[:, :4])  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for p, b in zip(pred.tolist(), box.tolist()):
                jdict.append({'image_id': image_id,
                              'category_id': int(p[5]),
                              'bbox': [round(x, 3) for x in b],
                              'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
    a = {'p': p,
         'r': r,
         'ap': ap,
         'f1': f1,
         'ap_class': ap_class,
         'ap50': ap50,
         'mp': mp,
         'mr': mr,
         'map50': map50,
         'map': map,
         'nt': nt,
         'names': names,
         'res_per_class': [(names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]) for i, c in enumerate(ap_class)],
         'conf_matrix': confusion_matrix.matrix,
         'losses': (loss.cpu() / len(dataloader)).tolist()}

    if from_scratch:
        with open(f'./evals/{w}.pickle', 'wb') as handle:
            pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if len(jdict):
            pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
            print('\nsaving json %s...' % pred_json)
            with open(pred_json, 'w') as f:
                json.dump(jdict, f)

    # Print results
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
        maps[c] = ap[i]

    # Plots
    if from_scratch:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        del model

    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t, a

import argparse
import os
import pickle
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import cv2
import torch
import torchvision
import yaml
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm

import evaluate
import test
from models.experimental import attempt_load
from models.yolo import Model, nn, np, amp
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader, letterbox
from utils.general import colorstr, increment_path, init_seeds, one_cycle, check_img_size, labels_to_class_weights, \
    strip_optimizer, xywh2xyxy, scale_coords
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.torch_utils import select_device, logger, intersect_dicts, ModelEMA, de_parallel

import os.path
from os import path
from math import isclose, sqrt

DEVICES = 9
# CLASSES = 2
AL_CONF_THRESHOLD = 0.25
AL_IOU_THRESHOLD = 0.45


# BASE_OPTIONS = {
#         'classes': CLASSES,
#         'epochs': 20,
#         'al_samples': 20,
#         'al_method': 'sum',
#         'save_dir': 'runs/test/',
#         'yolo_dir': f'KITTI/yolo{CLASSES}/',
#         'clients_dir': f'KITTI/yolo{CLASSES}/clients/run-14/',
#         'batch_size': 16,
#         'weights': './startingpoint.pt',
#         'img_size': 640,
#         'device': '',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
#         'adam': False,  # use torch.optim.Adam() optimizer
#         'test': False,
#         'test_final': False,
#         'config': 'data/run-14-0.yaml',
#         'hyp': 'data/hyp.scratch.yaml',
#         'model': 'models/yolov5s.yaml',  # cfg
#         'pretrain': False,
#         'name': '14',
#         'freeze_backbone': False,
#         'workers': 4,
#         'nosave': False,
#         'cache': False
# } # OVERRIDE epochs, weights, config, name, freeze_backbone, classes, clients_dir,

# def create_options_object(custom_options):
#     class Object(object):
#         pass

#     options = Object()
#     _opts = BASE_OPTIONS.copy()
#     _opts.update(custom_options)
#     for p in _opts:
#         setattr(options, p, _opts[p])


def start_training(options):
    if not ((path.exists(options.weights) or options.pretrain) and path.exists(options.model) and path.exists(
            options.hyp)):
        print(
            f"Missing files. Existing are: weights: {options.weights}{path.exists(options.weights)}, model: {path.exists(options.model)}, hyperparameters: {path.exists(options.hyp)}")
        return
    # select device for training
    device = select_device(options.device, batch_size=options.batch_size)
    cuda = device.type != 'cpu'

    epochs = options.epochs
    # load hyperparameters from yaml file
    with open(options.hyp) as f:
        hyp = yaml.safe_load(f)

        # print hyperparameters
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # create needed directories
    save_dir = Path(
        str(increment_path(Path('runs/') / f'run-{options.name}-{options.client}-{options.round}', exist_ok=False)))
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    best_pickle = wdir / 'best.pickle'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(options), f, sort_keys=False)

    # Configure
    init_seeds(1)

    nc = int(options.classes)  # number of classes
    if nc == 1:  # classes names
        names = ['item']
    elif nc == 2:
        names = ['car', 'pedestrian']
    elif nc == 8:
        names = ['car', 'cyclist', 'misc', 'pedestrian', 'person_sitting', 'tram', 'truck', 'van']

    # Model
    if options.weights.endswith('.pt'):
        ckpt = torch.load(options.weights, map_location=device)  # load checkpoint
        model = Model(options.model or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (options.model or hyp.get('anchors')) else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info(
            'Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), options.weights))  # report
    else:  # we're creating a new pretrained point, so create new model.
        model = Model(options.model, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    train_path = options.clients_dir + 'l-' + str(options.client) + '.txt'
    test_path = options.yolo_dir + 'val.txt'

    # Freeze necessary layers.
    freeze = ['model.%s.' % x for x in range(10)] if options.freeze_backbone else []
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('Freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / options.batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= options.batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if options.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if options.weights.endswith('.pt'):
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (options.weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = check_img_size(640, gs), check_img_size(640, gs)

    # DP mode
    if cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, options.batch_size, gs, options,
                                            hyp=hyp, augment=True, cache=False, rect=False, rank=-1,
                                            world_size=1, workers=options.workers,
                                            image_weights=False, quad=False, prefix=colorstr('train: '))

    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g. Possible class labels are 0-%g' % (mlc, nc, nc - 1)

    # Process 0
    testloader = create_dataloader(test_path, imgsz_test, options.batch_size * 2, gs, options,  # testloader
                                   hyp=hyp, cache=False, rect=True, rank=-1,
                                   world_size=1, workers=options.workers,
                                   pad=0.5, prefix=colorstr('val: '))[0]
    # TODO

    # Anchors
    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
    model.half().float()  # pre-reduce anchor precision

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = 0.0
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(4, device=device)  # mean losses

        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / options.batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # mAP
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
        final_epoch = epoch + 1 == epochs
        pickledump = None
        if options.test or (options.test_final and final_epoch):   # Calculate mAP
            results, maps, times, pickledump = evaluate.evaluate(None,
                                                                 model=ema.ema,
                                                                 dataloader=testloader,
                                                                 save_dir=save_dir,
                                                                 nc=nc,
                                                                 compute_loss=compute_loss,
                                                                 plots=False)

        # Write
        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        if (not options.nosave) or final_epoch:  # if save
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'training_results': results_file.read_text(),
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': None}

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
                with open(best_pickle, 'wb') as handle:
                    pickle.dump(pickledump, handle, protocol=pickle.HIGHEST_PROTOCOL)
            del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    # Strip optimizers
    final = best if best.exists() else last  # final model
    for f in last, best:
        if f.exists():
            strip_optimizer(f)  # strip optimizers
    torch.cuda.empty_cache()


class CustomImageLoader:
    def __init__(self, files, img_size=640, stride=32):
        images = [x for x in files if x.split('.')[-1].lower() == "jpg"]

        self.img_size = img_size
        self.stride = stride
        self.files = images
        self.nf = len(images)  # number of files
        self.mode = 'image'
        self.cap = None
        assert self.nf > 0, f'No supported images found. '

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        # Read image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        assert img0 is not None, 'Image Not Found ' + path

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def __len__(self):
        return self.nf  # number of files


def al_sample(samples, result, method):
    temp = {}

    for item, res_dic in result.items():
        if len(res_dic) == 0:
            continue
        conf = 0
        for dic in res_dic:
            if method == 'sum' or method == 'avg':
                conf += (1.0 - dic["conf"])
            elif method == 'max':
                conf = max(conf, 1.0 - dic["conf"])
        temp[item] = conf / len(res_dic) if method == 'avg' else conf
    return sorted(temp, key=temp.get, reverse=True)[:samples]


def al_get_confidences(weights_file, files):
    with torch.no_grad():
        imgsz = 640
        device = select_device('') # TODO different device based on options.
        half = device.type != 'cpu'
        model = attempt_load(weights=weights_file, map_location=device)
        model.half()
        imgsz = check_img_size(imgsz, s=model.stride.max())
        dataset = CustomImageLoader(files, img_size=imgsz)

        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        result = {}

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32

            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            prediction = model(img, augment=True)[0]

            xc = prediction[..., 4] > AL_CONF_THRESHOLD  # candidate predictions

            nms_output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
            for xi, x in enumerate(prediction):  # image index, image inference
                x = x[xc[xi]]  # Confidence constraints

                # If none remain process next image
                if not x.shape[0]:
                    continue

                # Compute conf, can't use this conf yet cause we prefer NMS.
                x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

                # Box (center x, center y, width, height) to (x1, y1, x2, y2)
                box = xywh2xyxy(x[:, :4])

                # Detections matrix nx6, with every row as: (xyxy, conf, cls)
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > AL_CONF_THRESHOLD]

                if not x.shape[0]:  # no boxes
                    continue
                elif x.shape[0] > 30000:  # excess boxes
                    x = x[x[:, 4].argsort(descending=True)[:30000]]  # sort by confidence

                # Batched NMS
                boxes, scores = x[:, :4], x[:, 4]  # boxes (offset by class), scores
                i = torchvision.ops.nms(boxes, scores, AL_IOU_THRESHOLD)  # NMS

                nms_output[xi] = x[i]

            # Process detections
            result[path] = []

            for i, det in enumerate(nms_output):  # detections per image
                p, im0, frame = path, im0s, getattr(dataset, 'frame', 0)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # print(det)
                    for *xyxy, conf, _ in reversed(det):  # for output vector of yolo model
                        data = {"conf": conf.item()}
                        result[path].append(data)
        return result


def al_step(client, weights, options):
    # Check which files are unlabeled for each client device.
    files = []
    with open(options.clients_dir + "ul-{}.txt".format(client)) as f:
        files = [line.rstrip('\n') for line in f]
    print(f'Loaded {len(files)} unlabelled files, which will be labelled through active learning')

    # Choose correct AL method, label and get a selection.
    if options.al_method == 'rnd':
        selection = random.sample(files, options.al_samples)
    else:
        confidence_results = al_get_confidences(weights, files)
        # Selects images from ul-0.txt for client 0
        selection = al_sample(options.al_samples, confidence_results, options.al_method)
    print(f'Labelled {len(selection)}/{options.al_samples} unlabelled files')

    # Write new labelled and unlabeled to files:
    with open(options.clients_dir + 'l-{}.txt'.format(client), "a") as f:
        for file_name in selection:
            f.write(file_name + '\n')
    with open(options.clients_dir + "ul-{}.txt".format(client), "w") as f:
        ul_without_selection = [x for x in files if x not in selection]
        for line in ul_without_selection:
            f.write(line + "\n")
    print("Reflected AL step in ul and labelled set files")


def error_gen(actual, rounded):
    divisor = sqrt(1.0 if actual < 1.0 else actual)
    return abs(rounded - actual) ** 2 / divisor


def round_to_100(percents):
    if not isclose(sum(percents), 100):
        raise ValueError
    n = len(percents)
    rounded = [int(x) for x in percents]
    up_count = 100 - sum(rounded)
    errors = [(error_gen(percents[i], rounded[i] + 1) - error_gen(percents[i], rounded[i]), i) for i in range(n)]
    rank = sorted(errors)
    for i in range(up_count):
        rounded[rank[i][1]] += 1
    return rounded


def get_state_dict(config, channels, classes, fname):
    model = Model(config, channels, classes)
    ckpt = torch.load(fname, map_location=torch.device('cpu'))  # load
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].shape == v.shape}  # filter
    model.load_state_dict(state_dict, strict=False)  # load
    if len(ckpt['model'].names) == classes:
        model.names = ckpt['model'].names
    return state_dict


def pseudo_fed_avg(clients_and_sizes, iteration, options):  # [[0, weightsfile, 500]]
    states = {}
    total_sample_size = 0
    for client, weights, sample_size in clients_and_sizes:
        states[client] = get_state_dict(options.model, 3, options.classes, weights)
        total_sample_size += sample_size
    percentages_non_rounded = [(sample_size / total_sample_size) * 100 for _, _, sample_size in clients_and_sizes]
    percentages_rounded = round_to_100(percentages_non_rounded)

    fname = clients_and_sizes[0][1]  # take first model as a base and load it as a changable model
    new_model = torch.load(fname, map_location=torch.device('cpu'))
    state_dict = new_model['model'].float().state_dict()

    for key, value in state_dict.items():
        values_for_key = [states[client][key] * (p / 100) for client, p in
                          zip(range(len(percentages_rounded)), percentages_rounded)]
        state_dict[key] = sum(values_for_key)

    new_model['model'].load_state_dict(state_dict)
    torch.save(new_model, f"pseudo_fl/fed-{options.name}-{iteration}.pt")


def aggchain(opt):
    results = {}
    for client in range(DEVICES):
        pickle_file = f'runs/run-{opt.name}-{client}-{opt.round}/weights/best.pickle'
        # pickle_file = f'evals/run-{opt.name}-{client}-{opt.round}.pickle'
        with open(pickle_file, 'rb') as f:
            result = pickle.load(f)
            results[client] = dict()
            for clsname, seen, nt, p, r, ap50, ap in result['res_per_class']:
                results[client][
                    clsname] = ap50 if opt.aggr_metric == "ap50" else r if opt.aggr_metric == "r" else ap
    # get the best devices according to the metric
    runs = [0 for _ in range(DEVICES)]
    weights = 1 / opt.classes  # 1/classes
    for cls in results[0]:
        best = max([(i, results[i][cls]) for i in range(DEVICES)], key=lambda k: k[1])[0]
        runs[best] += weights
    # and aggregate them into one model
    all_state_dict = [get_state_dict(opt.model, 3, opt.classes, f"runs/run-{opt.name}-{i}-{opt.round}/weights/best.pt")
                      for i in range(DEVICES)]
    base_model = runs.index(max(runs))
    print(f"Aggregating with weights: {[(client, weight) for client, weight in enumerate(runs)]}")
    fname = f'runs/run-{opt.name}-{base_model}-{opt.round}/weights/best.pt'
    model = torch.load(fname, map_location=torch.device('cpu'))
    state_dict = model['model'].float().state_dict()
    for key, value in state_dict.items():
        state_dict[key] = sum(weight * all_state_dict[client][key] for client, weight in enumerate(runs))

    model['model'].load_state_dict(state_dict)
    torch.save(model, f"pseudo_fl/{opt.name}agg.pt")


def init(options):
    if not os.path.exists(options.clients_dir):
        os.makedirs(options.clients_dir)
    for client in range(0, DEVICES):
        if os.path.exists(options.clients_dir + "l-{}.txt".format(client)):
            os.remove(options.clients_dir + "l-{}.txt".format(client))
        if os.path.exists(options.clients_dir + "l-{}.cache".format(client)):
            os.remove(options.clients_dir + "l-{}.cache".format(client))
        with open(options.yolo_dir + "clients/{}.txt".format(client), "r") as f:
            with open(options.clients_dir + "ul-{}.txt".format(client), "w") as f2:
                for line in f.readlines():
                    f2.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='', help='action to execute: either [al, train, fedavg]')
    parser.add_argument('--round', type=int, default=0, help='')
    parser.add_argument('--client', type=int, default=0, help='')
    parser.add_argument('--classes', type=int, default=2, help='amount of classes')
    parser.add_argument('--epochs', type=int, default=20,
                        help='amount of epochs that will be trained when action is train')
    parser.add_argument('--al_samples', type=int, default=10,
                        help='amount of samples that will be selected in AL when action is al')
    parser.add_argument('--reuse_weights', action='store_true',
                        help='use the weights from last fedavg iteration for a new training iteration. WARNING: Overwrites weights argument.')
    parser.add_argument('--al_method', type=str, default='sum',
                        help='active learning aggregation method for samples when action is al')
    parser.add_argument('--save_dir', type=str, default='runs/test', help='')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--weights', type=str, default='', help='')
    parser.add_argument('--img_size', type=int, default=640, help='')
    parser.add_argument('--aggr_metric', type=str, default='r', help='aggregation metric, only used in chained aggregation first round.')
    parser.add_argument('--device', type=str, default='', help='')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--test', action='store_true', help='')
    parser.add_argument('--test_final', action='store_true', help='')
    parser.add_argument('--hyp', type=str, default='hyps/hyp.scratch.yaml', help='')
    parser.add_argument('--model', type=str, default='models/yolov5s.yaml', help='')
    parser.add_argument('--pretrain', action='store_true', help='')
    parser.add_argument('--name', type=str, default='14', help='')
    parser.add_argument('--freeze_backbone', action='store_true', help='')
    parser.add_argument('--workers', type=int, default=4, help='')
    parser.add_argument('--nosave', action='store_true', help='')
    parser.add_argument('--cache', action='store_true', help='')
    opt = parser.parse_args(sys.argv[1:])
    opt.yolo_dir = f'KITTI/yolo{opt.classes}/'
    opt.clients_dir = f'KITTI/yolo{opt.classes}/clients/run-{opt.name}/'
    opt.single_cls = False  # needed to reuse their dataloader ...
    if opt.reuse_weights:
        opt.weights = f"pseudo_fl/fed-{opt.name}-{opt.round - 1}.pt"
    if opt.action == 'al':
        al_step(opt.client, opt.weights, opt)
    elif opt.action == 'train':
        start_training(opt)
    elif opt.action == "fedavg":
        fed_set = [(client, f'runs/run-{opt.name}-{client}-{opt.round}/weights/best.pt', (opt.round + 1) * 10)
                   for client in range(DEVICES)]
        pseudo_fed_avg(fed_set, opt.round, opt)
    elif opt.action == "aggchain":
        aggchain(opt)
    elif opt.action == "init":
        init(opt)

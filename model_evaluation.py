import torch
from torch.utils.data import DataLoader
from config import TEST_CSV, TEST_DIR, BATCH_SIZE, NUM_WORKERS, NUM_CLASSES, DEVICE, CLASSES, OUT_DIR
from custom_utils import collate_fn, ConfusionMatrix
from model import create_model
from dataset import HemorrhageDataset

import numpy as np
import matplotlib.pyplot as plt

import time

from tqdm import tqdm

from custom_utils import box_iou

from pathlib import Path


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)


def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec



def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = {i: v for i, v in enumerate(names)}  # to dict
    if plot:
        plot_pr_curve(px, py, ap, Path(OUT_DIR) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(OUT_DIR) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(OUT_DIR) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(OUT_DIR) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype('int32')


@torch.no_grad()
def eval():
    # Configure
    iouv = torch.linspace(0.5, 0.95, 10, device=DEVICE)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    plots = True

    # Select device
    cuda = DEVICE.type != "cpu"

    # Dataset, DataLoader
    dataset = HemorrhageDataset(TEST_DIR, TEST_CSV)
    loader = DataLoader(dataset=dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=NUM_WORKERS,
                        collate_fn=collate_fn
                        )
    loader.dataset.set_transform(0)  # always evaluate on untransformed images

    model = create_model(NUM_CLASSES)
    model.to(DEVICE)
    model.eval()  # To inference mode

    seen = 0
    names = {k: v for k, v in enumerate(CLASSES[1:])}  # Skip the __background__ class
    confusion_matrix = ConfusionMatrix(nc=NUM_CLASSES - 1)  # Initialize confusion matrix
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    pbar = tqdm(loader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    jdict, stats, ap, ap_class = [], [], [], []

    # Loop over all batches
    for batch_index, (images, targets) in enumerate(pbar):
        t1 = time_sync()
        if cuda:
            images = list(image.to(DEVICE) for image in images)  # Move images to DEVICE for faster computation
        t2 = time_sync()
        dt[0] += t2 - t1  # Prep time

        # During inference model takes only batch of images of shape (B, C, H, W) as argument
        out = model(images)

        t3 = time_sync()
        dt[1] += t3 - t2  # Inference time

        # Compute metrics
        for i, pred_dict in enumerate(out):  # Loop over all predictions
            target = targets[i]
            # Ground truth labels
            gt_labels = target["labels"]

            # Prediction labels
            pred_labels = pred_dict["labels"]

            num_gt_labels, num_pred_labels = len(gt_labels), len(pred_labels)  # label counts

            correct = torch.zeros(num_pred_labels, niou, dtype=torch.bool, device=DEVICE)  # init
            seen += 1

            if num_pred_labels == 0:
                if num_gt_labels:
                    stats.append((correct, *torch.zeros((3, 0), device=DEVICE)))
                continue

            # Set predictions to following format: (Array[N, 6]), x1, y1, x2, y2, conf, class
            pred_boxes = pred_dict["boxes"]
            pred_conf = pred_dict["scores"]
            pred_labels = torch.reshape(pred_labels, (len(pred_labels), 1))
            pred_conf = torch.reshape(pred_conf, (len(pred_conf), 1))
            pred_fn = torch.cat((pred_boxes, pred_conf, pred_labels), 1)  # Concatenate prediction tensor

            # Set ground truth tensor of (Array[M, 5]), class, x1, y1, x2, y2
            gt_boxes = target["boxes"]
            gt_labels = torch.reshape(gt_labels, (len(gt_labels), 1))
            gt_fn = torch.cat((gt_labels, gt_boxes), 1)  # Concatenate ground truth tensor

            # Evaluate
            if num_gt_labels:
                correct = process_batch(pred_fn, gt_fn, iouv)
                confusion_matrix.process_batch(pred_fn, gt_fn)
            stats.append((correct, pred_fn[:, 4], pred_fn[:, 5], gt_fn[:, 0]))

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=OUT_DIR, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=NUM_CLASSES - 1)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    for i, c in enumerate(ap_class):
        print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))


if __name__ == "__main__":
    eval()

import random

from model import create_model
from dataset import HemorrhageDataset
from config import TRAIN_CSV, TRAIN_DIR, OUT_DIR, NUM_CLASSES, DEVICE, ROOT_DIR, CLASSES, COLOR_MAP
from show_sample import sample_show

import torch
import torchvision.transforms.functional as TF
from torchvision.ops import batched_nms, nms

import random
import os
from PIL import Image, ImageDraw, ImageFont


@torch.no_grad()
def visual_comparison(image_dir, annot_dir, model_path_file):
    dataset = HemorrhageDataset(image_dir, annot_dir)
    # Create model and load save parameters, put model into eval mode
    model = create_model(NUM_CLASSES)
    checkpoint = torch.load(model_path_file, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(DEVICE)

    # Show original images
    set_len = dataset.__len__()
    ind = random.randint(0, set_len)
    sample_show(image_dir, annot_dir, ind)

    # Show model prediction
    dataset.set_transform(0)
    tensor, target = dataset[ind]

    pil_image = TF.to_pil_image(tensor)
    draw = ImageDraw.Draw(pil_image)
    fontsize = 20
    font_path = "/Users/krystof/Desktop/arial.ttf"
    font = ImageFont.truetype(font_path, fontsize)

    tensor = tensor.unsqueeze(0)
    out = model(tensor)
    out = out[-1]
    pred_bboxes = out["boxes"]
    classes = out["labels"]
    scores = out["scores"]
    print(out)


    # Filter by scores
    conf_thres = 0.6
    pred_labels = torch.reshape(classes, (len(classes), 1))
    pred_conf = torch.reshape(scores, (len(scores), 1))
    pred = torch.cat((pred_bboxes, pred_conf, pred_labels), 1)
    filtered = pred[pred[:, 4] > conf_thres]

    # NMS
    scores = filtered[:, 4]
    labels = filtered[:, 5]
    boxes = filtered[:, :4]

    iou_thres = 0.4
    ind = nms(boxes, scores, iou_thres)

    scores = scores[ind]
    labels = labels[ind]
    boxes = boxes[ind]

    pred_labels = torch.reshape(labels, (len(labels), 1))
    pred_conf = torch.reshape(scores, (len(scores), 1))
    pred = torch.cat((boxes, pred_conf, pred_labels), 1)


    if len(pred) > 0:
        for row in filtered:
            x, y, x2, y2 = row[0].numpy(), row[1].numpy(), row[2].numpy(), row[3].numpy()
            rectangle = (x, y, x2, y2)
            cls = int(row[-1].numpy())
            fill = COLOR_MAP[CLASSES[cls]]
            draw.text(xy=(x, y - 25), font=font, text=CLASSES[cls], align="right", fill=fill)
            draw.rectangle(rectangle, None, fill, 2)

    pil_image.show()


if __name__ == "__main__":
    visual_comparison(TRAIN_DIR, TRAIN_CSV, os.path.join(ROOT_DIR, "outputs/best_model.pth"))


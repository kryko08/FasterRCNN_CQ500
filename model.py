import torchvision
import torch

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

from config import NUM_CLASSES, TRAIN_CSV, TRAIN_DIR, BATCH_SIZE, NUM_WORKERS
from dataset import HemorrhageDataset
from custom_utils import collate_fn


def create_model(num_classes):

    # Load Faster R-CNN with Resnet50 backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # New head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# Test forward method
if __name__ == "__main__":
    model = create_model(NUM_CLASSES)
    dataset = HemorrhageDataset(TRAIN_DIR, TRAIN_CSV)
    loader = DataLoader(dataset=dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=NUM_WORKERS,
                        collate_fn=collate_fn
                        )
    loader.dataset.set_transform(0)
    # Training
    images, targets = next(iter(loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)  # Returns losses and detections
    print(output)
    # Evaluation
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)  # Returns predictions
    print(predictions)





import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

RANDOM_SEED = 265
EPOCH_COUNT = 10
BATCH_SIZE = 256
MODEL_NAME = "v3_adam_lr0.001_wd0.001"

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

train_data = torch.load(f'data/localization_train.pt')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

val_data = torch.load(f'data/localization_val.pt')
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()

        # Input = (1, 48, 60)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Input = (32, 24, 30)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Input = (64, 12, 15)
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,3))
        )

        # Input = (128, 6, 5)
        self.cnn_size = 128 * 6 * 5

        self.confidence = nn.Sequential(
            nn.Linear(self.cnn_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_size, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )

        self.bbox = nn.Sequential(
            nn.Linear(self.cnn_size, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )

    def forward(self, x) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(-1, self.cnn_size)

        return self.confidence(out), self.classifier(out), self.bbox(out)

def log(msg):
    print(msg)
    with open(f"models/{MODEL_NAME}.txt", "a") as f:
        f.write(msg + "\n")

def train(model, optimizer):

    log(f"Model: {MODEL_NAME}")

    model.train()
    for epoch in range(EPOCH_COUNT):

        total_loss = 0.0
        total_size = 0

        for i, (images, labels) in enumerate(train_loader):

            true_confidence = labels[:, 0]
            true_class = F.one_hot(labels[:, -1].long(), num_classes=10).float()
            true_bbox = labels[:, 1:5]

            has_object_mask = true_confidence > 0.5

            optimizer.zero_grad()

            pred_confidence, pred_class, pred_bbox = model(images)

            pred_confidence = pred_confidence.squeeze()  # Fix the shape
            
            loss_confidence = F.binary_cross_entropy(pred_confidence, true_confidence)
            loss_bbox = F.mse_loss(pred_bbox[has_object_mask], true_bbox[has_object_mask])
            loss_class = F.cross_entropy(pred_class[has_object_mask], true_class[has_object_mask])

            loss = (loss_confidence + loss_bbox + loss_class)
            loss.backward()

            optimizer.step()

            total_size += images.size(0)
            total_loss += loss.item()

        torch.save(model.state_dict(), f'models/train/{MODEL_NAME}_e{epoch}.pt')
        log(f'Epoch {epoch + 1}/{EPOCH_COUNT}, \tLoss: {total_loss / total_size}')
    torch.save(model.state_dict(), f'models/{MODEL_NAME}.pt')


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    """
    # Get the coordinates of bounding boxes

    box1 = box1.unsqueeze(0)
    box2 = box2.unsqueeze(0)

    x1, y1, w1, h1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    x2, y2, w2, h2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Calculate the coordinates of the intersection rectangle
    x_inter1 = torch.max(x1, x2)
    y_inter1 = torch.max(y1, y2)
    x_inter2 = torch.min(x1 + w1, x2 + w2)
    y_inter2 = torch.min(y1 + h1, y2 + h2)

    # Calculate the area of intersection rectangle
    intersection = torch.clamp(x_inter2 - x_inter1 + 1, min=0) * torch.clamp(y_inter2 - y_inter1 + 1, min=0)

    # Calculate the area of both bounding boxes
    area_box1 = w1 * h1
    area_box2 = w2 * h2

    # Calculate the IoU
    iou = intersection / (area_box1 + area_box2 - intersection)

    return iou

def validate(model, data_loader):
    confidence_total = 0
    confidence_correct = 0

    classifier_total = 0
    classifier_correct = 0

    combined_total = 0
    combined_correct = 0

    total_iou = 0
    total_iou_count = 0

    model.eval()
    for i, (images, labels) in enumerate(val_loader):

        true_confidence = labels[:, 0]
        true_class = F.one_hot(labels[:, -1].long(), num_classes=10).float()
        true_bbox = labels[:, 1:5]

        pred_confidence, pred_class, pred_bbox = model(images)

        total_iou += bbox_iou(pred_bbox, true_bbox).sum().item()

        batch_size = images.size(0)
        for i in range(batch_size):
            
            has_object = true_confidence[i].item() > 0.5
            pred_has_object = pred_confidence[i].item() > 0.5

            confidence_total += 1
            combined_total += 1
            if has_object is False and pred_has_object is False:
                confidence_correct += 1
                combined_correct += 1
                continue

            if has_object is False:
                continue
            
            if has_object == pred_has_object:
                confidence_correct += 1

            true_class_label = true_class[i].argmax().item()
            pred_class_label = pred_class[i].argmax().item()

            confidence_correct
            classifier_total += 1
            if true_class_label == pred_class_label:
                classifier_correct += 1
                combined_correct += 1

            total_iou_count += 1
            total_iou += bbox_iou(pred_bbox[i], true_bbox[i]).item()

    confidence_accuracy = confidence_correct * 100 / confidence_total
    classifier_accuracy = classifier_correct * 100 / classifier_total

    accuracy = combined_correct * 100 / combined_total
    iou = total_iou / total_iou_count
    score = accuracy + iou

    log(f"IoU: {iou}, Confidence: {confidence_accuracy}, Classifier: {classifier_accuracy}, Accuracy: {accuracy}")
    return score, accuracy, iou

model = Cnn()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

# model.load_state_dict(torch.load(f'models/{MODEL_NAME}_e1.pt'))
train(model, optimizer)
validate(model, val_loader)

# Assuming tensor t is a 2D image tensor
sample_image = val_data[0][0]
sample_label = val_data[0][1]

plt.imshow(sample_image[0], cmap='gray')

def draw_bbbox(bbox:torch.Tensor, label:str = None):
    [x, y, w, h] = bbox.tolist()
    w *= sample_image.shape[2]
    h *= sample_image.shape[1]
    x *= sample_image.shape[2]
    y *= sample_image.shape[1]
    x -= w / 2
    y -= h / 2
    plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none'))

draw_bbbox(sample_label[1:5])

_, sample_pred_class, sample_pred_bbox = model(sample_image)
draw_bbbox(sample_pred_bbox[0], sample_pred_class.argmax().item())

plt.axis('off')
plt.show()
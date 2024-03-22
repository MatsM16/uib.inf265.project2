import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Size 1x48x60
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Size 16x24x30
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Size 32x12x15
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Size 64x6x7

        self.fc1 = nn.Linear(16*6*7, 400)
        self.fc2 = nn.Linear(400, 4)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 16*6*7)  
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

train_data = torch.load(f'data/localization_train.pt')
train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)

val_data = torch.load(f'data/localization_val.pt')

model = CNN()

def train():
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    losses = []

    for epoch in range(10):
        total_loss = 0
        total_size = 0
        for i, [images, labels] in enumerate(train_loader):
            
            label_box = labels[:, 1:5]

            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, label_box)
            loss.backward()
            optimizer.step()

            total_loss += loss
            total_size += labels.size(0)

        torch.save(model.state_dict(), f'models/loc_e{epoch+1}.pt')

        losses.append(total_loss)

        print(f'Epoch: {epoch+1}, Loss: {total_loss / total_size}')

    torch.save(model.state_dict(), f'models/loc.pt')

def load():
    model.load_state_dict(torch.load('models/loc_e2.pt'))

def validate(dataset):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for image, label in dataset:
            label_class = F.one_hot(label[-1].long(), num_classes=10).float()

            pred = model(image)
            _, predicted = torch.max(pred, 1)

            total += label.size(0)
            correct += (predicted == label_class).sum().item()

    accuracy = 100.0 * correct / total
    print(f'Accuracy: {accuracy:.2f}% ({correct}/{total})')
    return accuracy

def validate_loc(dataset):
    model.eval()
    with torch.no_grad():
        for image, label in dataset:
            pred = model(image)
            def calculate_iou(label, pred):
                intersection = torch.sum(torch.min(label, pred))
                union = torch.sum(torch.max(label, pred))
                iou = intersection / union
                return iou
            iou = calculate_iou(label, pred)
            print(f'IoU: {iou:.2f}')
        

train()
validate(val_data)
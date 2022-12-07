from dataset import ShoeDataset
from models import SimpleModel
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import timm
train_data_path = "/mnt/nvme0n1p5/Kaggle/archive/Shoes Dataset/Train"

train_dataset = ShoeDataset(train_data_path, 224)
train_loader = DataLoader(train_dataset, 16, shuffle=True)

# model = SimpleModel(5)
model = timm.create_model('mobilenetv2_100', num_classes=5, pretrained=True)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 10
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)
model.to(device)
for epoch in range(epochs):
    print("Training epoch {}/{}".format(epoch, epochs))
    model.train()
    epoch_loss = 0
    for i, (imgs, labels) in enumerate(train_loader):
        # imgs shape [n,c,h,w]
        imgs = imgs.to(device)
        labels = labels.to(device)
        labels = labels.long()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        print("Epoch [{}], Step [{}/{}], Loss [{}]".format(epoch, i, len(train_loader), loss.item()))
    print("Epoch_loss : [{}]".format(epoch_loss/len(train_loader)))


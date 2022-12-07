import torch
import torch.nn as nn
class MiniModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2)
    def forward(self, x):
        return self.pool(self.relu(self.conv1(x)))
class SimpleModel(nn.Module):
    def __init__(self, n_classes = 5):
        super().__init__()
        list_channels = [3,16,32,64,128,256]
        self.n_classes = n_classes
        list_modules = []
        for i in range(1, len(list_channels)):
            in_channels = list_channels[i-1]
            out_channels = list_channels[i]
            list_modules.append(MiniModel(in_channels=in_channels, out_channels=out_channels, kernel_size=3))
        self.convs = nn.ModuleList(list_modules)
        self.linear = nn.Linear(list_channels[-1], n_classes)
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = torch.mean(x, dim=(2,3))
        x = self.linear(x)
        return x


if __name__=="__main__":
    x = torch.zeros((1,3,256,256))
    model = SimpleModel(4)
    out = model(x)
    print(out.shape)

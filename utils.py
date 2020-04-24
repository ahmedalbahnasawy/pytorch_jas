import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms


model = models.resnet50()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr =1e-3)
dataset = datasets.FakeData(size=1000, transform=transforms.ToTensor())
loader = DataLoader(dataset, num_workers=1, pin_memory = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

for data, target in loader:
    data = data.to(device, non_blocking=True)
    target = torch.tensor(target, dtype = torch.long , device= device)
    #target = target.to(device, non_blocking =False)
    optimizer.zero_grad()
    oput = model(data)
    loss = criterion(oput, target)
    print(loss)
    loss.backward()
    optimizer.step()
print('Done')
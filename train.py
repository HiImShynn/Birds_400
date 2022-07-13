import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import time

from data.data_prep import Birds
from model.vgg_19 import VGG_19

# Hyperparameter
batch_size = 4
lr = 1e-3
momentum = 0.8

# Device
device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
torch.cuda.empty_cache()

# Prepare data
trans = transforms.Compose([
    transforms.ToTensor(),
])
train_set = Birds("./data", "data/birds.csv", "train", transform= trans)
test_set = Birds("./data", "data/birds.csv", "test", transform= trans)
valid_set = Birds("./data", "data/birds.csv", "valid", transform= trans)
data_train = DataLoader(
    train_set,
    batch_size= batch_size,
    shuffle= True
)
data_test = DataLoader(
    test_set,
    batch_size= batch_size,
    shuffle= True
)
data_valid = DataLoader(
    valid_set,
    batch_size= batch_size,
    shuffle= True
)

# Prepare model
model = VGG_19(3, 400, [3, 64, 128, 256, 512], [2, 2, 4, 4, 4])
model.to(device)
optimizer = optim.Adam(model.parameters(), lr= lr)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.8)
criterion = nn.CrossEntropyLoss()

def train(num_epochs):
    for epoch in range(num_epochs):
        train_corrects = 0
        train_loss = 0
        begin = time.time()
    
    #trainning step
        for i, (inputs, labels) in enumerate(data_test):
            # forward
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            #calculating loss
            loss = criterion(outputs, labels)

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_corrects += (outputs.argmax(dim = 1) == labels).float().sum()

        train_accuracy = train_corrects * 100 / len(train_set)
    
    #validation step
        valid_corrects = 0
        valid_loss = 0
        for i, (inputs, labels) in enumerate(data_valid):
            #forward
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            #calculating loss
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            valid_corrects += (outputs.argmax(dim = 1) == labels).float().sum()

        valid_accuracy = valid_corrects * 100 / len(valid_set)
        end = time.time()
    print("Epoch: {}/{} - Train_accuracy: {} - Train_loss: {} - Valid_accuracy: {} - Valid_loss: {} - Time: {}".format(
        epoch, num_epochs, train_accuracy, train_loss, valid_accuracy, valid_loss, end - begin
    ))

if __name__ == "__main__":
    train(2)
from tqdm import tqdm
import torch

def train_model(model, trainloader, criterion, optimizer, device, disable_loading_bar=False):
    model.train()
    for inputs, labels in tqdm(trainloader, desc="Training", leave=False, disable=disable_loading_bar, mininterval=1.0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def test_model(model, testloader, criterion, device, disable_loading_bar=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Testing", leave=False, disable=disable_loading_bar, mininterval=1.0):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return test_loss / len(testloader.dataset), correct / len(testloader.dataset)

def train_model_robust(model, trainloader, criterion, optimizer, device, noise_funct, disable_loading_bar=False):
    model.train()
    for inputs, labels in tqdm(trainloader, desc="Training", leave=False, disable=disable_loading_bar, mininterval=1.0):
        inputs = noise_funct(inputs)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
import argparse
import torch
import torchvision
import os

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from helpers.setup_train import get_dataset, get_model
from tqdm import tqdm

def make_run_dir(args):
    # Create directories for saving model weights
    base_dir = './model_weights'
    dataset_dir = os.path.join(base_dir, args.dataset)
    model_dir = os.path.join(dataset_dir, args.model)
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Find the next run number
    run_number = 1
    while os.path.exists(os.path.join(model_dir, f'run_{run_number}')):
        run_number += 1
    run_dir = os.path.join(model_dir, f'run_{run_number}')
    os.makedirs(run_dir)

    return run_dir

def train_model(model, trainloader, criterion, optimizer, device):
    model.train()
    for inputs, labels in tqdm(trainloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def test_model(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return test_loss / len(testloader.dataset), correct / len(testloader.dataset)

def main(args):
    if args.cuda:
        # Set CUDA device
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    trainloader, testloader = get_dataset(args.dataset, args.batch_size)
    num_classes = 10 if args.dataset in ['CIFAR10', 'MNIST'] else 100 if args.dataset == 'CIFAR100' else 1000
    model = get_model(args.model, num_classes, args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run_dir = make_run_dir(args)

    # Update the path for saving the best model
    best_model_path = os.path.join(run_dir, 'best_model.pt')

    best_accuracy = 0
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        train_model(model, trainloader, criterion, optimizer, device)
        test_loss, test_accuracy = test_model(model, testloader, criterion, device)
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
        
        # Save the model if the test accuracy is the best we've seen so far
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= args.plateau_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    # Save the last set of weights
    last_model_path = os.path.join(run_dir, 'last_model.pt')
    torch.save(model.state_dict(), last_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a torchvision model on a dataset')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'MNIST', 'ImageNet'], help='Dataset to use')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'alexnet', 'vgg16', 'squeezenet', 'densenet', 'inception', 
                                                                         'googlenet', 'shufflenet', 'mobilenet', 'resnext50_32x4d', 
                                                                         'wide_resnet50_2', 'mnasnet', 'efficientnet', 'efficientnet_v2'], help='Model to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--plateau_patience', type=int, default=6, help='Patience for early stopping on plateau')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA if available')
    parser.add_argument('--device', type=str, default='7', help='CUDA device number')
    args = parser.parse_args()

    main(args)
import argparse
import torch
import torchvision
import os

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from helpers.setup_train import get_dataset, get_model, replace_relu_with_selu, replace_relu_with_any
from tqdm import tqdm
import csv
from helpers.utils import make_run_dir
from helpers.train_functs import train_model, test_model

def main(args):
    if args.cuda:
        # Set CUDA device
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')


    for args.model in args.models:
        
        resize = 64 if args.model == 'alexnet' and (args.resize is None or args.resize < 64) else args.resize
        # Do not need to resize ImageNet images if the model is AlexNet and the resize is less than 64
        resize = None if args.dataset == "ImageNet" and args.resize and args.resize < 224 else args.resize
        
        trainloader, testloader = get_dataset(args.dataset, args.batch_size, resize=resize)
        num_classes = 10 if args.dataset in ['CIFAR10', 'MNIST'] else 100 if args.dataset == 'CIFAR100' else 1000

        try:
            model = get_model(args.model, num_classes, args.pretrained).to(device)
            if args.replace_relu_with_selu:
                replace_relu_with_selu(model)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            path = "./model_weights/selu" if args.replace_relu_with_selu else "./model_weights"
            run_dir = make_run_dir(args, path)

            # Update the path for saving the best model
            best_model_path = os.path.join(run_dir, 'best_model.pt')

            best_accuracy = 0
            epochs_no_improve = 0

            print(f'Beginning training for {args.model} on {args.dataset}')
            for epoch in range(args.epochs):
                train_model(model, trainloader, criterion, optimizer, device, args.disable_loading_bar)
                test_loss, test_accuracy = test_model(model, testloader, criterion, device, args.disable_loading_bar)
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
            # Save the training parameters and accuracy to a CSV file
            csv_file = os.path.join(run_dir, f'training_results_{best_accuracy:.4f}.csv')
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Model', 'Dataset', 'Batch Size', 'Epochs', 'Learning Rate', 'Pretrained', 'Best Accuracy'])
                writer.writerow([args.model, args.dataset, args.batch_size, args.epochs, args.lr, args.pretrained, best_accuracy])
            # Save all arguments to a CSV file
            args_file = os.path.join(run_dir, 'training_args.csv')
            with open(args_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Argument', 'Value'])
                for arg in vars(args):
                    writer.writerow([arg, getattr(args, arg)])
        except Exception as e:
            print(f"Skipping {args.model} - An error occurred during training: {e}")
            continue
        
# nohup python train.py --models resnet18 alexnet efficientnet_v2 efficientnet mobilenet shufflenet googlenet inception densenet squeezenet vgg16 --dataset CIFAR100 --pretrained --device 2 --disable_loading_bar > output_logs/gpu2.log 2>&1 &
# nohup python train.py --models resnet18 alexnet efficientnet_v2 efficientnet mobilenet shufflenet googlenet inception densenet squeezenet vgg16 --dataset CIFAR10 --pretrained --device 7 --disable_loading_bar > output_logs/gpu7.log 2>&1 &
# nohup python train.py --models resnet18 alexnet efficientnet_v2 efficientnet mobilenet shufflenet googlenet inception densenet squeezenet vgg16 --dataset ImageNet --pretrained --device 3 --disable_loading_bar > output_logs/gpu3.log 2>&1 &
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a torchvision model on a dataset')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'MNIST', 'ImageNet'], help='Dataset to use')
    parser.add_argument('--models', type=str, nargs='+', default=['resnet18'], 
                        choices=['resnet18', 'alexnet', 'vgg16', 'squeezenet', 'densenet', 'inception', 'googlenet', 
                                 'shufflenet', 'mobilenet', 'resnext50_32x4d', 'wide_resnet50_2', 'mnasnet', 
                                 'efficientnet', 'efficientnet_v2'], help='List of models to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--plateau_patience', type=int, default=10, help='Patience for early stopping on plateau')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA if available')
    parser.add_argument('--device', type=str, default='4', help='CUDA device number')
    parser.add_argument('--resize', type=int, default=None, help='Resize the input image to the given size')
    parser.add_argument('--disable_loading_bar', action='store_true', help='Show loading bar during training and testing')
    parser.add_argument('--replace_relu_with_selu', type=bool, default=True, help='Replace ReLU activation with SELU')
    # Implement robust training for Gaussian and PGD attacks (Already started in the helpers/train_functs.py and utils.py)
    args = parser.parse_args()
    print(args)

    main(args)

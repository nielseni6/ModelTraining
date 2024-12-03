import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

def replace_relu_with_selu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.SELU(inplace=True))
        else:
            replace_relu_with_selu(child)

def replace_relu_with_leakyrelu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.LeakyReLU(inplace=True))
        else:
            replace_relu_with_leakyrelu(child)

def replace_relu_with_gelu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.GELU())
        else:
            replace_relu_with_gelu(child)

def replace_relu_with_any(module, activation):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, activation)
        else:
            replace_relu_with_any(child, activation)

def get_dataset(dataset_name, batch_size, resize=None):
    if dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Resize(resize) if resize else transforms.Lambda(lambda x: x),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Resize(resize) if resize else transforms.Lambda(lambda x: x),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            transforms.Resize(resize) if resize else transforms.Lambda(lambda x: x),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            transforms.Resize(resize) if resize else transforms.Lambda(lambda x: x),
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'MNIST':
        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize(resize) if resize else transforms.Lambda(lambda x: x),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize(resize) if resize else transforms.Lambda(lambda x: x),
        ])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    elif dataset_name == 'ImageNet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize(resize) if resize else transforms.Lambda(lambda x: x),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize(resize) if resize else transforms.Lambda(lambda x: x),
        ])
        trainset = torchvision.datasets.ImageFolder(root='/data/ImageNet/train', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root='/data/ImageNet/val', transform=transform_test)
    else:
        raise ValueError("Dataset not supported")
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def get_model(model_name, num_classes, pretrained=True):
    model_dict = {
        'resnet18': (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.DEFAULT),
        'alexnet': (torchvision.models.alexnet, torchvision.models.AlexNet_Weights.DEFAULT),
        'vgg16': (torchvision.models.vgg16, torchvision.models.VGG16_Weights.DEFAULT),
        'squeezenet': (torchvision.models.squeezenet1_0, torchvision.models.SqueezeNet1_0_Weights.DEFAULT),
        'densenet': (torchvision.models.densenet161, torchvision.models.DenseNet161_Weights.DEFAULT),
        'inception': (torchvision.models.inception_v3, torchvision.models.Inception_V3_Weights.DEFAULT),
        'googlenet': (torchvision.models.googlenet, torchvision.models.GoogLeNet_Weights.DEFAULT),
        'shufflenet': (torchvision.models.shufflenet_v2_x1_0, torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT),
        'mobilenet': (torchvision.models.mobilenet_v2, torchvision.models.MobileNet_V2_Weights.DEFAULT),
        'resnext50_32x4d': (torchvision.models.resnext50_32x4d, torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT),
        'wide_resnet50_2': (torchvision.models.wide_resnet50_2, torchvision.models.Wide_ResNet50_2_Weights.DEFAULT),
        'mnasnet': (torchvision.models.mnasnet1_0, torchvision.models.MNASNet1_0_Weights.DEFAULT),
        'efficientnet': (torchvision.models.efficientnet_b0, torchvision.models.EfficientNet_B0_Weights.DEFAULT),
        'efficientnet_v2': (torchvision.models.efficientnet_v2_s, torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)
    }
    
    if model_name not in model_dict:
        raise ValueError("Model not supported")
    
    model_fn, default_weights = model_dict[model_name]
    weights = default_weights if pretrained else None
    model = model_fn(weights=weights)
    
    if model_name in ['resnet18', 'resnext50_32x4d', 'wide_resnet50_2', 'googlenet']:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name in ['alexnet', 'vgg16', 'mobilenet', 'mnasnet', 'efficientnet', 'efficientnet_v2']:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == 'squeezenet':
        model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=(1, 1))
    elif model_name == 'densenet':
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'inception':
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'shufflenet':
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

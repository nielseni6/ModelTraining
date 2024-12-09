from tqdm import tqdm
import torch

def train_model(model, trainloader, criterion, optimizer, device, disable_loading_bar=False):
    """
    Trains the given model using the provided training data, loss function, and optimizer.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        trainloader (torch.utils.data.DataLoader): DataLoader for the training data.
        criterion (torch.nn.Module): Loss function to be used for training.
        optimizer (torch.optim.Optimizer): Optimizer to be used for updating model parameters.
        device (torch.device): Device on which to perform training (e.g., 'cpu' or 'cuda').
        disable_loading_bar (bool, optional): If True, disables the tqdm loading bar. Defaults to False.

    Returns:
        None
    """
    model.train()
    for inputs, labels in tqdm(trainloader, desc="Training", leave=False, disable=disable_loading_bar, mininterval=1.0):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def test_model(model, testloader, criterion, device, disable_loading_bar=True):
    """
    Evaluate the performance of a trained model on a test dataset.

    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        testloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function used to calculate the test loss.
        device (torch.device): Device on which to perform the evaluation (e.g., 'cpu' or 'cuda').
        disable_loading_bar (bool, optional): If True, disables the tqdm loading bar. Default is True.

    Returns:
        tuple: A tuple containing:
            - test_loss (float): The average loss over the test dataset.
            - accuracy (float): The accuracy of the model on the test dataset.
    """
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
    """
    Trains a given model using a robust training loop with noise injection.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        trainloader (torch.utils.data.DataLoader): DataLoader for the training data.
        criterion (torch.nn.Module): Loss function to be used for training.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        device (torch.device): Device on which the model and data should be loaded (e.g., 'cpu' or 'cuda').
        noise_funct (callable): Function to apply noise to the inputs during training.
        disable_loading_bar (bool, optional): If True, disables the tqdm loading bar. Defaults to False.

    Returns:
        None
    """
    model.train()
    for inputs, labels in tqdm(trainloader, desc="Training", leave=False, disable=disable_loading_bar, mininterval=1.0):
        inputs, labels = inputs.to(device), labels.to(device)
        # Apply noise to the inputs to make the model robust to noisy data
        inputs = noise_funct(inputs, labels)
        optimizer.zero_grad()
        outputs = model(inputs)
        # For GoogLeNet, use the logits attribute for the outputs, rather than default object output
        outputs = outputs.logits if model.__class__.__name__.lower() == 'googlenet' else outputs
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
import os
import torch
import random
import torch.nn as nn

def make_run_dir(args, base_dir):
    # Create directories for saving model weights
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

class GaussianNoise:
    def __init__(self, mean=0, std=0.05, random_std=False, low=0.01, high=0.1):
        self.mean = mean
        self.std = std
        self.random_std = random_std
        self.low = low
        self.high = high

    def __call__(self, image, **kwargs):
        if not self.random_std:
            noise = torch.normal(self.mean, self.std, image.size(), device=image.device)
        else:
            noise = torch.normal(self.mean, random.uniform(self.low, self.high), image.size(), device=image.device)
        return image + noise

# CIFAR10 eps = 8/255, steps = 7, alpha = 2/255
# MNIST eps = 0.3, steps = 40, alpha = 0.01
class PGDAttack:
    def __init__(self, model, loss, eps=8/255, alpha=2/255, iters=7, random_start=True):
        """
        Initialize the PGD attack.

        Parameters:
        model (torch.nn.Module): The model to attack.
        loss (torch.nn.Module): Loss function.
        eps (float): Maximum perturbation.
        alpha (float): Step size.
        iters (int): Number of iterations.
        random_start (bool): Whether to start with a random perturbation.
        """
        self.model = model
        self.loss = loss # nn.CrossEntropyLoss()
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.random_start = random_start

    def __call__(self, images, labels=None):
        """
        Perform the PGD attack on the input images.

        Parameters:
        images (torch.Tensor): Input images.
        labels (torch.Tensor): True labels of the input images.

        Returns:
        torch.Tensor: Adversarial images.
        """
        device = images.device if images.is_cuda else torch.device('cpu')
        images = images.clone().detach().to(device)
        if labels is not None:
            labels = labels.clone().detach().to(device)
        adv_images = images.clone().detach().to(device)
        
        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            min, max = float(images.min() - self.eps), float(images.max() + self.eps)
            adv_images = torch.clamp(adv_images, min=min, max=max).detach()

        for _ in range(self.iters):
            adv_images.requires_grad_(True)
            adv_images = adv_images.to(device)
            # optimizer.zero_grad()
            outputs = self.model(adv_images)

            if self.model.__class__.__name__.lower() == 'googlenet':
                outputs = outputs.logits

            if labels is None:
                _, labels = torch.max(outputs, 1)
                
            cost = self.loss(outputs, labels).to(device)


            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            min, max = float(images.min() - self.eps), float(images.max() + self.eps)
            adv_images = torch.clamp(images + delta, min=min, max=max).detach()

        return adv_images.to(device)
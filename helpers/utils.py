import os
import torch

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

def add_gaussian_noise(image, mean=0, std=1):
    """
    Generate Gaussian noise with the same size as the input image and add it to the image.

    Parameters:
    image (torch.Tensor): Input image.
    mean (float): Mean of the Gaussian noise.
    std (float): Standard deviation of the Gaussian noise.

    Returns:
    torch.Tensor: Image with added Gaussian noise.
    """
    noise = torch.normal(mean, std, image.size(), device=image.device)
    noisy_image = image + noise
    return noisy_image

# CIFAR10 eps = 8/255, steps = 7, alpha = 2/255
# MNIST eps = 0.3, steps = 40, alpha = 0.01
class PGDAttack:
    def __init__(self, model, loss, eps=8/255, alpha=2/255, iters=7):
        """
        Initialize the PGD attack.

        Parameters:
        model (torch.nn.Module): The model to attack.
        loss (torch.nn.Module): Loss function.
        eps (float): Maximum perturbation.
        alpha (float): Step size.
        iters (int): Number of iterations.
        """
        self.model = model
        self.loss = loss
        self.eps = eps
        self.alpha = alpha
        self.iters = iters

    def __call__(self, images, labels=None):
        """
        Perform the PGD attack on the input images.

        Parameters:
        images (torch.Tensor): Input images.
        labels (torch.Tensor): True labels of the input images.

        Returns:
        torch.Tensor: Adversarial images.
        """
        images = images.clone().detach().to(images.device)
        if labels is not None:
            labels = labels.clone().detach().to(labels.device)
        adv_images = images.clone().detach()
        
        for i in range(self.iters):
            adv_images.requires_grad_(True)
            outputs = self.model(adv_images)

            if labels is None:
                _, labels = torch.max(outputs, 1)
                
            cost = self.loss(outputs, labels).to(images.device)

            grad = torch.autograd.grad(cost, adv_images)[0]

            adv_images = adv_images + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
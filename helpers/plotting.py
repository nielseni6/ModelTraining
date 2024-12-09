import matplotlib.pyplot as plt
import numpy as np

def plot_image(image, fig_name='image.png'):
    # Assuming adv_images is a tensor, convert it to a NumPy array
    images_np = image.cpu().numpy()

    # Normalize the image to the range [0, 1]
    images_np = (images_np - images_np.min()) / (images_np.max() - images_np.min())

    # If adv_images contains multiple images, select one to save
    # For example, select the first image
    image_to_save = images_np[0]

    # If the image has multiple channels, transpose it to (height, width, channels)
    if image_to_save.shape[0] == 3:  # Assuming the image is in (channels, height, width) format
        image_to_save = np.transpose(image_to_save, (1, 2, 0))

    # Plot the image
    plt.imshow(image_to_save)
    plt.axis('off')  # Turn off axis

    # Save the image as a PNG file
    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
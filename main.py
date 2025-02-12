import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

import trainVAE


PATH : str = r"/mnt/c/Users/User/Desktop/datasets/CelebA/img_align_celeba/"
CSV_FILE : str = r"/mnt/c/Users/User/Desktop/datasets/CelebA/list_attr_celeba.csv"
IMG_FOLDER : str = r"/mnt/c/Users/User/Desktop/datasets/CelebA/img_align_celeba/img_align_celeba/"


def plot_label_clusters(encoder, data):
    # display a 2D plot of the digit classes in the latent space  
    
    for images, labels in data.take(data.__len__()):
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()
        
    z_mean,_,_ = encoder.predict(numpy_images, verbose=0)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=numpy_labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig("./clusters.png")
    

def generateSingleImage(model, sample=128):
    size = 128
    figsize=10
    
    image = np.zeros((size, size, 3)) 

    z = tf.random.uniform((1, sample))
    prediction = model.predict(z)
    image = tf.reshape(prediction[0], (size, size, 3))
    image = image.numpy()
    image = np.clip(image, 0, 1)
    
    plt.figure(figsize=(figsize, figsize))
    plt.imshow(image)
    plt.axis('OFF')
    plt.savefig("./image.png")

def plot_latent_space(decoder, n=30, figsize=15, latent_dim=2):
    """
    Plots a n*n 2D manifold of colored images from the VAE's latent space.

    Args:
        vae: The trained VAE model.
        n: Number of points along each axis of the latent space grid.
        figsize: Size of the matplotlib figure.
        latent_dim: Dimensionality of the latent space (default is 2).
    """
    digit_size = 128  # Assuming image size is 28x28
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n, 3))  # Add channel dimension for RGB
    
    # Linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            # Create a latent vector
            if latent_dim == 2:
                z = tf.constant([[xi, yi]])  # Shape: (1, 2)
            else:
                # If latent_dim > 2, fill the rest of the dimensions with zeros
                z = np.zeros((1, latent_dim))
                z[0, 0] = xi
                z[0, 1] = yi
                z = tf.constant(z)
            
            # Decode the latent vector to generate an image
            x_decoded = decoder.predict(z, verbose=0)
            
            # Reshape the decoded image to (28, 28, 3) for RGB
            if x_decoded.shape[-1] == 3:  # Check if the image has 3 color channels
                digit = x_decoded[0].reshape(digit_size, digit_size, 3)
            else:
                raise ValueError("Decoder output does not have 3 color channels. Expected shape: (batch_size, 28, 28, 3)")
            
            # Place the decoded digit in the figure
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
                :  # Include all color channels
            ] = digit

    # Plot the figure
    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure)  # No need for cmap="Greys_r" for colored images
    plt.savefig("./latent_space.png")
  
def main():
    model_path = "attempts/latent64_lr0.0005_epochs20"
    decoder_path = os.path.join(model_path, f"decoder_model/model.keras")
    encoder_path = os.path.join(model_path, f"encoder_model/model.keras")
    
    LATENT = 64
    decoder = tf.keras.models.load_model(decoder_path, compile=True)
    generateSingleImage(decoder, LATENT)
    
    encoder = tf.keras.models.load_model(encoder_path, compile=False)
    n = 9000
    fata= trainVAE.dataSetGeneratorFromCSV(CSV_FILE, IMG_FOLDER, n, n, 'Male')
    plot_label_clusters(encoder, fata)
    
    plot_latent_space(decoder,20, figsize=15, latent_dim=LATENT)

if __name__ == "__main__":
    main()

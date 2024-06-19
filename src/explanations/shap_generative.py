import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image



def generate_perturbations(image, num_perturbations=100):
    """Generate perturbations using the GAN."""
    perturbations = []
    for _ in range(num_perturbations):
        # Generate random latent vector
        latent_vector = tf.random.normal([1, 512])
        # Generate image using GAN
        generated_image = gan(latent_vector)
        generated_image = tf.image.resize(generated_image, [image.shape[0], image.shape[1]])
        perturbations.append(generated_image.numpy().squeeze())
    return np.array(perturbations)

def perturb_fn(images):
    """Function to apply perturbations and return predictions."""
    perturbations = []
    for img in images:
        perturbed_images = generate_perturbations(img)
        perturbations.append(perturbed_images)
    return np.array(perturbations)

# Load and preprocess your image
def load_image(path):
    img = Image.open(path).resize((128, 128))  # Resize as per GAN model's requirement
    img = np.array(img) / 255.0  # Normalize the image
    return img

# Dummy prediction function (replace with your model's prediction function)
def predict(images):
    # Assuming model.predict returns a probability distribution
    return np.random.rand(images.shape[0], 10)




if __name__ == "__main__":
    # Load pre-trained GAN from TensorFlow Hub
    gan_url = 'https://tfhub.dev/google/progan-128/1'  # Example URL, replace with the GAN model of your choice
    gan = hub.load(gan_url)

    # Initialize LIME image explainer
    explainer = lime_image.LimeImageExplainer()

    # Path to your image
    image_path = 'path_to_your_image.jpg'
    image = load_image(image_path)

    # Explain the prediction using LIME with GAN perturbations
    explanation = explainer.explain_instance(image, predict, hide_color=0, num_samples=1000, segmentation_fn=None, random_seed=42, pertubation_function=perturb_fn)

    # Visualize the explanation
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True
    )
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.show()
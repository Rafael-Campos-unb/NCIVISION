import pandas as pd
import numpy as np
import torch
from skimage.io import imread, imsave
from skimage.filters import sobel
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import resize
import os
import matplotlib.pyplot as plt
from collections import Counter
import cv2
from SimilarityCNN import SiameseNetwork


def image_to_glcm(image_path, levels=8):
    if not isinstance(image_path, str):
        raise TypeError(f"Expected path as string but received: {type(image_path)} -> {image_path}")

    # Loading image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"[ERROR] Image not loaded: {image_path}")

    # Normalize to int value between 0 and - 1 levels
    img = np.uint8((img / 255.0) * (levels - 1))

    # Generate GLCM
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)

    # Return GLCM flatten
    return glcm.flatten()


class GLCM_data:
    def __init__(self, dirpath, angles, levels):
        self.top_embedding_indices = None
        self.embedding_attention_map = None
        self.dirpath = dirpath
        self.angles = angles
        self.levels = levels
        self.distances = [1]
        self.dataset = None
        self.diagonal_coords = []

    def GLCM(self, image_path):
        img_grayscale = imread(image_path, as_gray=True)
        img_grayscale_uint8 = (img_grayscale * 255).astype(np.uint8)
        filename = f'{os.path.basename(image_path)}RDG_GRAY.png'
        filepath = os.path.join(self.dirpath, filename)
        imsave(filepath, img_grayscale_uint8)
        un_int_2d_array = (img_grayscale * (self.levels - 1)).astype(np.uint8)
        glcm = graycomatrix(
            un_int_2d_array,
            distances=self.distances,
            angles=self.angles,
            levels=self.levels,
            symmetric=True,
            normed=True
        )
        return glcm

    def plot_glcm_attention_heatmap(self, image_path, saliency):
        plt.figure(figsize=(6, 5))
        img = plt.imshow(saliency, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
        cbar = plt.colorbar(img)
        cbar.ax.yaxis.label.set_fontsize(25)
        cbar.ax.tick_params(labelsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.tight_layout()
        filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_SalienceHeatmap.png"
        filepath = os.path.join(self.dirpath, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()

    def plot_glcm_with_model_attention(self, image_path, model):
        glcm = self.GLCM(image_path)  # shape [levels, levels, 1, 1]

        # Convert to tensor and prepare entry
        glcm_tensor = torch.tensor(glcm, dtype=torch.float32).permute(2, 3, 0, 1)  # -> [1, 1, 8, 8]
        glcm_tensor = glcm_tensor.to(next(model.parameters()).device)
        glcm_tensor.requires_grad_()  # activate gradient calculations

        # Flatten -> submodel
        glcm_flat = glcm_tensor.view(glcm_tensor.size(0), -1)

        # Pass through embedder
        model.eval()
        output = model.glcm_embedder(glcm_flat)

        importance = output.sum()
        model.zero_grad()  # clean previous gradients
        importance.backward()

        # Calculate saliency maps (gradients of entries)
        if glcm_tensor.grad is not None:
            saliency = glcm_tensor.grad.detach().squeeze().cpu().numpy()

            # Normalization
            saliency = np.maximum(saliency, 0)
            if saliency.max() > 0:
                saliency /= saliency.max()

            self.embedding_attention_map = saliency
            self.top_embedding_indices = np.unravel_index(np.argsort(saliency.ravel())[-3:], saliency.shape)

            # Plot original image with attention
            self.plot_attention_on_rdg(image_path)

            # Plot saliency heatmap
            self.plot_glcm_attention_heatmap(image_path, saliency)

    def plot_attention_on_rdg(self, image_path):
        img = imread(image_path, as_gray=True)
        img_shape = img.shape  # e.g., (256, 256)

        # Normalize image to [0, 1]
        img = img / 255.0 if img.max() > 1 else img

        saliency_resized = resize(
            self.embedding_attention_map,
            img_shape,
            order=3,  # bicubic
            mode='reflect',
            anti_aliasing=True
        )

        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        plt.imshow(saliency_resized, cmap='hot', alpha=0.5)  # sobreposition red/yellow

        plt.axis('off')
        filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_RDG_ATTENTION.png"
        plt.savefig(os.path.join(self.dirpath, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def calculate_rdg_gradient(self, image_path):
        img = imread(image_path, as_gray=True)
        gradient = sobel(img)
        return np.mean(gradient)

    def process_images(self, model=None):
        contrast_list = []
        homogeneity_list = []
        energy_list = []
        correlation_list = []
        gradient_list = []
        image_names = []

        for image_name in os.listdir(self.dirpath):
            if image_name.endswith('.jpg'):
                image_path = os.path.join(self.dirpath, image_name)
                name_no_ext = image_name.replace('.jpg', '')
                image_names.append(name_no_ext)

                # GLCM
                glcm = self.GLCM(image_path)

                self.plot_glcm_with_model_attention(image_path, model)

                contrast = graycoprops(glcm, 'contrast')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                correlation = graycoprops(glcm, 'correlation')[0, 0]
                gradient = self.calculate_rdg_gradient(image_path)

                contrast_list.append(contrast)
                homogeneity_list.append(homogeneity)
                energy_list.append(energy)
                correlation_list.append(correlation)
                gradient_list.append(gradient)

        data = {
            'Contrast': contrast_list,
            'Homogeneity': homogeneity_list,
            'Energy': energy_list,
            'Correlation': correlation_list,
            'Gradient': gradient_list
        }

        self.dataset = pd.DataFrame(data, index=image_names)
        self.dataset.to_csv('dataset.csv')
        return self.dataset

    def getMean(self):
        return self.dataset.mean()

    def getSdv(self):
        return self.dataset.std()


if __name__ == '__main__':
    dirpath = './NCIs'
    angles = [0]
    levels = 8

    glcm_data = GLCM_data(dirpath, angles, levels)

    model_path = "./models/siamese_model_calculated_similarity.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Generating dataset and images with attention in embeddings")
    dataset = glcm_data.process_images(model=model)

    print("\nDataset created:")
    print(dataset)

    print("\nMean of each column:")
    print(glcm_data.getMean())

    print("\nStandard deviation of each column:")
    print(glcm_data.getSdv())







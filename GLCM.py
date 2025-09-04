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
        raise TypeError(f"Esperado caminho como string, mas recebido: {type(image_path)} -> {image_path}")

    # Carregar a imagem em escala de cinza
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"[ERRO] Imagem não carregada: {image_path}")

    # Normalizar para valores inteiros entre 0 e levels - 1
    img = np.uint8((img / 255.0) * (levels - 1))

    # Gerar GLCM
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)

    # Retornar GLCM flatten
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
        imsave(f'{os.path.basename(image_path)}RDG_GRAY.png', img_grayscale_uint8)
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
        # plt.title(f'Importância do Modelo no GLCM\n{os.path.basename(image_path)}')
        plt.tight_layout()
        filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_GLCM_ATTENTION.png"
        # plt.show()
        # save_path = os.path.join(self.dirpath, filename)
        plt.savefig(f'{os.path.basename(image_path)}.png', dpi=300)
        plt.close()

    def plot_glcm_with_model_attention(self, image_path, model):
        glcm = self.GLCM(image_path)  # shape [levels, levels, 1, 1]

        # Converte para tensor e prepara como entrada
        glcm_tensor = torch.tensor(glcm, dtype=torch.float32).permute(2, 3, 0, 1)  # -> [1, 1, 8, 8]
        glcm_tensor = glcm_tensor.to(next(model.parameters()).device)
        glcm_tensor.requires_grad_()  # ativa cálculo de gradiente

        # Flatten para enviar para o submodelo
        glcm_flat = glcm_tensor.view(glcm_tensor.size(0), -1)

        # Passa pelo embedder
        model.eval()
        output = model.glcm_embedder(glcm_flat)

        # Garante que seja escalar antes do backward
        importance = output.sum()
        model.zero_grad()  # limpa gradientes anteriores
        importance.backward()

        # Calcula mapa de saliência (gradiente da entrada)
        if glcm_tensor.grad is not None:
            saliency = glcm_tensor.grad.detach().squeeze().cpu().numpy()

            # Normalização
            saliency = np.maximum(saliency, 0)
            if saliency.max() > 0:
                saliency /= saliency.max()

            # Armazena para uso posterior
            self.embedding_attention_map = saliency
            self.top_embedding_indices = np.unravel_index(np.argsort(saliency.ravel())[-3:], saliency.shape)

            # Plot da imagem original com atenção
            self.plot_attention_on_rdg(image_path)

            # Novo: salva o heatmap também!
            self.plot_glcm_attention_heatmap(image_path, saliency)

    def plot_attention_on_rdg(self, image_path):
        # Carrega a imagem RDG em escala de cinza
        img = imread(image_path, as_gray=True)
        img_shape = img.shape  # e.g., (256, 256)

        # Normaliza a imagem para [0, 1] para melhor sobreposição
        img = img / 255.0 if img.max() > 1 else img

        # Upsample do mapa de atenção GLCM para o tamanho da imagem
        saliency_resized = resize(
            self.embedding_attention_map,
            img_shape,
            order=3,  # bicubic
            mode='reflect',
            anti_aliasing=True
        )

        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray')
        plt.imshow(saliency_resized, cmap='hot', alpha=0.5)  # sobreposição em vermelho/amarelo

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
        # self.dataset.to_csv('dataset.csv')
        # self.dataset.to_csv('datasetAChEI.csv')
        self.dataset.to_csv('datasetAChEICannabinoids.csv')
        return self.dataset

    def getMean(self):
        return self.dataset.mean()

    def getSdv(self):
        return self.dataset.std()


if __name__ == '__main__':
    # dirpath = 'C:/NCIVISION/NCI_BCRABL_cropped/'
    # dirpath = 'C:/NCIVISION/NCI_AChEI_cropped'
    dirpath = 'C:/NCIVISION/NCIs'
    angles = [0]
    levels = 8

    glcm_data = GLCM_data(dirpath, angles, levels)

    # model_path = "NCIVISION.pth"
    # model_path = "NCIVISION_AChEI.pth"
    model_path = "NCIVISION_AChEICannabinoids.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Gerando dataset e imagens com atenção nos embeddings...")
    dataset = glcm_data.process_images(model=model)

    print("\nDataset gerado:")
    print(dataset)

    print("\nMédia de cada coluna:")
    print(glcm_data.getMean())

    print("\nDesvio padrão de cada coluna:")
    print(glcm_data.getSdv())







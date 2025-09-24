import torch
import torch.nn as nn


class CNN_MEPFeatures(nn.Module):
    def __init__(self, output_dim):
        super(CNN_MEPFeatures, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),      # 16 → 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),     # 32 → 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),    # 64 → 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.f = nn.Linear(128, output_dim)  

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.f(x)
        return x


class GLCMEmbedder(nn.Module):
    def __init__(self, glcm_dim=64, output_dim=64):  # 32 → 64
        super(GLCMEmbedder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(glcm_dim, 128),  # 64 → 128
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class SiameseNetwork(nn.Module):
    def __init__(self, cnn_output_dim=128, glcm_output_dim=64, final_embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.cnn = CNN_MEPFeatures(output_dim=cnn_output_dim)
        self.glcm_embedder = GLCMEmbedder(glcm_dim=64, output_dim=glcm_output_dim)
        self.ff = nn.Sequential(
            nn.Linear(cnn_output_dim + glcm_output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, final_embedding_dim),
            nn.Dropout(0.3)
        )

    def forward_once(self, mep_img, glcm_feat):
        cnn_out = self.cnn(mep_img)
        glcm_out = self.glcm_embedder(glcm_feat)
        combined = torch.cat([cnn_out, glcm_out], dim=1)
        embedding = self.ff(combined)
        return embedding

    def forward(self, anchor_mep, anchor_glcm, positive_mep, positive_glcm, negative_mep, negative_glcm):
        anchor_embed = self.forward_once(anchor_mep, anchor_glcm)
        positive_embed = self.forward_once(positive_mep, positive_glcm)
        negative_embed = self.forward_once(negative_mep, negative_glcm)
        return anchor_embed, positive_embed, negative_embed

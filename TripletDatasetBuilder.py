import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
import logging
import torchvision.transforms as transforms
from GLCM import image_to_glcm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_molecule_mask(image_array: np.ndarray) -> np.ndarray:
    """
    Applies a mask to the image to focus on the molecule and remove noise.

    Args:
        image_array (np.ndarray): The input image as a numpy array in the [0, 1] range.

    Returns:
        np.ndarray: The image with the mask applied.
    """
    masked_image = image_array.copy()
    white_threlshold = 0.95
    masked_image[image_array >= white_threlshold] = 0.0
    return masked_image


class TripletDataset(Dataset):
    """
    Dataset class for triplet learning, faithfully recreating the original data pipeline.
    """

    def __init__(self,
                 triplet_list: List[Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]],
                 validate_files: bool = True,
                 target_size: Tuple[int, int] = (64, 64),
                 glcm_levels: int = 8):
        """
        Initializes the triplet dataset.
        """
        if not triplet_list:
            raise ValueError("Triplet list cannot be empty")

        self.triplet_list = triplet_list
        self.target_size = target_size
        self.glcm_levels = glcm_levels

        # Transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.target_size, antialias=True),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Initial validations
        if validate_files:
            self._validate_file_existence()
        self._validate_triplet_structure()
        logger.info(f"Dataset initialized with {len(self.triplet_list)} triplets.")
        logger.info(f"Image (MEP) pipeline: Resize to {target_size}, Normalize to [-1, 1].")

    def _validate_triplet_structure(self):
        """Validates that all triplets have the correct structure (dicts with 'mep' and 'glcm' keys)."""
        for i, triplet in enumerate(self.triplet_list):
            if len(triplet) != 3:
                raise ValueError(f"Triplet {i} must have 3 elements (anchor, positive, negative).")
            for name, element in [("anchor", triplet[0]), ("positive", triplet[1]), ("negative", triplet[2])]:
                if not isinstance(element, dict) or 'mep' not in element or 'glcm' not in element:
                    raise ValueError(
                        f"The '{name}' element of triplet {i} must be a dictionary with 'mep' and 'glcm' keys.")

    def _validate_file_existence(self):
        """Validates the existence of all referenced image files."""
        missing_files = []
        for triplet in self.triplet_list:
            for element in triplet:
                for path in element.values():
                    if not os.path.exists(path):
                        missing_files.append(path)

        if missing_files:
            unique_missing = list(set(missing_files))
            error_msg = f"Image files not found:\n" + "\n".join(unique_missing[:10])
            if len(unique_missing) > 10:
                error_msg += f"\n... and {len(unique_missing) - 10} more files."
            raise FileNotFoundError(error_msg)

    def __len__(self) -> int:
        """Returns the number of triplets in the dataset."""
        return len(self.triplet_list)

    def _load_mep_image(self, image_path: str) -> torch.Tensor:
        """Loads and preprocesses the MEP image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"MEP image file not found: {image_path}")

        try:
            # 1. Read as grayscale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # 2. Convert to float and normalize to [0, 1]
            image = image.astype(np.float32) / 255.0

            # 3. Apply the molecule mask
            image = apply_molecule_mask(image)

            # 4. Add channel dimension for ToTensor compatibility
            image = np.expand_dims(image, axis=-1)  # Shape (H, W, 1)

            # 5. Apply the transformation pipeline (ToTensor, Resize, Normalize)
            image_tensor = self.transform(image)
            return image_tensor

        except Exception as e:
            raise ValueError(f"Error loading MEP image {image_path}: {e}")

    def _load_glcm_features(self, image_path: str) -> torch.Tensor:
        """Loads GLCM features, validating their size without altering them."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"GLCM image file not found: {image_path}")

        try:
            glcm_features = image_to_glcm(image_path, levels=self.glcm_levels)

            # Validate feature size. If incorrect, throw a clear error.
            if len(glcm_features) != 64:
                raise ValueError(f"Inconsistent GLCM feature size for {image_path}. "
                                 f"Expected: 64, Got: {len(glcm_features)}. "
                                 f"Please fix the `image_to_glcm` function.")

            feature_tensor = torch.from_numpy(glcm_features).float()
            return feature_tensor

        except Exception as e:
            raise ValueError(f"Error loading GLCM features from {image_path}: {e}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Returns a complete triplet (anchor, positive, negative)."""
        if idx >= len(self.triplet_list):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.triplet_list)}")

        try:
            anchor, positive, negative = self.triplet_list[idx]

            # Load MEP images
            anchor_mep = self._load_mep_image(anchor['mep'])
            pos_mep = self._load_mep_image(positive['mep'])
            neg_mep = self._load_mep_image(negative['mep'])

            # Load GLCM features
            anchor_glcm = self._load_glcm_features(anchor['glcm'])
            pos_glcm = self._load_glcm_features(positive['glcm'])
            neg_glcm = self._load_glcm_features(negative['glcm'])

            return anchor_mep, anchor_glcm, pos_mep, pos_glcm, neg_mep, neg_glcm

        except Exception as e:
            logger.error(f"Critical error while loading triplet {idx}: {e}")
            raise  # Re-raise the exception to stop training, as it's a data issue.

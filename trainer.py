import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional, Any
import time
import os
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from TripletDatasetBuilder import TripletDataset
from SimilarityCNN import SiameseNetwork
from params import ModelParams


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 20
    margin: float = 1.0
    early_stop_threshold: Optional[float] = 0.01
    test_size: float = 0.3
    random_state: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    device: str = 'auto'

    def __post_init__(self):
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class ModelPaths:
    """Configuration class for model paths and file names"""
    model_save_path: str = './models/'
    plot_save_path: str = './plots/'
    model_name: str = 'siamese_model'
    experiment_name: str = 'default_experiment'

    def __post_init__(self):
        # Create directories if they don't exist
        Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        Path(self.plot_save_path).mkdir(parents=True, exist_ok=True)

    @property
    def model_file(self) -> str:
        return os.path.join(self.model_save_path, f'{self.model_name}_{self.experiment_name}.pth')

    @property
    def plot_file(self) -> str:
        return os.path.join(self.plot_save_path, f'training_loss_{self.experiment_name}.png')


class SiameseTrainer:
    """
    Trainer class for Siamese Networks with triplet loss
    """

    def __init__(self,
                 config: TrainingConfig,
                 paths: ModelPaths,
                 model: Optional[nn.Module] = None):
        """
        Initialize trainer

        Args:
            config: Training configuration
            paths: Model and output paths configuration
            model: Pre-initialized model (optional)
        """
        self.config = config
        self.paths = paths
        self.device = torch.device(config.device)

        # Initialize model
        self.model = model if model is not None else SiameseNetwork()
        self.model.to(self.device)

        # Initialize loss function and optimizer
        self.loss_fn = nn.TripletMarginLoss(margin=config.margin, p=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # Training history
        self.train_losses = []
        self.val_losses = []

        print(f"✅ Trainer initialized on device: {self.device}")

    def prepare_data(self, tripletlist: List[Tuple]) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders

        Args:
            tripletlist: List of triplets for training

        Returns:
            Tuple of (train_loader, val_loader)
        """
        print(f"📊 Preparing data from {len(tripletlist)} triplets...")

        # Split data
        train_triplets, val_triplets = train_test_split(
            tripletlist,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )

        print(f"   Training triplets: {len(train_triplets)}")
        print(f"   Validation triplets: {len(val_triplets)}")

        # Create datasets
        train_dataset = TripletDataset(train_triplets)
        val_dataset = TripletDataset(val_triplets)

        # Create DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            anchor_mep, anchor_glcm, pos_mep, pos_glcm, neg_mep, neg_glcm = [
                x.to(self.device) for x in batch
            ]

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            anchor_out, pos_out, neg_out = self.model(
                anchor_mep, anchor_glcm, pos_mep, pos_glcm, neg_mep, neg_glcm
            )

            # Calculate loss
            loss = self.loss_fn(anchor_out, pos_out, neg_out)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def validate_epoch(self, val_loader: DataLoader, silent: bool = True) -> float:
        """
        Validate for one epoch

        Args:
            val_loader: Validation data loader
            silent: Whether to suppress output

        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                anchor_mep, anchor_glcm, pos_mep, pos_glcm, neg_mep, neg_glcm = [
                    x.to(self.device) for x in batch
                ]

                # Forward pass
                anchor_out, pos_out, neg_out = self.model(
                    anchor_mep, anchor_glcm, pos_mep, pos_glcm, neg_mep, neg_glcm
                )

                # Calculate loss
                loss = self.loss_fn(anchor_out, pos_out, neg_out)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if not silent:
            print(f"   Validation Loss: {avg_val_loss:.4f}")

        return avg_val_loss

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Complete training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training results dictionary
        """
        print(f"🚀 Starting training for {self.config.epochs} epochs...")
        start_time = time.time()

        torch.manual_seed(self.config.random_state)

        best_val_loss = float('inf')
        early_stopped = False

        for epoch in range(self.config.epochs):
            # Train epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate epoch
            val_loss = self.validate_epoch(val_loader, silent=True)
            self.val_losses.append(val_loss)

            # Track best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            print(f"Epoch [{epoch + 1}/{self.config.epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if (self.config.early_stop_threshold is not None and
                    train_loss <= self.config.early_stop_threshold):
                print(f"Stopping early: loss {train_loss:.4f} < threshold {self.config.early_stop_threshold}")
                early_stopped = True
                break

        training_time = time.time() - start_time

        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"   Best validation loss: {best_val_loss:.4f}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'early_stopped': early_stopped,
            'epochs_completed': len(self.train_losses)
        }

    def plot_training_curves(self, save_plot: bool = True, show_plot: bool = True):
        """
        Plot training and validation loss curves

        Args:
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
        """
        if not self.train_losses:
            print("⚠️ No training data to plot")
            return

        plt.figure(figsize=(8, 6))  
        epochs = range(1, len(self.train_losses) + 1)

        plt.plot(epochs, self.train_losses, marker='o', label='Training Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, marker='s', label='Validation Loss', linewidth=2)

        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.xticks(epochs, fontsize=12)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        if save_plot:
            plot_filename = 'triplet_loss_train_val_plot.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {plot_filename}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def save_model(self, additional_info: Optional[Dict] = None):
        """
        Save the trained model

        Args:
            additional_info: Additional information to save with the model
        """
        torch.save(self.model.state_dict(), 'NCIVISION.pth')
        print(f"💾 Model saved to: NCIVISION.pth")

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        if additional_info:
            save_dict.update(additional_info)

        torch.save(save_dict, self.paths.model_file)
        print(f"💾 Complete model saved to: {self.paths.model_file}")

    def load_model(self, model_path: str):
        """
        Load a trained model

        Args:
            model_path: Path to the saved model
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle both simple state_dict and full checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
            if 'val_losses' in checkpoint:
                self.val_losses = checkpoint['val_losses']
        else:
            # Simple state_dict
            self.model.load_state_dict(checkpoint)

        print(f"📥 Model loaded from: {model_path}")


def train_molecular_similarity_model(
        dataset_path: str,
        smiles_col: str,
        molecule_col: str,
        anchor_molecule: str,
        experiment_name: str,
        positive_threshold: float = 0.70,
        negative_threshold: float = 0.40,
        base_mep_path: str = './MEP/MEP{}.jpg',
        base_glcm_path: str = './NCIs/{}.jpg',
        config: Optional[TrainingConfig] = None
) -> SiameseTrainer:
    """
    Complete training pipeline from molecular similarity data

    Args:
        dataset_path: Path to the molecular dataset CSV
        smiles_col: Name of the SMILES column
        molecule_col: Name of the molecule name column
        anchor_molecule: Name of the anchor molecule
        experiment_name: Name for this experiment
        positive_threshold: Minimum threshold for positive molecules
        negative_threshold: Maximum threshold for negative molecules
        base_mep_path: Template path for MEP images
        base_glcm_path: Template path for GLCM images
        config: Training configuration (optional)

    Returns:
        Trained SiameseTrainer instance
    """
    # Default configuration
    if config is None:
        config = TrainingConfig()

    # Create paths configuration
    paths = ModelPaths(experiment_name=experiment_name)

    # Load and process molecular data
    print(f"📂 Loading dataset from: {dataset_path}")
    dataset = pd.read_csv(dataset_path)
    print(f"   Dataset shape: {dataset.shape}")

    # Create ModelParams instance
    model_params = ModelParams(dataset, smiles_col, molecule_col)

    # Calculate molecular fingerprints and similarity
    print("🧬 Calculating molecular fingerprints...")
    model_params.MorganFingerprints()

    print("📊 Calculating Tanimoto similarity matrix...")
    model_params.TanimotoSimMatrix()

    # Generate triplet list
    print(f"🔗 Generating triplets with anchor: {anchor_molecule}")
    tripletlist = model_params.generate_tripletlist_from_similarity(
        anchor_molecule=anchor_molecule,
        base_mep_path=base_mep_path,
        base_glcm_path=base_glcm_path,
        positive_threshold=positive_threshold,
        negative_threshold=negative_threshold
    )

    print(f"✅ Generated {len(tripletlist)} triplets")

    if len(tripletlist) == 0:
        raise ValueError("No triplets generated! Check your thresholds and anchor molecule.")

    # Initialize trainer
    trainer = SiameseTrainer(config, paths)

    # Prepare data
    train_loader, val_loader = trainer.prepare_data(tripletlist)

    # Train model
    results = trainer.train(train_loader, val_loader)

    # Plot results
    trainer.plot_training_curves()

    # Save model
    trainer.save_model(additional_info={
        'results': results,
        'anchor_molecule': anchor_molecule,
        'positive_threshold': positive_threshold,
        'negative_threshold': negative_threshold,
        'dataset_info': {
            'path': dataset_path,
            'smiles_col': smiles_col,
            'molecule_col': molecule_col,
            'shape': dataset.shape
        }
    })

    return trainer


if __name__ == '__main__':
    multiprocessing.freeze_support()

    print("🚀 Trainer module loaded successfully!")
    print("Use train_molecular_similarity_model() to start training.")
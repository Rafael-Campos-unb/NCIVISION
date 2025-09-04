# Example with calculated dataset similarity
import pandas as pd
import multiprocessing
from trainer import TrainingConfig, ModelPaths, SiameseTrainer
from params import ModelParams


def main():
    # 1. Load dataset
    dataset = pd.read_csv('ExampleDataset.csv')

    # 2. Calculate molecular similarity
    model_params = ModelParams(dataset, 'SMILES', 'MOLECULE')
    model_params.MorganFingerprints()
    model_params.TanimotoSimMatrix()

    # 3. Generate triplets
    tripletlist = model_params.generate_tripletlist_from_similarity(
        anchor_molecule='pona',
        positive_threshold=0.70,
        negative_threshold=0.40
    )

    # 4. Configure training
    config = TrainingConfig(
        epochs=20,
        batch_size=64,
        learning_rate=1e-3,
        margin=1.0,
        early_stop_threshold=0.01,
        num_workers=4,
        pin_memory=True
    )
    paths = ModelPaths(experiment_name='calculated_similarity')

    # 5. Train model
    trainer = SiameseTrainer(config, paths)
    train_loader, val_loader = trainer.prepare_data(tripletlist)
    results = trainer.train(train_loader, val_loader)

    # 6. Save results
    trainer.plot_training_curves()
    trainer.save_model()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
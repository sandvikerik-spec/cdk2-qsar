"""
train_gnn.py - Entry point for CDK2 GNN training pipeline.
Usage: python scripts/train_gnn.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import pandas as pd
import os
from torch_geometric.loader import DataLoader

from src.data import load_or_fetch_data, scaffold_split
from src.features import smiles_to_graphs
from src.model import CDK2GNN
from src.train import train, evaluate


def main():
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {config['model']}\n")

    # Data
    df = load_or_fetch_data(config)
    train_df, test_df = scaffold_split(
        df,
        config["data"]["test_size"],
        config["data"]["random_seed"]
    )

    # Convert to graphs
    print("\nConverting molecules to graphs...")
    train_graphs, _ = smiles_to_graphs(train_df)
    test_graphs,  _ = smiles_to_graphs(test_df)

    # Data loaders
    train_loader = DataLoader(
        train_graphs,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )
    test_loader = DataLoader(
        test_graphs,
        batch_size=64,
        shuffle=False
    )

    # Model
    model = CDK2GNN(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Train
    print()
    model, history = train(model, train_loader, test_loader, config, device)

    # Final evaluation
    test_metrics = evaluate(model, test_loader, device)
    print("\n" + "=" * 55)
    print("FINAL TEST RESULTS (scaffold split)")
    print("=" * 55)
    print(f"  R2:        {test_metrics['r2']:.3f}")
    print(f"  RMSE:      {test_metrics['rmse']:.3f} pIC50 units")
    print(f"  Pearson r: {test_metrics['pearson']:.3f}")
    print("=" * 55)

    # Save model and history
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), config["training"]["checkpoint_path"])
    print(f"\nModel saved to {config['training']['checkpoint_path']}")

    history_df = pd.DataFrame(history)
    history_df.to_csv("models/training_history.csv", index=False)
    print("Training history saved to models/training_history.csv")


if __name__ == "__main__":
    main()

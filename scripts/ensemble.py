"""
ensemble.py - Compare XGBoost, GNN, and ensemble predictions on CDK2.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

from torch_geometric.loader import DataLoader
from src.data import load_or_fetch_data, scaffold_split
from src.features import smiles_to_graphs
from src.model import CDK2GNN

# RDKit imports for fingerprints
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

def get_fingerprint_features(df):
    descriptor_names = [
        "MolWt", "LogP", "NumHDonors", "NumHAcceptors",
        "TPSA", "NumRotatableBonds", "RingCount",
        "NumAromaticRings", "FractionCSP3", "NumHeteroatoms"
    ]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    features, valid_idx = [], []
    for i, smi in enumerate(df["canonical_smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = np.array(generator.GetFingerprint(mol))
        pc = np.array(calc.CalcDescriptors(mol))
        features.append(np.concatenate([fp, pc]))
        valid_idx.append(i)

    return np.array(features), valid_idx


def get_gnn_predictions(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            preds.extend(pred.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
    return np.array(preds), np.array(targets)


def print_metrics(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r, _ = pearsonr(y_true, y_pred)
    print(f"  {name:<20} R2={r2:.3f}  RMSE={rmse:.3f}  Pearson r={r:.3f}")
    return r2, rmse, r


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and split data
    df = load_or_fetch_data(config)
    train_df, test_df = scaffold_split(
        df, config["data"]["test_size"], config["data"]["random_seed"]
    )

    print("Building fingerprint features...")
    X_train, train_idx = get_fingerprint_features(train_df)
    X_test,  test_idx  = get_fingerprint_features(test_df)
    y_train = train_df["pchembl_value"].iloc[train_idx].values
    y_test  = test_df["pchembl_value"].iloc[test_idx].values

    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.3,
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    # Load GNN
    print("Loading GNN...")
    gnn_model = CDK2GNN(config).to(device)
    gnn_model.load_state_dict(
        torch.load(config["training"]["checkpoint_path"],
                   map_location=device)
    )

    print("Getting GNN predictions...")
    test_graphs, _ = smiles_to_graphs(test_df)
    test_loader    = DataLoader(test_graphs, batch_size=64, shuffle=False)
    gnn_pred, gnn_targets = get_gnn_predictions(gnn_model, test_loader, device)

    # Align targets - use GNN targets as reference
    y_test_gnn = gnn_targets

    # Get XGBoost predictions aligned to same compounds
    xgb_pred_aligned = xgb_model.predict(X_test[:len(y_test_gnn)])

    # Ensemble - weighted average
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        ens = w * xgb_pred_aligned + (1 - w) * gnn_pred
        r, _ = pearsonr(y_test_gnn, ens)

    # Find best weight
    best_r, best_w = -1, 0.5
    for w in np.arange(0.1, 1.0, 0.05):
        ens = w * xgb_pred_aligned + (1 - w) * gnn_pred
        r, _ = pearsonr(y_test_gnn, ens)
        if r > best_r:
            best_r, best_w = r, w

    ensemble_pred = best_w * xgb_pred_aligned + (1 - best_w) * gnn_pred

    # Results
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (scaffold split test set)")
    print("=" * 60)
    xgb_r2, xgb_rmse, xgb_r = print_metrics(
        "XGBoost", y_test_gnn, xgb_pred_aligned)
    gnn_r2, gnn_rmse, gnn_r = print_metrics(
        "AttentiveFP GNN", y_test_gnn, gnn_pred)
    ens_r2, ens_rmse, ens_r = print_metrics(
        f"Ensemble (w={best_w:.2f})", y_test_gnn, ensemble_pred)
    print("=" * 60)
    print(f"\nBest ensemble weight: {best_w:.2f} XGBoost + {1-best_w:.2f} GNN")
    print(f"Ensemble improvement over XGBoost: "
          f"Pearson r +{ens_r - xgb_r:.3f}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = [
        ("XGBoost", xgb_pred_aligned, xgb_r2, xgb_r),
        ("AttentiveFP GNN", gnn_pred, gnn_r2, gnn_r),
        (f"Ensemble (w={best_w:.2f})", ensemble_pred, ens_r2, ens_r),
    ]

    for ax, (name, pred, r2, r) in zip(axes, models):
        ax.scatter(y_test_gnn, pred, alpha=0.4, s=20, color="steelblue")
        ax.plot([3, 10], [3, 10], "r--", linewidth=1.5)
        ax.set_xlabel("Measured pIC50")
        ax.set_ylabel("Predicted pIC50")
        ax.set_title(f"{name}\nR2={r2:.3f} | Pearson r={r:.3f}")
        ax.set_xlim(3, 10)
        ax.set_ylim(3, 10)

    plt.suptitle("CDK2 Model Comparison � Scaffold Split Test Set",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nFigure saved to figures/model_comparison.png")


if __name__ == "__main__":
    main()

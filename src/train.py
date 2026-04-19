"""
train.py - Training loop with early stopping for CDK2 GNN model.
"""

import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            preds.extend(pred.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())

    preds   = np.array(preds)
    targets = np.array(targets)

    r, _ = pearsonr(preds, targets)
    return {
        "r2":      r2_score(targets, preds),
        "rmse":    mean_squared_error(targets, preds) ** 0.5,
        "pearson": r,
    }


def train(model, train_loader, val_loader, config, device):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )
    loss_fn  = torch.nn.MSELoss()
    patience = config["training"]["patience"]

    best_val_r2  = -np.inf
    best_weights = None
    patience_ctr = 0
    history      = []

    epochs = config["training"]["epochs"]
    print(f"Training for up to {epochs} epochs (patience={patience})...")
    print(f"{'Epoch':<8} {'Train Loss':<14} {'Val R2':<10} {'Val r':<10} {'Patience'}")
    print("-" * 55)

    for epoch in range(epochs):
        train_loss  = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step(val_metrics["rmse"])
        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        if val_metrics["r2"] > best_val_r2:
            best_val_r2  = val_metrics["r2"]
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 5 == 0:
            print(f"{epoch:<8} {train_loss:<14.4f} {val_metrics['r2']:<10.3f} "
                  f"{val_metrics['pearson']:<10.3f} {patience_ctr}/{patience}")

        if patience_ctr >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nBest validation R2: {best_val_r2:.3f}")
    model.load_state_dict(best_weights)
    return model, history

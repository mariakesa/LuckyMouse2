import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import wandb
import os

from data.dataset import NeuronVisionDataset
from evaluation.wandb_binomial_logger import log_binomial_diagnostics


def run_training(
    model_class,
    model_config,
    dataset_path_dict,
    wandb_config,
    num_epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    num_folds=10,
    trials_per_stimulus=50,
    save_dir="saved_models",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(save_dir, exist_ok=True)

    wandb.init(
        project=wandb_config["project"],
        name=f"{wandb_config['name']}_crossval",
        group=wandb_config["name"],
        config={
            "model": model_config,
            "dataset": dataset_path_dict,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "num_folds": num_folds,
        }
    )

    full_dataset = NeuronVisionDataset(
        embeddings_path=dataset_path_dict["embeddings"],
        neural_data_path=dataset_path_dict["neural"]
    )

    stimulus_indices = np.arange(full_dataset.num_stimuli)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    all_test_preds, all_test_counts, all_test_neurons = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(stimulus_indices)):
        print(f"\n🔁 Fold {fold_idx + 1}/{num_folds}")

        def stim_to_indices(stim_idxs):
            return [s * full_dataset.num_neurons + n for s in stim_idxs for n in range(full_dataset.num_neurons)]

        train_indices = stim_to_indices(train_idx)
        test_indices = stim_to_indices(test_idx)

        train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(full_dataset, test_indices), batch_size=batch_size)

        model = model_class(**model_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            all_preds, all_counts, all_neurons = [], [], []
            for batch in train_loader:
                optimizer.zero_grad()
                x = batch["image_embedding"].to(device)
                neuron_idx = batch["neuron_idx"].to(device)
                count = batch["response"].to(device).float()
                logits = model(x, neuron_idx).squeeze()
                p_hat = torch.sigmoid(logits)
                loss = torch.nn.functional.binary_cross_entropy(
                    p_hat,
                    count / trials_per_stimulus,
                    weight=torch.full_like(count, fill_value=trials_per_stimulus)
                ) / count.numel()
                
                loss.backward()
                optimizer.step()

                all_preds.append(p_hat.detach().cpu().numpy())
                all_counts.append(count.cpu().numpy())
                all_neurons.append(neuron_idx.cpu().numpy())

            pred_probs = np.concatenate(all_preds)
            true_counts = np.concatenate(all_counts)
            neuron_ids = np.concatenate(all_neurons)

            log_binomial_diagnostics(pred_probs, true_counts, neuron_ids, trials_per_stimulus, phase="train")
            print(f"Fold {fold_idx}, Epoch {epoch}: BCE Loss = {loss.item():.4f}")

        # Evaluation
        model.eval()
        eval_preds, eval_counts, eval_neurons = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["image_embedding"].to(device)
                neuron_idx = batch["neuron_idx"].to(device)
                count = batch["response"].to(device).float()
                logits = model(x, neuron_idx).squeeze()
                p_hat = torch.sigmoid(logits)
                eval_preds.append(p_hat.cpu().numpy())
                eval_counts.append(count.cpu().numpy())
                eval_neurons.append(neuron_idx.cpu().numpy())

        fold_preds = np.concatenate(eval_preds)
        fold_counts = np.concatenate(eval_counts)
        fold_neurons = np.concatenate(eval_neurons)

        all_test_preds.append(fold_preds)
        all_test_counts.append(fold_counts)
        all_test_neurons.append(fold_neurons)

        log_binomial_diagnostics(fold_preds, fold_counts, fold_neurons, trials_per_stimulus, phase="test")

        # Save model
        fold_dir = os.path.join(save_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))

    wandb.finish()

import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def log_binomial_diagnostics(pred_probs, true_counts, neuron_ids, trials_per_stimulus, phase="train"):
    """
    Logs diagnostics for binomial model.

    Args:
        pred_probs (np.ndarray): Predicted probabilities, shape (N,)
        true_counts (np.ndarray): Observed counts, shape (N,) in [0, trials_per_stimulus]
        neuron_ids (np.ndarray): Neuron IDs per prediction, shape (N,)
        trials_per_stimulus (int): Number of trials that contributed to each count
        phase (str): "train" or "test"
    """
    eps = 1e-8
    pred_probs = np.clip(pred_probs, eps, 1 - eps)
    log_likelihood = (
        true_counts * np.log(pred_probs) +
        (trials_per_stimulus - true_counts) * np.log(1 - pred_probs)
    )

    total_log_likelihood = np.sum(log_likelihood)
    avg_data_prob = np.exp(np.mean(log_likelihood / trials_per_stimulus))

    wandb.log({
        f"{phase}/total_log_likelihood": total_log_likelihood,
        f"{phase}/avg_data_probability": avg_data_prob,
    })

    # === Histogram of predicted avg probabilities per neuron ===
    neuron_ids = np.asarray(neuron_ids)
    pred_probs = np.asarray(pred_probs)
    df = {}
    for nid, p in zip(neuron_ids, pred_probs):
        df.setdefault(nid, []).append(p)
    avg_per_neuron = np.array([np.mean(v) for v in df.values()])

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(avg_per_neuron, bins=30, ax=ax)
    ax.set_title(f"{phase.capitalize()} Avg Predicted Probabilities per Neuron")
    ax.set_xlabel("Average Predicted Spike Probability")
    wandb.log({f"{phase}/avg_pred_prob_histogram": wandb.Image(fig)})
    plt.close(fig)

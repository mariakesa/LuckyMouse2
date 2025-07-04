import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

class NeuronVisionDataset(Dataset):
    def __init__(self, embeddings_path, neural_data_path):
        with open(embeddings_path, "rb") as f:
            embeddings_raw = pickle.load(f)

        self.image_embeddings = torch.tensor(
            embeddings_raw['natural_scenes'], dtype=torch.float32
        )  # (118, D)

        # Load new reduced binomial spike count matrix (N_neurons, 118)
        self.neural_counts = np.load(neural_data_path, mmap_mode='r')  # (N_neurons, 118)

        self.num_stimuli = self.image_embeddings.shape[0]
        self.num_neurons = self.neural_counts.shape[0]

        self.total_samples = self.num_stimuli * self.num_neurons

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        stim_idx = idx // self.num_neurons
        neuron_idx = idx % self.num_neurons

        image_embedding = self.image_embeddings[stim_idx]
        spike_count = float(self.neural_counts[neuron_idx, stim_idx])  # Binomial count

        return {
            "image_embedding": image_embedding,
            "neuron_idx": neuron_idx,
            "stimulus_idx": stim_idx,
            "response": spike_count
        }

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = NeuronVisionDataset(
        embeddings_path="pixel_transformer_neuro/data/processed/google_vit-base-patch16-224_embeddings_softmax.pkl",
        neural_data_path="pixel_transformer_neuro/data/processed/hybrid_neural_responses_reduced.npy"
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for batch in loader:
        print(batch["image_embedding"].shape)  # (64, D)
        print(batch["neuron_idx"].shape)       # (64,)
        print(batch["response"].shape)         # (64,)

        print(batch["response"])
        break

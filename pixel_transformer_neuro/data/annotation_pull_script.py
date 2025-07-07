import os
import numpy as np
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
from dotenv import load_dotenv

# ------------------------------
# Setup
# ------------------------------
load_dotenv()
allen_cache_path = Path(os.environ.get('CAIM_ALLEN_CACHE_PATH'))
boc = BrainObservatoryCache(manifest_file=str(allen_cache_path / 'brain_observatory_manifest.json'))

input_dir = "/home/maria/Documents/AllenBrainObservatory/neural_activity_matrices"
output_path = "/home/maria/LuckyMouse2/pixel_transformer_neuro/data/cell_specimen_ids_in_order.npy"

all_cell_ids = []

# ------------------------------
# Main loop over session files
# ------------------------------
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith("_neural_responses.npy"):
        session_id_str = filename.split("_")[0]
        session_id = int(session_id_str)

        try:
            session_data = boc.get_ophys_experiment_data(session_id)
            cell_ids = session_data.get_cell_specimen_ids()
            sorted_ids = sorted(cell_ids)  # sort to preserve deterministic order
            all_cell_ids.extend(sorted_ids)
        except Exception as e:
            print(f"⚠️ Error with session {session_id}: {e}")

# ------------------------------
# Save to .npy file
# ------------------------------
all_cell_ids = np.array(all_cell_ids, dtype=np.int64)
np.save(output_path, all_cell_ids)
print(f"✅ Saved {len(all_cell_ids)} cell_specimen_ids to {output_path}")

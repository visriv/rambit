from pathlib import Path
import pickle

def save_pkl(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def save_npy(arr, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)

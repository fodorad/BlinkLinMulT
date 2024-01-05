import pickle
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


DB_DIR = Path('data/db/zju_v2')
DB_DIR_OUT = Path('data/db_processed/zju_v2')


def convert(paths: list[str], data: np.ndarray, labels: np.ndarray) -> list[dict]:
    samples = []
    for ind in range(len(paths)):
        samples.append({
            'path': paths[ind],  # str
            'label': labels[ind],  # ()
            'eye': {
                'eye_original': data[ind, :, :, :],  # (H, W, C) == (24, 24, 3)
                'eye': cv2.resize(data[ind, :, :, :], (64, 64), interpolation=cv2.INTER_AREA) # (H, W, C) == (64, 64, 3)
            }
        })
    return samples


def save_samples(thr: float = 0.2, seed: int = 42) -> None:
    train_neg_paths = list(map(str, (DB_DIR / 'Dataset_A_Eye_Images' / 'openEyesTraining').glob('*.jpg')))
    train_pos_paths = list(map(str, (DB_DIR / 'Dataset_A_Eye_Images' / 'closedEyesTraining').glob('*.jpg')))
    test_neg_paths = list(map(str, (DB_DIR / 'Dataset_A_Eye_Images' / 'openEyesTest').glob('*.jpg')))
    test_pos_paths = list(map(str, (DB_DIR / 'Dataset_A_Eye_Images' / 'closedEyesTest').glob('*.jpg')))
    train_neg_paths, valid_neg_paths = train_test_split(train_neg_paths, test_size=thr, random_state=seed)
    train_pos_paths, valid_pos_paths = train_test_split(train_pos_paths, test_size=thr, random_state=seed)

    train_neg = np.array([cv2.imread(elem) for elem in train_neg_paths])
    train_pos = np.array([cv2.imread(elem) for elem in train_pos_paths])
    valid_neg = np.array([cv2.imread(elem) for elem in valid_neg_paths])
    valid_pos = np.array([cv2.imread(elem) for elem in valid_pos_paths])
    test_neg = np.array([cv2.imread(elem) for elem in test_neg_paths])
    test_pos = np.array([cv2.imread(elem) for elem in test_pos_paths])

    train_paths = train_neg_paths + train_pos_paths
    train_samples = np.concatenate([train_neg, train_pos], axis=0)
    train_labels = np.concatenate([
        np.zeros(shape=(train_neg.shape[0],)),
        np.ones(shape=(train_pos.shape[0],))
    ])
    valid_paths = valid_neg_paths + valid_pos_paths
    valid_samples = np.concatenate([valid_neg, valid_pos], axis=0)
    valid_labels = np.concatenate([
        np.zeros(shape=(valid_neg.shape[0],)),
        np.ones(shape=(valid_pos.shape[0],))
    ])
    test_paths = test_neg_paths + test_pos_paths
    test_samples = np.concatenate([test_neg, test_pos], axis=0)
    test_labels = np.concatenate([
        np.zeros(shape=(test_neg.shape[0],)),
        np.ones(shape=(test_pos.shape[0],))
    ])

    with open(DB_DIR_OUT / '0_data.pkl', 'wb') as f:
        pickle.dump(convert(train_paths, train_samples, train_labels), f)

    with open(DB_DIR_OUT / '1_data.pkl', 'wb') as f:
        pickle.dump(convert(valid_paths, valid_samples, valid_labels), f)

    with open(DB_DIR_OUT / '2_data.pkl', 'wb') as f:
        pickle.dump(convert(test_paths, test_samples, test_labels), f)


if __name__ == '__main__':
    save_samples()
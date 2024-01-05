import pickle
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from exordium.video.iris import IrisWrapper
from exordium.video.tddfa_v2 import TDDFA_V2
from exordium.utils.decorator import timer
from blinklinmult import PathType


DB_DIR = Path('data/db/CEW')
DB_DIR_OUT = Path('data/db_processed/cew')


@timer
def save_eye_crops(output_path: PathType, bb_size: int = 40) -> None:
    # read closed eye filenames and eye coordinates
    with open(DB_DIR / 'dataset_B_FacialImages' / 'EyeCoordinatesInfo_ClosedFace.txt', 'r') as f:
        lines = f.readlines()
        lines = [line.replace('\n', '').split(' ') for line in lines]
        closed_file_coords = {
            elem[0]: (1, np.array(elem[1:]).astype(int))
            for elem in lines
        }

    # read open eye filenames and eye coordinates
    with open(DB_DIR / 'dataset_B_FacialImages' / 'EyeCoordinatesInfo_OpenFace.txt', 'r') as f:
        lines = f.readlines()
        lines = [line.replace('\n', '').split(' ') for line in lines]
        open_file_coords = {
            elem[0]: (0, np.array(elem[1:]).astype(int))
            for elem in lines
        }

    file_coords = open_file_coords | closed_file_coords

    face_model = TDDFA_V2()
    eye_model = IrisWrapper()

    closed_face_dir = DB_DIR / 'dataset_B_FacialImages' / 'ClosedFace'
    open_face_dir = DB_DIR / 'dataset_B_FacialImages' / 'OpenFace'
    eye_dir = DB_DIR_OUT / 'eyes'
    eye_dir.mkdir(parents=True, exist_ok=True)
    (eye_dir / 'left').mkdir(parents=True, exist_ok=True)
    (eye_dir / 'right').mkdir(parents=True, exist_ok=True)

    samples = []
    for id, (name, (label, coords)) in enumerate(tqdm(file_coords.items(), total=len(file_coords), desc='[CEW] eye crops')):
        face_dir = closed_face_dir if label else open_face_dir
        face_path = str(face_dir / name)
        face = cv2.imread(face_path)

        sample = {'id': id, 'name': name, 'path': face_path, 'label': label, 'headpose': face_model(face)['headpose']}
        left_eye_path = str(eye_dir / 'left' / f'{id:06d}.png')
        right_eye_path = str(eye_dir / 'right' / f'{id:06d}.png')

        # eye crop rgb
        y_min = max(coords[1] - bb_size // 2, 0)
        y_max = min(100, coords[1] + bb_size // 2)
        x_min = max(coords[0] - bb_size // 2, 0)
        x_max = min(100, coords[0] + bb_size // 2)
        left_eye = face[y_min:y_max, x_min:x_max, :]
        y_min = max(coords[3] - bb_size // 2, 0)
        y_max = min(100, coords[3] + bb_size // 2)
        x_min = max(coords[2] - bb_size // 2, 0)
        x_max = min(100, coords[2] + bb_size // 2)
        right_eye = face[y_min:y_max, x_min:x_max, :]
        cv2.imwrite(left_eye_path, left_eye)
        cv2.imwrite(right_eye_path, right_eye)

        left_eye_features = eye_model.eye_to_features(left_eye_path)
        right_eye_features = eye_model.eye_to_features(right_eye_path)
        sample |= {'left_eye': left_eye_features, 'right_eye': right_eye_features}
        samples.append(sample)

    with open(str(output_path), 'wb') as f:
        pickle.dump(samples, f)

    print(f'[CEW] feature extraction is done: {str(output_path)}')


if __name__ == '__main__':
    save_eye_crops(output_path=DB_DIR_OUT / '0_data.pkl')
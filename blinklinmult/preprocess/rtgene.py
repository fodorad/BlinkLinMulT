import os
import pickle
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
from exordium.video.iris import IrisWrapper
from exordium.utils.decorator import timer
from exordium.video.tddfa_v2 import TDDFA_V2, FaceLandmarks


DB_DIR = Path('data/db/rt-gene')
DB_DIR_OUT = Path('data/db_processed/rt-gene')
(DB_DIR_OUT / 'cache').mkdir(parents=True, exist_ok=True)
IDS = list(range(17))
SUBJECTS: pd.DataFrame = pd.read_csv(f'{str(DB_DIR)}/rt_bene_subjects.csv',
                                     names=['row_ind', 'csv_name', 'left_eye_dir', 'right_eye_dir', 'subset_name', 'fold_id']).set_index('row_ind')


def save_face_crops():
    db_stat = {0.0: 0, 0.5: 0, 1.0: 0}

    for i, subject_row in SUBJECTS.iterrows():
        id = int(subject_row["csv_name"][1:4])
        (DB_DIR_OUT / 'faces' / str(id)).mkdir(parents=True, exist_ok=True)
        csv_content = pd.read_csv(f"{str(DB_DIR)}/{subject_row['csv_name']}", names=['filename', 'label'])

        faces = []
        labels = []
        face_paths = []
        frame_ids = []
        for j, content_row in tqdm(csv_content.iterrows(), total=len(csv_content)):
            face_path = str(DB_DIR / f's{id:03d}_noglasses/natural/face' / content_row['filename'].replace('left', 'face'))
            if not Path(face_path).exists(): continue # skip sample if there is no corresponding frame
            face_paths.append(face_path)
            face = cv2.imread(face_path)
            faces.append(face)
            frame_id = int(Path(face_path).stem[5:11])
            frame_ids.append(frame_id)
            cv2.imwrite(str(DB_DIR_OUT / 'faces' / str(id) / f'{frame_id:06d}.png'), face)
            labels.append(content_row['label'])
            db_stat[content_row['label']] += 1

        faces = np.stack(faces)
        labels = np.array(labels)

        with open(DB_DIR_OUT / 'cache' / f'{id}_faces.pkl', 'wb') as f:
            pickle.dump({'face_paths': face_paths, 'frame_ids': frame_ids, 'faces': faces, 'labels': labels}, f)

        print(f'id {id} done')

    print(db_stat)


def save_eye_crops():
    face_model = TDDFA_V2()
    for i, subject_row in SUBJECTS.iterrows():
        id = int(subject_row["csv_name"][1:4])
        face_dir = DB_DIR_OUT / 'faces' / str(id)
        face_paths = [str(face_dir / elem) for elem in sorted(os.listdir(face_dir))]

        eye_dir = DB_DIR_OUT / 'eyes' / str(id)
        eye_dir.mkdir(parents=True, exist_ok=True)
        (eye_dir / 'left').mkdir(parents=True, exist_ok=True)
        (eye_dir / 'right').mkdir(parents=True, exist_ok=True)

        for face_path in tqdm(face_paths, desc=f'Extract eye patches from id {id}'):
            landmarks = face_model(face_path)['landmarks']
            left_eye_xyxy = np.array(list(landmarks[FaceLandmarks.LEFT_EYE_LEFT,:]) + \
                                     list(landmarks[FaceLandmarks.LEFT_EYE_RIGHT,:])).reshape(4,)
            right_eye_xyxy = np.array(list(landmarks[FaceLandmarks.RIGHT_EYE_LEFT,:]) + \
                                      list(landmarks[FaceLandmarks.RIGHT_EYE_RIGHT,:])).reshape(4,)
            out = face_model.face_to_xyxy_eyes_crop(face_path, left_eye_xyxy, right_eye_xyxy)
            cv2.imwrite(str(eye_dir / 'left' / f'{Path(face_path).stem}.png'), out['left_eye'])
            cv2.imwrite(str(eye_dir / 'right' / f'{Path(face_path).stem}.png'), out['right_eye'])

        print(f'id {id} done')


@timer
def extract_features(data: dict,
                     face_dir: str | Path,
                     left_eye_dir: str | Path,
                     right_eye_dir: str | Path,
                     output_path: str | Path):

    face_paths = [str(Path(face_dir) / elem) for elem in sorted(os.listdir(face_dir))]
    left_eye_paths = [str(Path(left_eye_dir) / elem) for elem in sorted(os.listdir(left_eye_dir))]
    right_eye_paths = [str(Path(right_eye_dir) / elem) for elem in sorted(os.listdir(right_eye_dir))]

    face_model = TDDFA_V2()
    eye_model = IrisWrapper()

    headposes = []
    for face_path in tqdm(face_paths, desc='Extract headpose feature'):
        headposes.append({'id': int(Path(face_path).stem), 'headpose': face_model(face_path)['headpose']})

    eyes = []
    for left_eye_path, right_eye_path in tqdm(zip(left_eye_paths, right_eye_paths), total=len(left_eye_paths), desc='Extract eye features'):
        eyes.append({'id': int(Path(left_eye_path).stem),
                     'left_eye': eye_model.eye_to_features(left_eye_path),
                     'right_eye': eye_model.eye_to_features(right_eye_path)})

    ids = sorted([elem['id'] for elem in headposes])
    features = []
    for id in tqdm(ids, total=len(ids), desc='Merge headpose and eye features, then save to pickle'):
        label = data['labels'][data['frame_ids'].index(id)]
        headpose = next((elem for elem in headposes if elem['id'] == id))
        eye = next((elem for elem in eyes if elem['id'] == id))
        features.append({'label': label} | headpose | eye)

    with open(output_path, 'wb') as f:
        pickle.dump(features, f)

    print(f'rt-gene feature extraction is done: {output_path}')


if __name__ == '__main__':
    save_face_crops()
    save_eye_crops()

    print(SUBJECTS)
    get_ind = lambda fold_ind: SUBJECTS[SUBJECTS['fold_id']==fold_ind].index.to_numpy()
    test_ids = list(get_ind(3))
    training_ids = sorted(list(np.concatenate([get_ind(i) for i in range(3)])))
    print(training_ids, test_ids)

    for id in IDS:
        print(f'Started id {id}')

        with open(DB_DIR_OUT / 'cache' / f'{id}_faces.pkl', 'rb') as f:
            data = pickle.load(f)

        extract_features(data=data,
                         face_dir=DB_DIR_OUT / 'faces' / str(id),
                         left_eye_dir=DB_DIR_OUT / 'eyes'/ str(id) / 'left',
                         right_eye_dir=DB_DIR_OUT / 'eyes'/ str(id) / 'right',
                         output_path=DB_DIR_OUT / f'{id}_data.pkl')
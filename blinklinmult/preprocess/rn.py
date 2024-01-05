import os
import pickle
from pathlib import Path
from tqdm import tqdm
from exordium.video.iris import IrisWrapper
from exordium.utils.decorator import timer
from exordium.video.tddfa_v2 import TDDFA_V2
from blinklinmult.preprocess.reader import Tag


DB_DIR_RN15_test = Path('data/db/RN/test/rn15')
DB_DIR_OUT_RN15_test = Path('data/db_processed/rn15_test')
DB_DIR_RN15_train = Path('data/db/RN/train/rn15')
DB_DIR_OUT_RN15_train = Path('data/db_processed/rn15_train')
DB_DIR_RN15_val = Path('data/db/RN/val/rn15')
DB_DIR_OUT_RN15_val = Path('data/db_processed/rn15_val')

DB_DIR_RN30_test = Path('data/db/RN/test/rn30')
DB_DIR_OUT_RN30_test = Path('data/db_processed/rn30_test')
DB_DIR_RN30_train = Path('data/db/RN/train/rn30')
DB_DIR_OUT_RN30_train = Path('data/db_processed/rn30_train')
DB_DIR_RN30_val = Path('data/db/RN/val/rn30')
DB_DIR_OUT_RN30_val = Path('data/db_processed/rn30_val')


def save_samples(db_dir, db_dir_out, fps):
    tag_paths = list(Path(db_dir).glob('*/*.tag'))
    samples = []

    for tag_path in tqdm(tag_paths, desc='Save mp4'):
        id = tag_path.parent.name
        frames_dir = db_dir / id / 'frames'
        output_dir = db_dir_out / 'visualize'
        tag = Tag(tag_path=tag_path, frames_dir=frames_dir)
        samples += tag.generate_positive_samples(output_dir=output_dir, fps=fps)

    return samples


def save_face_crops(db_dir, db_dir_out):
    tag_paths = list(Path(db_dir).glob('*/*.tag'))
    for tag_path in tqdm(tag_paths, desc='Save faces'):
        id = tag_path.parent.name
        faces_dir = db_dir_out / 'faces' / id
        tag = Tag(tag_path=tag_path)
        tag.save_annotated_face_crops(faces_dir)


def save_eye_crops(db_dir, db_dir_out):
    tag_paths = list(Path(db_dir).glob('*/*.tag'))
    for tag_path in tqdm(tag_paths, desc='Save eyes'):
        id = tag_path.parent.name
        eyes_dir = db_dir_out / 'eyes' / id
        tag = Tag(tag_path=tag_path)
        tag.save_annotated_eye_crops(eyes_dir)


@timer
def extract_features(tag_path: str | Path,
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
        headposes.append({'id': int(Path(face_path).stem), 'headpose': face_model.inference(face_path)['headpose']})

    eyes = []
    for left_eye_path, right_eye_path in tqdm(zip(left_eye_paths, right_eye_paths), total=len(left_eye_paths), desc='Extract eye features'):
        eyes.append({'id': int(Path(left_eye_path).stem),
                     'left_eye': eye_model.eye_to_features(left_eye_path),
                     'right_eye': eye_model.eye_to_features(right_eye_path)})

    tag = Tag(tag_path=tag_path)
    ids = sorted([elem['id'] for elem in headposes])

    features = []
    for id in tqdm(ids, total=len(ids), desc='Merge headpose and eye features, then save to pickle'):
        label = tag.blink_label(id)
        headpose = next((elem for elem in headposes if elem['id'] == id))
        eye = next((elem for elem in eyes if elem['id'] == id))
        features.append({'label': label} | headpose | eye)

    with open(output_path, 'wb') as f:
        pickle.dump(features, f)

    print(f'TalkingFace feature extraction is done: {output_path}')


if __name__ == '__main__':

    save_samples(DB_DIR_RN15_test, DB_DIR_OUT_RN15_test, 15)
    save_samples(DB_DIR_RN15_train, DB_DIR_OUT_RN15_train, 15)
    save_samples(DB_DIR_RN15_val, DB_DIR_OUT_RN15_val, 15)
    save_samples(DB_DIR_RN30_test, DB_DIR_OUT_RN30_test, 30)
    save_samples(DB_DIR_RN30_train, DB_DIR_OUT_RN30_train, 30)
    save_samples(DB_DIR_RN30_val, DB_DIR_OUT_RN30_val, 30)
    save_face_crops(DB_DIR_RN15_train, DB_DIR_OUT_RN15_train)
    save_face_crops(DB_DIR_RN15_val, DB_DIR_OUT_RN15_val)
    save_face_crops(DB_DIR_RN15_test, DB_DIR_OUT_RN15_test)
    save_face_crops(DB_DIR_RN30_train, DB_DIR_OUT_RN30_train)
    save_face_crops(DB_DIR_RN30_val, DB_DIR_OUT_RN30_val)
    save_face_crops(DB_DIR_RN30_test, DB_DIR_OUT_RN30_test)
    save_eye_crops(DB_DIR_RN15_train, DB_DIR_OUT_RN15_train)
    save_eye_crops(DB_DIR_RN15_val, DB_DIR_OUT_RN15_val)
    save_eye_crops(DB_DIR_RN15_test, DB_DIR_OUT_RN15_test)
    save_eye_crops(DB_DIR_RN30_train, DB_DIR_OUT_RN30_train)
    save_eye_crops(DB_DIR_RN30_val, DB_DIR_OUT_RN30_val)
    save_eye_crops(DB_DIR_RN30_test, DB_DIR_OUT_RN30_test)

    for fps in [15, 30]:
        for subset in ['train', 'val', 'test']:

            ids = os.listdir(f'data/db/RN/{subset}/rn{fps}')
            item = '18trainvalLeftRight_MergedDoubleBlinks_eval.txt'
            if item in ids:
                ids.remove('18trainvalLeftRight_MergedDoubleBlinks_eval.txt') # only exception in test/rn30

            tag_paths = list(Path(f'data/db/RN/{subset}/rn{fps}').glob('*/*.tag'))

            for id in ids:
                print(f'Started rn{fps}_{subset}: {id}')
                tag_path = next((elem for elem in tag_paths if elem.parent.name == id))
                db_dir_out = Path(f'data/db_processed/rn{fps}_{subset}')
                extract_features(tag_path=tag_path,
                                 face_dir=db_dir_out / 'faces' / id,
                                 left_eye_dir=db_dir_out / 'eyes'/ id / 'left',
                                 right_eye_dir=db_dir_out / 'eyes'/ id / 'right',
                                 output_path=db_dir_out / f'{id}_data.pkl')
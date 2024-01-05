import os
import pickle
from pathlib import Path
from tqdm import tqdm
from exordium.video.tddfa_v2 import TDDFA_V2
from exordium.video.iris import IrisWrapper
from blinklinmult.preprocess.reader import Tag


DB_DIR = Path('data/db_processed/talkingface')
DB_DIR_OUT = Path('data/db_processed/talkingface/visualize')


def get_original_samples(**kwargs):
    tag = Tag(tag_path=DB_DIR / 'talking.tag', frames_dir=DB_DIR / 'frames')
    positive_samples = tag.generate_positive_samples(output_dir=DB_DIR_OUT / 'positive', win_size=18, fps=30)
    negative_samples = tag.generate_negative_samples(num_samples=len(positive_samples), output_dir=DB_DIR_OUT / 'negative', win_size=18, fps=30)
    return positive_samples + negative_samples


def save_face_crops():
    tag = Tag(tag_path=DB_DIR / 'talking.tag', frames_dir=DB_DIR / 'frames')
    tag.save_annotated_face_crops('data/db_processed/talkingface/faces')


def save_eye_crops():
    tag = Tag(tag_path=DB_DIR / 'talking.tag', frames_dir=DB_DIR / 'frames')
    tag.save_annotated_eye_crops('data/db_processed/talkingface/eyes')


def extract_features(face_dir: str | Path = 'data/db_processed/talkingface/faces',
                     left_eye_dir: str | Path = 'data/db_processed/talkingface/eyes/left',
                     right_eye_dir: str | Path = 'data/db_processed/talkingface/eyes/right',
                     output_path: str | Path = 'data/db_processed/talkingface/0_data.pkl'):

    face_paths = [str(Path(face_dir) / elem) for elem in sorted(os.listdir(face_dir))]
    left_eye_paths = [str(Path(left_eye_dir) / elem) for elem in sorted(os.listdir(left_eye_dir))]
    right_eye_paths = [str(Path(right_eye_dir) / elem) for elem in sorted(os.listdir(right_eye_dir))]

    face_model = TDDFA_V2()
    iris_model = IrisWrapper()

    headposes = []
    for face_path in tqdm(face_paths, desc='Extract headpose feature'):
        headposes.append({'id': int(Path(face_path).stem), 'headpose': face_model.inference(face_path)['headpose']})

    eyes = []
    for left_eye_path, right_eye_path in tqdm(zip(left_eye_paths, right_eye_paths), total=len(left_eye_paths), desc='Extract eye features'):
        eyes.append({'id': int(Path(left_eye_path).stem),
                     'left_eye': iris_model.eye_to_features(left_eye_path),
                     'right_eye': iris_model.eye_to_features(right_eye_path)})

    tag = Tag(tag_path=DB_DIR / 'talking.tag', frames_dir=DB_DIR / 'frames')
    ids = [elem['id'] for elem in headposes]

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
    samples = get_original_samples()
    save_face_crops()
    save_eye_crops()
    extract_features()
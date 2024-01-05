import os
import pickle
from pathlib import Path
from tqdm import tqdm
from exordium.video.io import video2frames
from exordium.utils.decorator import timer
from exordium.video.tddfa_v2 import TDDFA_V2
from exordium.video.iris import IrisWrapper
from blinklinmult.preprocess.reader import Tag


DB_DIR = Path('data/db/eyeblink8')
DB_DIR_OUT = Path('data/db_processed/eyeblink8')
IDS = os.listdir(DB_DIR)


def extract_frames():
    videos = sorted(list(DB_DIR.rglob('*/*.avi')))

    for video in videos:
        id = video.parent.name
        output_dir = DB_DIR_OUT / 'frames' / id
        video2frames(video, output_dir, fps=30)


def save_samples():
    tag_paths = list(Path(DB_DIR).glob('*/*.tag'))
    samples = []

    for tag_path in tqdm(tag_paths, desc='Save mp4'):
        id = tag_path.parent.name
        frames_dir = DB_DIR_OUT / 'frames' / id
        output_dir = DB_DIR_OUT / 'visualize'
        tag = Tag(tag_path=tag_path, frames_dir=frames_dir)
        samples += tag.generate_positive_samples(output_dir=output_dir, fps=30)

    return samples


def save_face_crops():
    tag_paths = list(Path(DB_DIR).glob('*/*.tag'))
    for tag_path in tqdm(tag_paths, desc='Save faces'):
        id = tag_path.parent.name
        frames_dir = DB_DIR_OUT / 'frames' / id
        faces_dir = DB_DIR_OUT / 'faces' / id
        tag = Tag(tag_path=tag_path, frames_dir=frames_dir)
        tag.save_annotated_face_crops(faces_dir)


def save_eye_crops():
    tag_paths = list(Path(DB_DIR).glob('*/*.tag'))
    for tag_path in tqdm(tag_paths, desc='Save faces'):
        id = tag_path.parent.name
        frames_dir = DB_DIR_OUT / 'frames' / id
        eyes_dir = DB_DIR_OUT / 'eyes' / id
        tag = Tag(tag_path=tag_path, frames_dir=frames_dir)
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

    tag_path = Path(tag_path)
    tag = Tag(tag_path=tag_path, frames_dir=DB_DIR_OUT / 'frames' / tag_path.parent.name)
    ids = sorted([elem['id'] for elem in headposes])

    features = []
    for id in tqdm(ids, total=len(ids), desc='Merge headpose and eye features, then save to pickle'):
        label = tag.blink_label(id)
        headpose = next((elem for elem in headposes if elem['id'] == id))
        eye = next((elem for elem in eyes if elem['id'] == id))
        features.append({'label': label} | headpose | eye)

    with open(output_path, 'wb') as f:
        pickle.dump(features, f)

    print(f'Feature extraction is done: {output_path}')


if __name__ == '__main__':
    extract_frames()
    save_samples()
    save_face_crops()
    save_eye_crops()

    tag_paths = list(Path(DB_DIR).glob('*/*.tag'))

    for id in IDS:
        print(f'Started id {id}')
        tag_path = next((elem for elem in tag_paths if elem.parent.name == id))
        extract_features(tag_path=tag_path,
                         face_dir=DB_DIR_OUT / 'faces' / id,
                         left_eye_dir=DB_DIR_OUT / 'eyes'/ id / 'left',
                         right_eye_dir=DB_DIR_OUT / 'eyes'/ id / 'right',
                         output_path=DB_DIR_OUT / f'{id}_data.pkl')
        print('EyeBlink8 is done.')
import os
from tqdm import tqdm
from pathlib import Path
import pickle
from exordium.video.iris import IrisWrapper
from exordium.utils.decorator import timer


DB_DIR = Path('data/db/mrl_eye')
DB_DIR_OUT = Path('data/db_processed/mrl')
IDS = [elem for elem in os.listdir('data/db/mrl_eye/mrlEyes_2018_01')
       if len(elem) == 5 and elem[0] == 's']


@timer
def save_features(input_path: str | Path, output_path: str | Path):
    paths = [Path(input_path) / elem for elem in os.listdir(str(input_path))]
    eye_model = IrisWrapper()

    samples = []
    for path in tqdm(paths, total=len(paths), desc=f'[MRL] {Path(input_path).name} features'):

        data = Path(path).stem.split('_')
        participant_id = int(data[0][1:])  # participant id
        sample = {
            'participant_id': participant_id,
            'id': int(data[1]),  # image number
            'name': Path(path).stem,  # image name
            'gender': int(data[2]),  # 0=male, 1=female
            'glasses': int(data[3]),  # 0=no, 1=yes
            'label': int(not bool(int(data[4]))),  # switch original blink label from 0=close, 1=open to 1=close, 0=open
            'reflection': int(data[5]),  # 0=none, 1=low, 2=high
            'lightning': int(data[6]),  # 0=bad, 1=good
            'sensor': int(data[7]),  # 1=RealSense SR300 640x480, 2=IDS Imaging, 1280x1024, 3=Aptina Imagin 752x480
        }

        eye_features = eye_model.eye_to_features(path)
        sample |= eye_features
        samples.append(sample)

    output_path = str(output_path)

    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)

    print(f'[MRL] feature extraction is done: {output_path}')


if __name__ == '__main__':

    for id in IDS:
        save_features(input_path=DB_DIR / 'mrlEyes_2018_01' / id,
                      output_path=DB_DIR_OUT / f'{int(id[1:])}_data.pkl')
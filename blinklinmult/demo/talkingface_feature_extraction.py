import pickle
from pathlib import Path
import cv2
from tqdm import tqdm
from exordium.video.io import video2frames
from exordium.video.tddfa_v2 import TDDFA_V2
from exordium.video.iris import IrisWrapper
from exordium.video.facedetector import RetinaFaceDetector, RetinaFaceLandmarks
from exordium.video.bb import crop_mid
from blinklinmult.preprocess.reader import Tag


# dataset: https://www.blinkingmatters.com/files/upload/research/talkingFace.zip
# annotation: data/db/talkingface/talking.tag
# video: data/db/talkingface/talking.avi
DB_DIR = Path('data/db/talkingface')
DB_PROCESSED_DIR = Path('data/db_processed/talkingface')


def get_original_samples(**kwargs):
    tag = Tag(tag_path=DB_DIR / 'talking.tag', frames_dir=DB_PROCESSED_DIR / 'frames')
    positive_samples = tag.generate_positive_samples(output_dir=DB_PROCESSED_DIR / 'positive', win_size=15, fps=30)
    negative_samples = tag.generate_negative_samples(num_samples=len(positive_samples), output_dir=DB_PROCESSED_DIR / 'negative', win_size=15, fps=30)
    return positive_samples + negative_samples


def save_face_crops():
    output_dir = DB_PROCESSED_DIR / 'faces'

    if output_dir.exists() and len(list(output_dir.glob('*.png'))) == 5000:
        return

    tag = Tag(tag_path=DB_DIR / 'talking.tag', frames_dir=DB_PROCESSED_DIR / 'frames')
    tag.save_annotated_face_crops(output_dir)


def save_eye_crops():
    output_dir = DB_PROCESSED_DIR / 'eyes'

    if output_dir.exists() and len(list((output_dir / 'left').glob('*.png'))) == 5000 and len(list((output_dir / 'right').glob('*.png'))) == 5000:
        return

    tag = Tag(tag_path=DB_DIR / 'talking.tag', frames_dir=DB_PROCESSED_DIR / 'frames')
    tag.save_annotated_eye_crops(output_dir)


def extract_features(frame_dir: str | Path = 'data/db_processed/talkingface/frames',
                     face_dir: str | Path = 'data/db_processed/talkingface/faces',
                     left_eye_dir: str | Path = 'data/db_processed/talkingface/eyes/left',
                     right_eye_dir: str | Path = 'data/db_processed/talkingface/eyes/right',
                     output_path: str | Path = 'data/db_processed/talkingface/talkingface.pkl'):

    frame_paths = sorted(list(Path(frame_dir).glob('*.png')))
    face_paths = sorted(list(Path(face_dir).glob('*.png')))
    left_eye_paths = sorted(list(Path(left_eye_dir).glob('*.png')))
    right_eye_paths = sorted(list(Path(right_eye_dir).glob('*.png')))

    tag = Tag(tag_path=DB_DIR / 'talking.tag', frames_dir=DB_PROCESSED_DIR / 'frames')
    tddfa_v2 = TDDFA_V2()
    retinaface = RetinaFaceDetector()
    iris_model = IrisWrapper()

    samples = {}
    for frame_path, annotation_face_path, annotation_left_eye_path, annotation_right_eye_path in tqdm(zip(frame_paths, face_paths, left_eye_paths, right_eye_paths), total=len(frame_paths), desc='Extract features'):
        assert int(frame_path.stem) == int(annotation_face_path.stem) == int(annotation_left_eye_path.stem) == int(annotation_right_eye_path.stem)

        sample_id = int(frame_path.stem)
        frame = cv2.imread(frame_path)

        # crop eyes using the annotation
        annotation_face = cv2.imread(annotation_face_path)
        annotation_left_eye = cv2.cvtColor(cv2.imread(annotation_left_eye_path), cv2.COLOR_BGR2RGB)
        annotation_right_eye = cv2.cvtColor(cv2.imread(annotation_right_eye_path), cv2.COLOR_BGR2RGB)
        annotation_features_left_eye = iris_model.eye_to_features(annotation_left_eye)
        annotation_features_right_eye = iris_model.eye_to_features(annotation_right_eye)

        # crop eyes using retinaface coarse landmarks
        retinaface_detection = retinaface.detect_image_path(frame_path).get_detection_with_biggest_bb()
        retinaface_landmarks = retinaface_detection.landmarks # (5,2)
        retinaface_face = retinaface_detection.bb_crop()
        retinaface_distance = abs((retinaface_landmarks[RetinaFaceLandmarks.LEFT_EYE.value,:] - retinaface_landmarks[RetinaFaceLandmarks.RIGHT_EYE.value,:])[0])
        retinaface_left_eye = crop_mid(image=frame, mid=retinaface_landmarks[RetinaFaceLandmarks.RIGHT_EYE.value,:], bb_size=retinaface_distance)
        retinaface_right_eye = crop_mid(image=frame, mid=retinaface_landmarks[RetinaFaceLandmarks.LEFT_EYE.value,:], bb_size=retinaface_distance)
        retinaface_features_left_eye = iris_model.eye_to_features(retinaface_left_eye)
        retinaface_features_right_eye = iris_model.eye_to_features(retinaface_right_eye)

        # crop eyes using tddfa_v2 fine landmarks
        tddfa_v2_features = tddfa_v2.face_to_eyes_crop(annotation_face)
        tddfa_v2_left_eye = tddfa_v2_features['left_eye']
        tddfa_v2_right_eye = tddfa_v2_features['right_eye']
        tddfa_v2_features_left_eye = iris_model.eye_to_features(tddfa_v2_left_eye) if tddfa_v2_left_eye is not None else None
        tddfa_v2_features_right_eye = iris_model.eye_to_features(tddfa_v2_right_eye) if tddfa_v2_right_eye is not None else None

        tddfa_v2_retinaface_features = tddfa_v2.face_to_eyes_crop(retinaface_face)
        tddfa_v2_retinaface_left_eye = tddfa_v2_retinaface_features['left_eye']
        tddfa_v2_retinaface_right_eye = tddfa_v2_retinaface_features['right_eye']
        tddfa_v2_retinaface_features_left_eye = iris_model.eye_to_features(tddfa_v2_retinaface_left_eye) if tddfa_v2_retinaface_left_eye is not None else None
        tddfa_v2_retinaface_features_right_eye = iris_model.eye_to_features(tddfa_v2_retinaface_right_eye) if tddfa_v2_retinaface_right_eye is not None else None

        samples[sample_id] = {
            'id': sample_id,
            'label': tag.blink_label(sample_id),
            'frame': frame,
            'frame_path': frame_path,
            'annotation_face': annotation_face,
            'annotation_face_path': annotation_face_path,
            'annotation_left_eye': annotation_left_eye,
            'annotation_left_eye_path': annotation_left_eye_path,
            'annotation_right_eye': annotation_right_eye,
            'annotation_right_eye_path': annotation_right_eye_path,
            'annotation_left_eye_features': annotation_features_left_eye,
            'annotation_right_eye_features': annotation_features_right_eye,
            'tddfa-annotation_landmarks': tddfa_v2_features['landmarks'],
            'tddfa-annotation_headpose': tddfa_v2_features['headpose'],
            'tddfa-annotation_left_eye': tddfa_v2_left_eye,
            'tddfa-annotation_right_eye': tddfa_v2_right_eye,
            'tddfa-annotation_left_eye_features': tddfa_v2_features_left_eye,
            'tddfa-annotation_right_eye_features': tddfa_v2_features_right_eye,
            'retinaface_landmarks': retinaface_landmarks,
            'retinaface_face': retinaface_face,
            'retinaface_left_eye': retinaface_left_eye,
            'retinaface_right_eye': retinaface_right_eye,
            'retinaface_left_eye_features': retinaface_features_left_eye,
            'retinaface_right_eye_features': retinaface_features_right_eye,
            'tddfa-retinaface_landmarks': tddfa_v2_retinaface_features['landmarks'],
            'tddfa-retinaface_headpose': tddfa_v2_retinaface_features['headpose'],
            'tddfa-retinaface_left_eye': tddfa_v2_retinaface_left_eye,
            'tddfa-retinaface_right_eye': tddfa_v2_retinaface_right_eye,
            'tddfa-retinaface_left_eye_features': tddfa_v2_retinaface_features_left_eye,
            'tddfa-retinaface_right_eye_features': tddfa_v2_retinaface_features_right_eye,
        }

    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)

    print(f'TalkingFace feature extraction is done: {output_path}')


def visualize(index: int = 0):
    from exordium.visualization.headpose import draw_headpose_axis
    from exordium.visualization.landmarks import visualize_landmarks, visualize_iris
    
    PICKLE_PATH = DB_PROCESSED_DIR / 'talkingface.pkl'
    OUTPUT_DIR = DB_PROCESSED_DIR / 'visualization'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load pickle file and get first sample
    with open(PICKLE_PATH, 'rb') as f:
        data = pickle.load(f)

    sample = data[index]
    print(f'Sample {index} loaded.')

    # print frame and face
    cv2.imwrite(OUTPUT_DIR / 'frame.png', sample['frame'])
    cv2.imwrite(OUTPUT_DIR / 'annotation_face.png', sample['annotation_face'])
    cv2.imwrite(OUTPUT_DIR / 'retinaface_face.png', cv2.cvtColor(sample['retinaface_face'], cv2.COLOR_RGB2BGR))
    print('Face images saved.')

    # Visualize cropped eyes
    cv2.imwrite(OUTPUT_DIR / 'annotation_left_eye.png', cv2.cvtColor(sample['annotation_left_eye'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(OUTPUT_DIR / 'annotation_right_eye.png', cv2.cvtColor(sample['annotation_right_eye'], cv2.COLOR_RGB2BGR))    

    cv2.imwrite(OUTPUT_DIR / 'tddfa-annotation_left_eye.png', sample['tddfa-annotation_left_eye'])
    cv2.imwrite(OUTPUT_DIR / 'tddfa-annotation_right_eye.png', sample['tddfa-annotation_right_eye'])

    cv2.imwrite(OUTPUT_DIR / 'retinaface_left_eye.png', sample['retinaface_left_eye'])
    cv2.imwrite(OUTPUT_DIR / 'retinaface_right_eye.png', sample['retinaface_right_eye'])    

    cv2.imwrite(OUTPUT_DIR / 'tddfa-retinaface_left_eye.png', cv2.cvtColor(sample['tddfa-retinaface_left_eye'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(OUTPUT_DIR / 'tddfa-retinaface_right_eye.png', cv2.cvtColor(sample['tddfa-retinaface_right_eye'], cv2.COLOR_RGB2BGR))
    print('Eye images saved.')

    # Visualize face landmarks
    img_landmarks = visualize_landmarks(sample['annotation_face'].copy(), sample['tddfa-annotation_landmarks'])
    cv2.imwrite(OUTPUT_DIR / 'annotation_face_tddfa-landmarks.png', img_landmarks)

    img_landmarks = visualize_landmarks(cv2.cvtColor(sample['retinaface_face'].copy(), cv2.COLOR_RGB2BGR), sample['tddfa-retinaface_landmarks'])
    cv2.imwrite(OUTPUT_DIR / 'retinaface_face_tddfa-landmarks.png', img_landmarks)

    img_landmarks = visualize_landmarks(sample['frame'].copy(), sample['retinaface_landmarks'])
    cv2.imwrite(OUTPUT_DIR / 'retinaface_face_retinaface-landmarks.png', img_landmarks)
    print('Face landmarks saved.')

    # Visualize left eye
    img_eye_left = visualize_iris(
        cv2.cvtColor(sample['annotation_left_eye_features']['eye'].copy(), cv2.COLOR_RGB2BGR),
        sample['annotation_left_eye_features']['landmarks'],
        sample['annotation_left_eye_features']['iris_landmarks'])
    cv2.imwrite(OUTPUT_DIR / 'annotation_left_eye_features.png', img_eye_left)

    img_eye_left = visualize_iris(
        sample['tddfa-annotation_left_eye_features']['eye'].copy(),
        sample['tddfa-annotation_left_eye_features']['landmarks'],
        sample['tddfa-annotation_left_eye_features']['iris_landmarks'])
    cv2.imwrite(OUTPUT_DIR / 'tddfa-annotation_left_eye_features.png', img_eye_left)

    img_eye_left = visualize_iris(
        sample['retinaface_left_eye_features']['eye'].copy(),
        sample['retinaface_left_eye_features']['landmarks'],
        sample['retinaface_left_eye_features']['iris_landmarks'])
    cv2.imwrite(OUTPUT_DIR / 'retinaface_left_eye_features.png', img_eye_left)

    img_eye_left = visualize_iris(
        cv2.cvtColor(sample['tddfa-retinaface_left_eye_features']['eye'].copy(), cv2.COLOR_RGB2BGR),
        sample['tddfa-retinaface_left_eye_features']['landmarks'],
        sample['tddfa-retinaface_left_eye_features']['iris_landmarks'])
    cv2.imwrite(OUTPUT_DIR / 'tddfa-retinaface_left_eye_features.png', img_eye_left)
    print('Left eye landmarks saved.')

    # Visualize right eye
    img_eye_right = visualize_iris(
        cv2.cvtColor(sample['annotation_right_eye_features']['eye'].copy(), cv2.COLOR_RGB2BGR),
        sample['annotation_right_eye_features']['landmarks'],
        sample['annotation_right_eye_features']['iris_landmarks'])
    cv2.imwrite(OUTPUT_DIR / 'annotation_right_eye_features.png', img_eye_right)

    img_eye_right = visualize_iris(
        sample['tddfa-annotation_right_eye_features']['eye'].copy(),
        sample['tddfa-annotation_right_eye_features']['landmarks'],
        sample['tddfa-annotation_right_eye_features']['iris_landmarks'])
    cv2.imwrite(OUTPUT_DIR / 'tddfa-annotation_right_eye_features.png', img_eye_right)

    img_eye_right = visualize_iris(
        sample['retinaface_right_eye_features']['eye'].copy(),
        sample['retinaface_right_eye_features']['landmarks'],
        sample['retinaface_right_eye_features']['iris_landmarks'])
    cv2.imwrite(OUTPUT_DIR / 'retinaface_right_eye_features.png', img_eye_right)

    img_eye_right = visualize_iris(
        cv2.cvtColor(sample['tddfa-retinaface_right_eye_features']['eye'].copy(), cv2.COLOR_RGB2BGR),
        sample['tddfa-retinaface_right_eye_features']['landmarks'],
        sample['tddfa-retinaface_right_eye_features']['iris_landmarks'])
    cv2.imwrite(OUTPUT_DIR / 'tddfa-retinaface_right_eye_features.png', img_eye_right)
    print('Right eye landmarks saved.')

    # Visualize headpose
    img_headpose = draw_headpose_axis(sample['annotation_face'].copy(), sample['tddfa-annotation_headpose'])
    cv2.imwrite(OUTPUT_DIR / 'tddfa-annotation_headpose.png', img_headpose)

    img_headpose = draw_headpose_axis(cv2.cvtColor(sample['retinaface_face'].copy(), cv2.COLOR_RGB2BGR), sample['tddfa-retinaface_headpose'])
    cv2.imwrite(OUTPUT_DIR / 'tddfa-retinaface_headpose.png', img_headpose)
    print('Headpose saved.')

    print(f'Visualization is done: {OUTPUT_DIR}')


if __name__ == '__main__':
    video2frames(input_path=DB_DIR / 'talking.avi', output_dir=DB_PROCESSED_DIR / 'frames')
    save_face_crops()
    save_eye_crops()
    extract_features()
    visualize()
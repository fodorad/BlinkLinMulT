"""This script is a quick showcase how to use the models on a single sample of TalkingFace blink sequence.
It guides through the prediction of the first eye blink of TalkingFace, left eye case.
"""
import pickle
import json
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from exordium.utils.normalize import standardization
from blinklinmult.demo.talkingface_feature_extraction import DB_DIR, DB_PROCESSED_DIR
from blinklinmult.preprocess.reader import Tag
from blinklinmult.models import DenseNet121, BlinkLinT, BlinkLinMulT


SAMPLE_IDX = 0

# load annotation file, get a blink sample for testing the models (inference only for now)
tag = Tag(tag_path=DB_DIR / 'talking.tag', frames_dir=DB_PROCESSED_DIR / 'frames')
positive_samples = tag.generate_positive_samples(output_dir=DB_PROCESSED_DIR / 'positive', win_size=15, fps=25)
# negative_samples = tag.generate_negative_samples(num_samples=len(positive_samples), output_dir=DB_PROCESSED_DIR / 'negative', win_size=15, fps=25)

sample_info = positive_samples[SAMPLE_IDX]
ids = np.array([int(Path(elem).stem) for elem in sample_info[0]]) # (15,)
video = np.array([cv2.imread(elem) for elem in sample_info[0]]) # (15, H, W, 3)
gt = sample_info[1] # (15,)

# load raw extracted features created with blinklinmult.preprocess.talkingface
with open(DB_PROCESSED_DIR / 'talkingface.pkl', 'rb') as f:
    data = pickle.load(f)
sample = [data[sample_id] for sample_id in ids]

# prepare low level input feature: RGB left eye patch only, imagenet standardization
test_transform = transforms.Compose([
    transforms.Resize((64, 64), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

low_level_features = torch.stack([
    test_transform(
        Image.fromarray(elem['annotation_left_eye_features']['eye'])
    ) 
    for elem in sample
], dim=0) # (15,64,64,3) -> (15,3,64,64) RGB

# prepare high level input feature: standardized headpose, eye and iris landmark-based features
headpose = np.array([elem['tddfa-retinaface_headpose'] for elem in sample]) # (15,3)
left_landmarks = np.array([elem['annotation_left_eye_features']['landmarks'] for elem in sample]).reshape(15, -1) # (15,71,2) -> (15,142)
left_iris_landmarks = np.array([elem['annotation_left_eye_features']['iris_landmarks'] for elem in sample]).reshape(15, -1) # (15,5,2) -> (15,10)
left_iris_diameters = np.array([elem['annotation_left_eye_features']['iris_diameters'] for elem in sample]) # (15,2)
left_eyelid_pupil_distances = np.array([elem['annotation_left_eye_features']['eyelid_pupil_distances'] for elem in sample]) # (15,2)
left_ear = np.expand_dims(np.array([elem['annotation_left_eye_features']['ear'] for elem in sample]), axis=-1) # (15,) -> (15,1)

# mean and std values are calculated from all left and right eyes' features, dataset-wise standardization
with open(DB_DIR / 'talkingface.json', 'r') as f:
    mean_std_dict = json.load(f)

high_level_features = torch.concat([
    standardization(torch.tensor(headpose),                    torch.tensor(mean_std_dict['headpose']['mean']),               torch.tensor(mean_std_dict['headpose']['std'])),
    standardization(torch.tensor(left_landmarks),              torch.tensor(mean_std_dict['eye_landmarks']['mean']),          torch.tensor(mean_std_dict['eye_landmarks']['std'])),
    standardization(torch.tensor(left_iris_landmarks),         torch.tensor(mean_std_dict['iris_landmarks']['mean']),         torch.tensor(mean_std_dict['iris_landmarks']['std'])),
    standardization(torch.tensor(left_iris_diameters),         torch.tensor(mean_std_dict['iris_diameters']['mean']),         torch.tensor(mean_std_dict['iris_diameters']['std'])),
    standardization(torch.tensor(left_eyelid_pupil_distances), torch.tensor(mean_std_dict['eyelid_pupil_distances']['mean']), torch.tensor(mean_std_dict['eyelid_pupil_distances']['std'])),
    standardization(torch.tensor(left_ear),                    torch.tensor(mean_std_dict['ear']['mean']),                    torch.tensor(mean_std_dict['ear']['std'])),
], dim=-1) # (15,160)

# load models with pretrained weights
densenet121 = DenseNet121(weights='densenet121-union')
densenet121.eval()
blinklint = BlinkLinT(weights='blinklint-union')
blinklint.eval()
blinklinmult = BlinkLinMulT(weights='blinklinmult-union')
blinklinmult.eval()

# add batch dim
input_low = torch.unsqueeze(low_level_features, dim=0).float() # (B, T, C, H, W); (1, 15, 3, 64, 64)
input_high = torch.unsqueeze(high_level_features, dim=0).float() # (B, T, F); (1, 15, 160)


def visualize_blink_probabilities(pred: np.ndarray, gt: np.ndarray, title: str, output_path: Path) -> None:
    """
    Visualizes blink probabilities over 15 frames.

    Args:
        pred (np.ndarray): Predicted probabilities of shape (15,) between 0 and 1.
        gt (np.ndarray): Ground truth of shape (15,), 0 and 1.
        index (int): Sample ID to show in the plot title.
    """
    frames = np.arange(15)
    plt.figure(figsize=(10, 5))
    plt.plot(frames, pred, linestyle='--', linewidth=2, label='Prediction')
    plt.plot(frames, gt, linestyle='-', linewidth=3, marker='o', label='Ground Truth')
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel("Blink probability")
    plt.xlim(0, 14)
    plt.ylim(0, 1)
    plt.legend()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[TalkingFace] {title} saved to {output_path}")


# single sample inference
with torch.no_grad():
    # frame-wise model, eye patches only
    y_preds = densenet121(input_low[0,:,:,:,:]) # (B, T, C, H, W) -> (B*T, C, H, W); (1*15, 3, 64, 64) -> (15, 1)
    y_pred = torch.sigmoid(y_preds).detach().cpu().numpy().squeeze() # (15,)
    visualize_blink_probabilities(y_pred, (gt!=-1).astype(int), f"[TalkingFace] positive {SAMPLE_IDX} (DenseNet121)", DB_PROCESSED_DIR / 'output' / f'TalkingFace_pos_{SAMPLE_IDX}_DenseNet121.png')

    # sequence model, eye patches only
    y_preds = blinklint(input_low) # (B, T, C, H, W); (1, 15, 3, 64, 64) -> (1, 15, 1)
    y_pred = torch.sigmoid(y_preds).detach().cpu().numpy().squeeze() # (15,)
    visualize_blink_probabilities(y_pred, (gt!=-1).astype(int), f"[TalkingFace] positive {SAMPLE_IDX} (BlinkLinT)", DB_PROCESSED_DIR / 'output' / f'TalkingFace_pos_{SAMPLE_IDX}_BlinkLinT.png')

    # sequence model, eye patches and additional calculated features
    y_clss, y_preds = blinklinmult([input_low, input_high]) # [(B, T, C, H, W), (B, T, F)]; [(1, 15, 3, 64, 64), (1, 15, 160)] -> (1, 15, 1)
    y_cls = torch.sigmoid(y_clss).detach().cpu().numpy().squeeze() # ()
    y_pred = torch.sigmoid(y_preds).detach().cpu().numpy().squeeze() # (15,)
    visualize_blink_probabilities(y_pred, (gt!=-1).astype(int), f"[TalkingFace] positive {SAMPLE_IDX} (BlinkLinMulT)", DB_PROCESSED_DIR / 'output' / f'TalkingFace_pos_{SAMPLE_IDX}_BlinkLinMulT.png')

print('Cheers!')
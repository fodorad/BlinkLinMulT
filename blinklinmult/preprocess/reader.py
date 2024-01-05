import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import bbox_visualizer as bbv
from exordium.video.io import frames2video
from exordium.video.bb import xywh2xyxy
from blinklinmult import PathType


class Tag:

    def __init__(self, tag_path: PathType,
                       timestamp_path: PathType | None = None,
                       video_path: PathType | None = None,
                       frames_dir: PathType | None = None,
                       extension: str = ".png") -> None:
        """Blink annotation (.tag) file description.

        Keys:
            'frame ID': Frame counter based on which the time-stamp can be obtained (separate file).
            'blink ID': Unique blink ID, eye blink interval is defined as a sequence of the same blink ID frames.
            'NF': non frontal face. While subject is looking sideways and eye blink occured, given variable changes from X to N.
            'LE': left eye.
            'RE': right eye.
            'FC': eye fully closed. If subject's eyes are closed from 90% to 100%, given flag will change from X to C.
            'NV': eye not visible. While subject's eye is not visible because of hand, bad light conditions, hair or even too fast head movement, this variable changes from X to N.
            ['F_X', 'F_Y', 'F_W', 'F_H']: face bounding box. x and y coordinates, width height.
            'RX': corner positions. Right corner x coordinate.
            'LY': corner positions. Left corner y coordinate.

        The authors claimed that 'NF', 'FC' and 'NV' are not consistently annotated.

        Args:
            tag_path (str | Path): path to the tag file.
            timestamp_path (str | Path, optional): path to the timestamp-frame mapping file. Defaults to Path(tag_path).parent/f'{Path(tag_path).stem}.txt'
            video_path (str | Path, optional): path to the original video. Defaults to Path(tag_path).parent/f'{Path(tag_path).stem}.mp4'
            frames_dir (str | Path, optional): path to the frames. frames expected format is {id:06d}.extension. Defaults to Path(tag_path).parent/'frames'.
            extension (str): extension of the images.
        """
        self.tag_path = Path(tag_path).resolve()
        self.video_id = self.tag_path.parent.name
        self.timestamp_path = (Path(timestamp_path) if timestamp_path is not None else self.tag_path.parent / f"{self.tag_path.stem}.txt")
        self.video_path = (Path(video_path) if video_path is not None else self.tag_path.parent / f"{self.tag_path.stem}.avi")
        self.frames_dir = (Path(frames_dir) if frames_dir is not None else self.tag_path.parent / "frames")
        self.extension = extension

        assert self.tag_path.exists(), f"File was not found: {self.tag_path}"
        assert (self.timestamp_path.exists()), f"File was not found: {self.timestamp_path}"
        assert self.video_path.exists(), f"File was not found: {self.video_path}"
        assert (self.frames_dir.is_dir()), f"Directory of the frame is invalid: {self.frames_dir}"

        with open(self.timestamp_path, "r") as f:
            frame_timestamp = f.readlines()

        frame_timestamp = [elem.replace("\n", "").split(" ") for elem in frame_timestamp]
        self.frame_timestamp = {int(frame_id): float(timestamp) for frame_id, timestamp in frame_timestamp}
        self.frame_paths = [self.frames_dir / name for name in sorted(os.listdir(self.frames_dir))]

        keys_type = [int] * 2 + [lambda x: x != "X"] * 5 + [int] * 12
        keys = ["frame ID", "blink ID", "NF",
                "LE_FC", "LE_NV", "RE_FC", "RE_NV",
                "F_X", "F_Y", "F_W", "F_H",
                "LE_LX", "LE_LY", "LE_RX", "LE_RY",
                "RE_LX", "RE_LY", "RE_RX", "RE_RY"]

        with open(self.tag_path, "r") as f:
            lines = f.readlines()

        start = next((i for (i, line) in enumerate(lines) if line.replace("\n", "") == "#start"), None)
        end = next((i for (i, line) in enumerate(lines) if line.replace("\n", "") == "#end"), None)

        assert (start is not None), f"Invalid #start token index ({start}). Check the annotation file: {self.tag_path}"
        assert (end is not None), f"Invalid #end token index ({end}). Check the annotation file: {self.tag_path}"
        assert (end > start), f"Invalid intervals. #end ({end}) should be bigger than #start ({start}). Check the annotation file: {self.tag_path}"

        self.annotation = []
        for line in lines[start + 1:end]:
            parts = line.split(":")

            assert len(parts) == len(keys), f"Invalid line. Length of the record is {len(parts)}, while the length of the keys is {len(keys)}"
            record = {k: keys_type[i](v) for i, (k, v) in enumerate(dict(zip(keys, parts)).items())}

            self.annotation.append({
                "frame_id": record["frame ID"],
                "blink_id": record["blink ID"],
                "non_frontal_face": record["NF"],
                "left_fully_closed": record["LE_FC"],
                "left_non_visible": record["LE_NV"],
                "right_fully_closed": record["RE_FC"],
                "right_non_visible": record["RE_NV"],
                "bb_xywh": np.array([record["F_X"], record["F_Y"], record["F_W"], record["F_H"]]),
                "left_eye_xy": np.array([record["LE_LX"], record["LE_LY"], record["LE_RX"], record["LE_RY"]]),
                "right_eye_xy": np.array([record["RE_LX"], record["RE_LY"], record["RE_RX"], record["RE_RY"]]),
            })

        # if the annotation starts with 0, but the frames start with 1, then increment the ids within the annotation to match the frames
        if self.annotation[0]["frame_id"] == 0 and int(self.frame_paths[0].stem) == 1:
            for record in self.annotation: record["frame_id"] += 1

        # record with id 42 should represent frame 000042.png
        # keep only those annotations, which have corresponding frames
        self.annotation = list(filter(lambda record: (self.frames_dir / f'{record["frame_id"]:06d}.png').exists(), self.annotation))
        ids, counts = np.unique([elem["blink_id"] for elem in self.annotation], return_counts=True)
        self.ids_counts = dict(zip(ids, counts))

        self.extra_percent_face = 0.5
        self.extra_percent_eye = 1

    def __getitem__(self, frame_id: int) -> dict:
        """Get the first annotation with the given frame id

        Args:
            frame_id (int): frame is identified by this number
                example: if frame name is 000042.png, frame id is 42

        Returns:
            dict: annotation of the requested frame.
        """
        annotation = next((elem for elem in self.annotation if elem["frame_id"] == frame_id), None)

        if annotation is None:
            raise ValueError(f"None value is returned. There is no frame with the requested frame_id ({frame_id})")

        return annotation

    def blink_label(self, frame_id: int) -> int:
        annotation = self.__getitem__(frame_id)
        return annotation["blink_id"]

    def first_frame(self, blink_id: int) -> dict:
        """Gets the first frame id of a given annotated blink.

        Args:
            blink_id (int): blink id. -1 for open eye, positive number for closed eye.

        Returns:
            dict: 'index' points to the index of the annotation within the list of all annotations
                'frame' points to the first frame id with the requested blink id.
        """
        annotation = next((elem for elem in self.annotation if elem["blink_id"] == blink_id), None)

        if annotation is None:
            raise ValueError(f"None value is returned. There is no annotation with the requested blink_id ({blink_id})")

        return {
            "index": self.annotation.index(annotation),
            "frame": annotation["frame_id"],
        }

    def frame_ids(self) -> list[int]:
        """Gets all available frame ids.

        Returns:
            list: sorted frame ids of the annotations.
        """
        return sorted({elem["frame_id"] for elem in self.annotation})

    def blink_ids(self) -> list[int]:
        """Gets all available blink ids (-1 means open eye, positive number means closed eye).

        Returns:
            list[int]: sorted blink ids.
        """
        return sorted(list(self.ids_counts.keys()))

    def __len__(self) -> int:
        """The length of the Tag file is the number of unique frame ids.

        Returns:
            int: number of unique frame ids.
        """
        return len(self.frame_ids())

    def print(self) -> None:
        """Print elements of the Tag file, calculate the counts of unique blink ids."""
        print(self.annotation)
        unique, counts = np.unique([elem["blink_id"] for elem in self.annotation], return_counts=True)

        for u, c in zip(unique, counts):
            print("value:", u, "count:", c)

        print("Binary annotation stats:", np.unique(self.binary(), return_counts=True))

    def get_positive_sample(self, blink_id: int,
                                  win_size: int = 15,
                                  output_dir: str | Path | None = None,
                                  fps: float = 30.0) -> tuple[list[str], np.ndarray]:
        """Get the blink sample with the associated blink id.

        Args:
            blink_id (int): blink id.
            win_size (int, optional): window size. The sample will be a sequence of frame paths, which has win_size length.
                The average length of a blink is 0.6 sec. In the case of 30 fps, it is 15-frame-long. -1 means that the win_size
                is calculated using average blink length and the fps. Defaults to 15.
            output_dir (str | Path | None, optional): save sample to this path. Defaults to None.
            fps (float, optional): fps of the saved video. Defaults to 30.

        Returns:
            tuple[list[str], np.ndarray]: list of frame paths, vector of labels with shape=(T,).
        """
        if win_size == -1:
            win_size = int(np.rint(fps * 0.5))

        blink_length = self.ids_counts[blink_id]
        blink_start = self.first_frame(blink_id)

        # normal use case, get the win_size long sequence
        if blink_length > win_size:
            start = blink_start["index"]
            end = start + win_size
        else:  # blinks shorter than the window size will be padded
            pad = (win_size - blink_length) // 2
            start = blink_start["index"] - pad
            end = start + win_size

        # blinks are at the very beginning of the tag file
        if start < 0:
            start = 0
            end = start + win_size

        # blinks are at the very end of the tag file
        if end >= self.__len__():
            end = self.__len__()
            start = end - win_size

        elems = self.annotation[start:end]
        frame_ids = [elem["frame_id"] for elem in elems]
        frame_paths = [str(Path(self.frames_dir) / f"{frame_id:06d}.png") for frame_id in frame_ids]
        labels = np.array([elem["blink_id"] for elem in elems])

        if output_dir is not None:
            frames2video(frame_paths, output_path=str(Path(output_dir) / f"{self.video_id}_{frame_ids[0]}_{frame_ids[-1]}.mp4"), fps=fps)

            for frame_path, gt in zip(frame_paths, list(labels)):
                output_dir_frame = (Path(output_dir) / f"{self.video_id}_{frame_ids[0]}_{frame_ids[-1]}")
                output_dir_frame.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_dir_frame / f"{Path(frame_path).stem}_{gt}.png"), cv2.imread(frame_path))

        return frame_paths, labels

    def generate_positive_samples(self, win_size: int = 15,
                                        output_dir: str | Path | None = None,
                                        fps: float = 30.0) -> list[tuple[list[str], np.ndarray]]:
        """Generate all blink samples annotated in this Tag file.

        Args:
            win_size (int, optional): window size. Defaults to 18.
            output_dir (str | Path | None, optional): save sample to this path. Defaults to None.
            fps (float, optional): fps of the saved video. Defaults to 30.

        Returns:
            list[tuple[list[str], np.ndarray]]: list of frame paths, vector of labels with shape=(T,).
        """
        blink_ids = self.blink_ids()
        blink_ids.remove(-1)

        positive_samples = []
        for blink_id in tqdm(blink_ids, desc="Generating positive (blink) samples"):
            positive_samples.append(self.get_positive_sample(blink_id, win_size=win_size, output_dir=output_dir, fps=fps))

        return positive_samples

    def generate_negative_samples(self, num_samples: int,
                                        win_size: int = 15,
                                        output_dir: str | Path | None = None,
                                        fps: float = 30.0) -> list[tuple[list[str], np.ndarray]]:
        """Generate open eye samples from this Tag file.

        Args:
            num_samples (int): number of randomly generated non-blink samples.
            win_size (int, optional): window size. The sample will be a sequence of frame paths, which has win_size length.
                The average length of a blink is 0.6 sec. In the case of 30 fps, it is 18-frame-long. -1 means that the win_size
                is calculated using average blink length and the fps. Defaults to 18.
            output_dir (str | Path | None, optional): save sample to this path. Defaults to None.
            fps (float, optional): fps of the saved video. Defaults to 30.

        Returns:
            list[tuple[list[str], np.ndarray]]: list of frame paths, vector of labels with shape=(T,).
        """
        np.random.seed(42)

        if win_size == -1:
            win_size = int(np.rint(fps * 0.6))

        with tqdm(total=num_samples, desc="Generating negative (non-blink) samples") as pbar:

            negative_samples = []
            while len(negative_samples) < num_samples:
                start = np.random.randint(0, self.__len__() - win_size)
                end = start + win_size

                elems = self.annotation[start:end]
                frame_ids = [elem["frame_id"] for elem in elems]
                frame_paths = [str(Path(self.frames_dir) / f"{frame_id:06d}.png") for frame_id in frame_ids]
                labels = np.array([elem["blink_id"] for elem in elems])

                if not np.all(labels == -1):
                    continue

                negative_samples.append((frame_paths, labels))
                pbar.update(1)

                if output_dir is not None:
                    frames2video(frame_paths, output_path=str(Path(output_dir) / f"{self.video_id}_{frame_ids[0]}_{frame_ids[-1]}.mp4"), fps=fps)

                    for frame_path, gt in zip(frame_paths, list(labels)):
                        output_dir_frame = (Path(output_dir) / f"{self.video_id}_{frame_ids[0]}_{frame_ids[-1]}")
                        output_dir_frame.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(output_dir_frame / f"{Path(frame_path).stem}_{gt}.png"), cv2.imread(frame_path))

        return negative_samples

    @staticmethod
    def draw_landmarks(frame: np.ndarray, annotation: dict) -> np.ndarray:
        """Draw annotation landmarks to the frame.

        Args:
            frame (np.ndarray): image represented with np.ndarray, shape=(H,W,3) and BGR channel order.
            annotation (dict): annotation dictionary from a Tag file.

        Returns:
            np.ndarray: image with annotation burned on it.
        """
        bb_xyxy = xywh2xyxy(annotation["bb_xywh"])
        frame = bbv.draw_rectangle(frame, bb_xyxy)
        frame = bbv.add_label(frame,
            f"{annotation['blink_id']}|"
            f"{int(annotation['non_frontal_face'])}|"
            f"{int(annotation['left_fully_closed'])}|"
            f"{int(annotation['left_non_visible'])}|"
            f"{int(annotation['right_fully_closed'])}|"
            f"{int(annotation['right_fully_closed'])}",
            bb_xyxy,
        )
        cv2.circle(frame, annotation["left_eye_xy"][:2], color=(0, 0, 255), radius=2, thickness=1)
        cv2.circle(frame, annotation["left_eye_xy"][2:], color=(0, 0, 160), radius=2, thickness=2)
        cv2.circle(frame, annotation["right_eye_xy"][:2], color=(0, 255, 0), radius=2, thickness=1)
        cv2.circle(frame, annotation["right_eye_xy"][2:], color=(0, 160, 0), radius=2, thickness=2)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def crop_face(self, frame: np.ndarray, bb_xywh: np.ndarray | list) -> np.ndarray:
        """Cut the face from a frame using the higher bounding box side

        Args:
            frame (np.ndarray): image containing the participant's face
            bb_xywh (np.ndarray | list): bounding box (x, y, w, h) order

        Returns:
            np.ndarray: cropped face
        """
        bb_size = int(max(bb_xywh[2:]) * (1 + self.extra_percent_face))
        # centre point
        cx = bb_xywh[0] + bb_xywh[2] // 2
        cy = bb_xywh[1] + bb_xywh[3] // 2
        # normalized bounding box
        nx1 = max([cx - bb_size // 2, 0])
        nx2 = min([cx + bb_size // 2, frame.shape[1]])
        ny1 = max([cy - bb_size // 2, 0])
        ny2 = min([cy + bb_size // 2, frame.shape[0]])
        return frame[ny1:ny2, nx1:nx2, :]

    def crop_eyes(self, frame: np.ndarray, left_eye_xy: np.ndarray | list, right_eye_xy: np.ndarray | list) -> tuple[np.ndarray, np.ndarray]:
        """Cut the eyes from a frame using the bigger side of the bounding box

        Args:
            frame (np.ndarray): image containing the participant's face
            left_eye_xy (np.ndarray | list): two annotated corner points of the left eye (x1, y1, x2, y2) order
            right_eye_xy (np.ndarray | list): two annotated corner points of the right eyes (x1, y1, x2, y2) order

        Returns:
            np.ndarray: cropped eyes
        """
        bb_size_eye = int(max([np.linalg.norm(left_eye_xy[:2] - left_eye_xy[2:]),
                               np.linalg.norm(right_eye_xy[:2] - right_eye_xy[2:])]) * (1 + self.extra_percent_eye))
        # centre point
        left_cx = left_eye_xy[0] + abs(left_eye_xy[0] - left_eye_xy[2]) // 2
        # face is not aligned using the manual annotation
        left_cy = (min(left_eye_xy[1], left_eye_xy[3]) + abs(left_eye_xy[1] - left_eye_xy[3]) // 2)
        right_cx = right_eye_xy[0] + abs(right_eye_xy[0] - right_eye_xy[2]) // 2
        right_cy = (min(right_eye_xy[1], right_eye_xy[3]) + abs(right_eye_xy[1] - right_eye_xy[3]) // 2)
        # normalized bounding box
        left_nx1 = max([left_cx - bb_size_eye // 2, 0])
        left_nx2 = min([left_cx + bb_size_eye // 2, frame.shape[1]])
        left_ny1 = max([left_cy - bb_size_eye // 2, 0])
        left_ny2 = min([left_cy + bb_size_eye // 2, frame.shape[0]])
        right_nx1 = max([right_cx - bb_size_eye // 2, 0])
        right_nx2 = min([right_cx + bb_size_eye // 2, frame.shape[1]])
        right_ny1 = max([right_cy - bb_size_eye // 2, 0])
        right_ny2 = min([right_cy + bb_size_eye // 2, frame.shape[0]])
        return (
            frame[left_ny1:left_ny2, left_nx1:left_nx2, :],
            frame[right_ny1:right_ny2, right_nx1:right_nx2, :],
        )

    def visualize(self, output_path: str | Path, fps: float = 30.0) -> None:
        """Visualize annotation on the full video.

        Args:
            output_path (str | Path): save video to this path.
            fps (float, optional): fps of the saved video. Defaults to 30.
        """
        frames = [Tag.draw_landmarks(cv2.imread(str(self.frames_dir / f'{annotation["frame_id"]:06d}.png')), annotation)
                  for annotation in self.annotation]
        print(f"Writing video: {output_path}")
        frames2video(input_path=frames, output_path=output_path, fps=fps, extension=self.extension, overwrite=True)
        print(f"Done: {output_path}")

    def save_annotated_face_crops(self, output_dir: str | Path) -> None:
        """Save manually annotated face crops as images

        Args:
            output_dir (str | Path): directory path for the face images
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        for annotation in tqdm(self.annotation, total=len(self.annotation), desc="crop faces"):
            if not all([elem == 0 for elem in list(annotation["bb_xywh"])]):
                img = cv2.imread(str(self.frames_dir / f'{annotation["frame_id"]:06d}.png'))
                face = self.crop_face(img, annotation["bb_xywh"])
                cv2.imwrite(str(output_dir / f'{annotation["frame_id"]:06d}.png'), face)

    def save_annotated_eye_crops(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir).resolve()
        (output_dir / "left").mkdir(parents=True, exist_ok=True)
        (output_dir / "right").mkdir(parents=True, exist_ok=True)

        for annotation in tqdm(self.annotation, total=len(self.annotation), desc="crop eyes"):
            if not all([elem == 0 for elem in list(annotation["bb_xywh"])]):
                img = cv2.imread(str(self.frames_dir / f'{annotation["frame_id"]:06d}.png'))
                left_eye, right_eye = self.crop_eyes(img, annotation["left_eye_xy"], annotation["right_eye_xy"])
                cv2.imwrite(str(output_dir / "left" / f'{annotation["frame_id"]:06d}.png'), left_eye)
                cv2.imwrite(str(output_dir / "right" / f'{annotation["frame_id"]:06d}.png'), right_eye)

    def map_annotation_to_frames_decord(self, output_dir: str | Path) -> None:
        """Frames are generated based on timestamps predefined in a supplementary file and the annotation file

        Args:
            output_dir (str | Path): the frames are generated to this directory.
        """
        import decord
        decord.bridge.set_bridge("torch")
        self.frames_dir = Path(output_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        try:
            vr = decord.VideoReader(str(self.video_path), ctx=decord.cpu(0))
            img_shape = vr[0].shape
        except:
            print("video cannot be loaded. convert to mp4, then continue.")
            video_path_src = self.video_path
            self.video_path = self.frames_dir.parent / f"{self.video_path.stem}.mp4"
            if not self.video_path.exists():
                CMD_STR = f"ffmpeg -y -i {str(video_path_src)} {str(self.video_path)}"
                print(CMD_STR)
                os.system(CMD_STR)
            vr = decord.VideoReader(str(self.video_path), ctx=decord.cpu(0))
            img_shape = vr[0].shape
        print(f"Properties of {self.video_path}")
        print(f"Number of frames:", len(vr))
        print("Image shape is:", img_shape)
        timestamps = np.array(vr.get_frame_timestamp(range(len(vr))))[:, 1]  # [[start, end],...]
        for frame_id, gt_timestamp in tqdm(self.frame_timestamp.items(), total=len(self.frame_timestamp), desc="Extract frames"):
            min_index = np.argmin(np.absolute(timestamps - gt_timestamp))
            img = cv2.cvtColor(np.array(vr[min_index]), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(self.frames_dir / f"{frame_id:06d}.png"), img)

    def map_annotation_to_frames_cv2(self, output_dir: str | Path) -> None:
        """Frames are generated based on timestamps predefined in a supplementary file and the annotation file

        Args:
            output_dir (str | Path): the frames are generated to this directory.
        """
        self.frames_dir = Path(output_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        video_path_src = self.video_path
        self.video_path = self.frames_dir.parent / f"{self.video_path.stem}.mp4"
        if not self.video_path.exists():
            CMD_STR = f"ffmpeg -y -i {str(video_path_src)} {str(self.video_path)}"
            print(CMD_STR)
            os.system(CMD_STR)

        cap = cv2.VideoCapture(str(self.video_path))

        print(f"Properties of {self.video_path}")
        print(f"Number of frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        timestamps = []
        while cap.isOpened():
            frame_exists, curr_frame = cap.read()

            if frame_exists:
                frames.append(curr_frame)
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            else:
                break

        cap.release()
        timestamps = np.array(timestamps)

        for frame_id, gt_timestamp in tqdm(self.frame_timestamp.items(), total=len(self.frame_timestamp), desc="Extract frames"):
            min_index = np.argmin(np.absolute(timestamps - gt_timestamp))
            image = frames[min_index]
            cv2.imwrite(str(self.frames_dir / f"{frame_id:06d}.png"), image)
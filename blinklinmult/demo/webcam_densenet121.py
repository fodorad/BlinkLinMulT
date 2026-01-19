#!/usr/bin/env python3
"""
Real-Time Blink Detection Demo using MediaPipe + DenseNet121 + PyTorch
Ultra-fast face detection (60 fps) with blink inference
"""
import os
from collections import deque
import time
import threading
import urllib.request
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import mediapipe as mp

# ============================================================================
# SETUP
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Real-Time Blink Detection Demo")
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: mps)"
    )
    return parser.parse_args()


def load_blink_model(device):
    """Load DenseNet121 model for blink detection"""
    print(f"Loading DenseNet121 on {device.upper()}...")
    from blinklinmult.models import DenseNet121
    model = DenseNet121(weights='densenet121-union').eval().to(device)
    print("✓ Blink model loaded")
    return model

def load_face_detector():
    """Load MediaPipe face detector"""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    def download_model(model_name="blaze_face_short_range.tflite"):
        """Download MediaPipe face detection model if not exists"""
        model_dir = "/tmp/mediapipe_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)
        
        if not os.path.exists(model_path):
            print(f"Downloading {model_name}...")
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"✓ Model downloaded to {model_path}")
            except Exception as e:
                print(f"✗ Download failed: {e}")
                return None
        
        return model_path

    model_path = download_model()
    if model_path is None:
        raise RuntimeError("Could not load face detection model")
    
    print("Loading MediaPipe Face Detection...")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    print("✓ Face detector loaded")
    return detector

# Image transforms
test_transform = transforms.Compose([
    transforms.Resize((64, 64), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============================================================================
# STATE
# ============================================================================

class DemoState:
    """Demo state management"""
    def __init__(self, device):
        self.device = device
        self.blink_model = load_blink_model(device)
        self.face_detector = load_face_detector()
        
        # Buffers for both eyes
        self.left_eye_history = deque(maxlen=60)  # 2s @ 30fps
        self.right_eye_history = deque(maxlen=60)
        self.frame_times = deque(maxlen=60)
        self.fps = 0.0
        self.frame_count = 0


class FPSCounter:
    def __init__(self, buffer=60):
        self.buffer = deque(maxlen=buffer)
        self.start = None
    
    def update(self):
        now = time.time()
        self.buffer.append(now)
        if self.start is None:
            self.start = now
    
    def get_fps(self):
        if len(self.buffer) < 2:
            return 0
        return len(self.buffer) / (self.buffer[-1] - self.buffer[0])


class WebcamStream:

    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.fps_counter = FPSCounter()
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.running = False
    
    def _capture_loop(self):
        while self.running:
            ret = self.cap.grab()
            if ret:
                ret, frame = self.cap.retrieve()
                if ret:
                    self.frame = frame
    
    def read(self):
        frame = self.frame
        self.fps_counter.update()
        return frame
    
    def __del__(self):
        self.running = False
        self.cap.release()
    
    def start(self):
        self.running = True
        self.thread.start()
        return self

    def stop(self):
        self.running = False
        self.thread.join()


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_dual_plot_image(left_history, right_history, width=640, height=200):
    """Create side-by-side plot images for left and right eye blink predictions"""
    if len(left_history) == 0 and len(right_history) == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create figure with 2 subplots side by side
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(width/100, height/100), dpi=100)
    
    # Left eye plot
    if len(left_history) > 0:
        x = np.arange(len(left_history))
        ax_left.plot(x, list(left_history), 'b-', linewidth=2)
    ax_left.set_title('Left Eye', fontsize=10)
    ax_left.set_ylim(0, 1)
    ax_left.set_xlim(max(0, len(left_history) - 60), max(len(left_history), 1))
    ax_left.set_ylabel('Blink Prob')
    ax_left.grid(True, alpha=0.3)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    
    # Right eye plot
    if len(right_history) > 0:
        x = np.arange(len(right_history))
        ax_right.plot(x, list(right_history), 'r-', linewidth=2)
    ax_right.set_title('Right Eye', fontsize=10)
    ax_right.set_ylim(0, 1)
    ax_right.set_xlim(max(0, len(right_history) - 60), max(len(right_history), 1))
    ax_right.set_ylabel('Blink Prob')
    ax_right.grid(True, alpha=0.3)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    
    plt.tight_layout(pad=0)
    
    # Convert to image using buffer_rgba
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    image = np.frombuffer(buf, dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image = cv2.resize(image, (width, height))
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    
    plt.close(fig)
    return image_bgr

def process_frame(frame_bgr, state):
    """Process single frame and return blink predictions for both eyes"""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, c = frame_rgb.shape
    
    try:
        # Face detection with MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = state.face_detector.detect(mp_image)
        
        if not detection_result.detections:
            return None, None
        
        # Get first (largest) face
        detection = detection_result.detections[0]
        
        # Get eye landmarks (keypoints 0=left_eye, 1=right_eye)
        if not detection.keypoints or len(detection.keypoints) < 2:
            return None, None
        
        # Extract eye coordinates
        left_eye = detection.keypoints[0]
        right_eye = detection.keypoints[1]
        
        left_x = int(left_eye.x * w)
        left_y = int(left_eye.y * h)
        right_x = int(right_eye.x * w)
        right_y = int(right_eye.y * h)
        
        # Calculate eye distance for crop size
        eye_distance = abs(right_x - left_x)
        if eye_distance < 10:
            return None, None
        
        crop_size = int(eye_distance * 1.5)
        
        # Process left eye
        left_pred = None
        x1 = max(0, left_x - crop_size // 2)
        x2 = min(w, left_x + crop_size // 2)
        y1 = max(0, left_y - crop_size // 2)
        y2 = min(h, left_y + crop_size // 2)
        
        left_eye_crop = frame_bgr[y1:y2, x1:x2]
        
        # Process right eye
        right_pred = None
        x1 = max(0, right_x - crop_size // 2)
        x2 = min(w, right_x + crop_size // 2)
        y1 = max(0, right_y - crop_size // 2)
        y2 = min(h, right_y + crop_size // 2)
        
        right_eye_crop = frame_bgr[y1:y2, x1:x2]
        
        # Batch inference on both eyes
        if left_eye_crop.size > 0 and right_eye_crop.size > 0:
            left_eye_rgb = cv2.cvtColor(left_eye_crop, cv2.COLOR_BGR2RGB)
            right_eye_rgb = cv2.cvtColor(right_eye_crop, cv2.COLOR_BGR2RGB)
            
            left_patch = test_transform(Image.fromarray(left_eye_rgb))
            right_patch = test_transform(Image.fromarray(right_eye_rgb))
            
            # Batch both eyes together
            batch = torch.stack([left_patch, right_patch]).to(state.device)
            
            with torch.no_grad():
                preds = torch.sigmoid(state.blink_model(batch))
            
            left_pred = preds[0].item()
            right_pred = preds[1].item()
        elif left_eye_crop.size > 0:
            left_eye_rgb = cv2.cvtColor(left_eye_crop, cv2.COLOR_BGR2RGB)
            eye_patch = test_transform(Image.fromarray(left_eye_rgb))
            eye_patch = eye_patch.unsqueeze(0).to(state.device)
            
            with torch.no_grad():
                left_pred = torch.sigmoid(state.blink_model(eye_patch)).item()
        elif right_eye_crop.size > 0:
            right_eye_rgb = cv2.cvtColor(right_eye_crop, cv2.COLOR_BGR2RGB)
            eye_patch = test_transform(Image.fromarray(right_eye_rgb))
            eye_patch = eye_patch.unsqueeze(0).to(state.device)
            
            with torch.no_grad():
                right_pred = torch.sigmoid(state.blink_model(eye_patch)).item()
        
        return left_pred, right_pred
    except Exception as e:
        return None, None


# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    args = parse_args()
    state = DemoState(args.device)
    
    print(f"\n{'='*60}")
    print(f"Real-Time Blink Detection Demo")
    print(f"Device: {args.device.upper()}")
    print(f"Press 'q' to quit")
    print(f"{'='*60}\n")

    print("Starting capture loop...\n")
    
    try:
        vs = WebcamStream()
        vs.start()
        while True:
            frame = vs.read()
            if frame is None:
                continue
            
            # Flip for selfie view
            frame = cv2.flip(frame, 1)
            
            # Process frame (returns both left and right eye predictions)
            left_pred, right_pred = process_frame(frame, state)
            
            if left_pred is not None or right_pred is not None:
                if left_pred is not None:
                    state.left_eye_history.append(left_pred)
                if right_pred is not None:
                    state.right_eye_history.append(right_pred)
                state.frame_times.append(time.time())
                state.frame_count += 1
                
                # Calculate FPS from WebcamStream
                state.fps = vs.fps_counter.get_fps()
            
            # Draw FPS on frame
            fps_text = f"FPS: {state.fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw blink probabilities
            y_offset = 70
            if left_pred is not None:
                prob_text = f"Left Eye: {left_pred:.2f}"
                color = (0, 0, 255) if left_pred > 0.5 else (0, 255, 0)
                cv2.putText(frame, prob_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, color, 2)
                y_offset += 40
            
            if right_pred is not None:
                prob_text = f"Right Eye: {right_pred:.2f}"
                color = (0, 0, 255) if right_pred > 0.5 else (0, 255, 0)
                cv2.putText(frame, prob_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, color, 2)
            
            # Display frame
            cv2.imshow("Real-Time Webcam Demo: Blink Detection", frame)
            
            # Create and display dual plot only every 10 frames
            if state.frame_count % 10 == 0:
                plot_image = create_dual_plot_image(state.left_eye_history, state.right_eye_history, width=640, height=200)
                cv2.imshow("Blink Prediction History", plot_image)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print("\n✓ Demo finished")
        print(f"Total frames processed: {state.frame_count}")
        print(f"Average FPS: {state.fps:.1f}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    finally:
        state.face_detector.close()
        vs.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

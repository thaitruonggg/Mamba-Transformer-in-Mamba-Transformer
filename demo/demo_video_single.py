"""
Behavior:
  - YOLO detects and tracks all traffic signs across frames.
  - MaMa-MoEx classifies each detected crop.
  - If the classifier is confident  -> green box + class name (remembered by tracker).
  - If the classifier is NOT confident -> orange box + "Misc (Unknown Sign)".
  - Every detected sign is always shown — nothing is hidden.
"""

import argparse
import cv2
import time
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

from MaMa_MoEx import MaMa_MoEx_Ti

# ==========================================
# DEFAULT CONFIGURATION
# ==========================================
DEFAULT_DETECTOR_PATH = "../yolo11_model/gtsdbbest.pt"
DEFAULT_CLASSIFIER_PATH = "pretrained_models/mama_moex_model.pth"
DEFAULT_INPUT_VIDEO = "test_images/1.mp4"
DEFAULT_OUTPUT_VIDEO = "result_video.mp4"

# Thresholds
DEFAULT_DETECTION_CONFIDENCE = 0.4    # YOLO: minimum confidence to consider a detection (0.0–1.0)
DEFAULT_CLASSIFIER_CONFIDENCE = 85.0  # MaMa: minimum softmax confidence to trust a class (0.0–100.0)
DEFAULT_ENTROPY_THRESHOLD = 1.5       # MaMa: reject if probability distribution is too flat (uncertain)
# ==========================================

# GTSRB class names (43 classes)
GTSRB_CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)",
    "Speed limit (100km/h)", "Speed limit (120km/h)", "No passing",
    "No passing for vehicles over 3.5t", "Right-of-way at the next intersection",
    "Priority road", "Yield", "Stop", "No vehicles",
    "Vehicles over 3.5t prohibited", "No entry", "General caution",
    "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road",
    "Road narrows on the right", "Road work", "Traffic signals",
    "Pedestrians", "Children crossing", "Bicycles crossing",
    "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead",
    "Turn left ahead", "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right", "Keep left",
    "Roundabout mandatory", "End of no passing",
    "End of no passing for vehicles over 3.5 metric tons"
]

# TT100K class names (232 classes)
TT100K_CLASS_NAMES = [
    "pl80", "w9", "p6", "ph4.2", "i8", "w14", "w33", "pa13", "im", "w58",
    "pl90", "il70", "p5", "pm55", "pl60", "ip", "p11", "pdd", "wc", "i2r",
    "w30", "pmr", "p23", "pl15", "pm10", "pss", "w1", "p4", "w38", "w50",
    "w34", "pw3.5", "iz", "w39", "w11", "p1n", "pr70", "pd", "pnl", "pg",
    "ph5.3", "w66", "il80", "pb", "pbm", "pm5", "w24", "w67", "w49", "pm40",
    "ph4", "w45", "i4", "w37", "ph2.6", "pl70", "ph5.5", "i14", "i11", "p7",
    "p29", "pne", "pr60", "pm13", "ph4.5", "p12", "p3", "w40", "pl5", "w13",
    "pr10", "p14", "i4l", "pr30", "pw4.2", "w16", "p17", "ph3", "i9", "w15",
    "w35", "pa8", "pt", "pr45", "w17", "pl30", "pcs", "pctl", "pr50", "ph4.4",
    "pm46", "pm35", "i15", "pa12", "pclr", "i1", "pcd", "pbp", "pcr", "w28",
    "ps", "pm8", "w18", "w2", "w52", "ph2.9", "ph1.8", "pe", "p20", "w36",
    "p10", "pn", "pa14", "w54", "ph3.2", "p2", "ph2.5", "w62", "w55", "pw3",
    "pw4.5", "i12", "ph4.3", "phclr", "i10", "pr5", "i13", "w10", "p26", "w26",
    "p8", "w5", "w42", "il50", "p13", "pr40", "p25", "w41", "pl20", "ph4.8",
    "pnlc", "ph3.3", "w29", "ph2.1", "w53", "pm30", "p24", "p21", "pl40", "w27",
    "pmb", "pc", "i6", "pr20", "p18", "ph3.8", "pm50", "pm25", "i2", "w22",
    "w47", "w56", "pl120", "ph2.8", "i7", "w12", "pm1.5", "pm2.5", "w32", "pm15",
    "ph5", "w19", "pw3.2", "pw2.5", "pl10", "il60", "w57", "w48", "w60", "pl100",
    "pr80", "p16", "pl110", "w59", "w64", "w20", "ph2", "p9", "il100", "w31",
    "w65", "ph2.4", "pr100", "p19", "ph3.5", "pa10", "pcl", "pl35", "p15", "w7",
    "pa6", "phcs", "w43", "p28", "w6", "w3", "w25", "pl25", "il110", "p1",
    "w46", "pn-2", "w51", "w44", "w63", "w23", "pm20", "w8", "pmblr", "w4",
    "i5", "il90", "w21", "p27", "pl50", "pl65", "w61", "ph2.2", "pm2", "i3",
    "pa18", "pw4"
]

# Map num_classes -> class name list
CLASS_NAME_MAP = {
    43: ("GTSRB", GTSRB_CLASS_NAMES),
    232: ("TT100K", TT100K_CLASS_NAMES),
}

# Return (dataset_name, class_names) for a given number of classes.
def get_class_names(num_classes):
    if num_classes in CLASS_NAME_MAP:
        return CLASS_NAME_MAP[num_classes]
    print(f"Warning: Unknown dataset with {num_classes} classes. Using generic class names.")
    return ("Unknown", [f"Class_{i}" for i in range(num_classes)])

# Colors (BGR for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_DARK_GREEN = (0, 170, 0)
COLOR_ORANGE = (0, 140, 255)
COLOR_DARK_ORANGE = (0, 100, 204)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# Load the YOLO detection model.
def load_detector(model_path, device):
    print(f"[Detector]  Loading YOLO from: {model_path}")
    detector = YOLO(model_path)
    detector.to(device)
    return detector

def load_classifier(model_path, device):
    """
    Load the MaMa-MoEx classification model.
    Auto-detects num_classes from the checkpoint's head.weight shape.

    Returns:
        classifier : the loaded model
        num_classes: number of output classes detected from checkpoint
    """
    print(f"[Classifier] Loading MaMa-MoEx from: {model_path}")

    # Peek at checkpoint to get num_classes
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['head.weight'].shape[0]
    print(f"[Classifier] Auto-detected {num_classes} output classes from checkpoint")

    classifier = MaMa_MoEx_Ti(pretrained=False)
    classifier.head = torch.nn.Linear(in_features=192, out_features=num_classes, bias=True)
    classifier.load_state_dict(checkpoint)
    classifier.to(device)
    classifier.eval()
    return classifier, num_classes

def compute_entropy(probabilities):
    """
    Compute Shannon entropy of a probability distribution.
    High entropy = model is unsure.  Low entropy = model is focused.
    """
    return -torch.sum(probabilities * torch.log(probabilities + 1e-7)).item()

# Draw a text label with a filled background rectangle above a bounding box.
def draw_label(frame, text, x1, y1, bg_color, text_color=COLOR_WHITE, font_scale=0.6, thickness=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), bg_color, -1)
    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

def process_video(input_path, output_path, detector, classifier, device,
                  detect_thresh, classify_thresh, entropy_thresh, class_names):
    """
    Process a video frame-by-frame with detection + tracking + classification.

    Every tracked detection is drawn:
      - Classified signs  → green box + class name (remembered by tracker)
      - Uncertain signs   → orange box + "Misc (Unknown Sign)"
    """
    if not os.path.exists(input_path):
        print(f"Error: Video not found at '{input_path}'")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{input_path}'")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    classify_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print(f"\n{'='*60}")
    print(f"Processing Video: {input_path}")
    print(f"  Resolution : {width}x{height} @ {fps} FPS")
    print(f"  Frames     : {total_frames}")
    print(f"  Output     : {output_path}")
    print(f"  Detection  : conf >= {detect_thresh}")
    print(f"  Classifier : conf >= {classify_thresh}%, entropy <= {entropy_thresh}")
    print(f"{'='*60}")

    # Tracker memory: once a sign is confidently classified, remember it
    track_memory = {}
    total_latency_ms = 0.0
    valid_frames = 0
    pbar = tqdm(total=total_frames, desc="Exporting Video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Stage 1: Detection + Tracking (YOLO + ByteTrack)
        results = detector.track(frame, conf=detect_thresh, persist=True,
                                 tracker="bytetrack.yaml", verbose=False)[0]

        if results.boxes.id is not None:
            track_ids = results.boxes.id.int().cpu().tolist()
            boxes = results.boxes.xyxy.cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                if x2 <= x1 or y2 <= y1:
                    continue

                # Stage 2: Classify with MaMa-MoEx
                cropped_img = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                crop_tensor = classify_transform(cropped_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = classifier(crop_tensor)
                    probabilities = F.softmax(output[0], dim=0)
                    conf_score, class_idx = torch.max(probabilities, dim=0)
                    entropy = compute_entropy(probabilities)

                conf_percent = conf_score.item() * 100
                class_name = class_names[class_idx.item()]

                # Decision: Classified or Misc?
                is_confident = (conf_percent >= classify_thresh) and (entropy <= entropy_thresh)

                # Update tracker memory only when confident
                if is_confident:
                    track_memory[track_id] = {"name": class_name, "conf": conf_percent}

                # Draw every detection
                if track_id in track_memory:
                    # Classified — use remembered label (green)
                    mem = track_memory[track_id]
                    label_text = f"ID:{track_id} {mem['name']} ({mem['conf']:.1f}%)"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GREEN, 3)
                    draw_label(frame, label_text, x1, y1, COLOR_DARK_GREEN)
                else:
                    # Not classified yet — show as Misc (orange)
                    label_text = f"ID:{track_id} Misc ({conf_percent:.1f}%)"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ORANGE, 3)
                    draw_label(frame, label_text, x1, y1, COLOR_DARK_ORANGE)

        # Timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        latency_sec = end_time - start_time
        latency_ms = latency_sec * 1000.0
        current_fps = 1.0 / latency_sec if latency_sec > 0 else 0.0

        total_latency_ms += latency_ms
        valid_frames += 1

        # Overlay metrics on video
        cv2.rectangle(frame, (10, 10), (280, 80), COLOR_BLACK, -1)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2)
        cv2.putText(frame, f"Latency: {latency_ms:.1f} ms", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2)

        out.write(frame)

        pbar.set_postfix(FPS=f"{current_fps:.1f}", Latency=f"{latency_ms:.1f}ms")
        pbar.update(1)

    # Cleanup
    pbar.close()
    cap.release()
    out.release()

    if valid_frames > 0:
        avg_latency = total_latency_ms / valid_frames
        avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        print(f"\n{'='*45}")
        print(f"FINAL METRICS:")
        print(f"{'='*45}")
        print(f"Total Frames Analyzed: {valid_frames}")
        print(f"Average Latency:       {avg_latency:.2f} ms")
        print(f"Average FPS:           {avg_fps:.2f} FPS")
        print(f"Tracked Signs:         {len(track_memory)} classified, rest shown as Misc")
        print(f"{'='*45}")

    print(f"\nDone! Video saved to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic Sign Detection + Classification Video Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_video_single.py --input TestImages/1.mp4
  python demo_video_single.py --input TestImages/1.mp4 --output result.mp4
  python demo_video_single.py --input TestImages/1.mp4 --detect-conf 0.3 --classify-conf 90
        """
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_VIDEO,
                        help=f"Path to input video (default: {DEFAULT_INPUT_VIDEO})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_VIDEO,
                        help=f"Path to output video (default: {DEFAULT_OUTPUT_VIDEO})")
    parser.add_argument("--detector", type=str, default=DEFAULT_DETECTOR_PATH,
                        help=f"Path to YOLO detector weights (default: {DEFAULT_DETECTOR_PATH})")
    parser.add_argument("--classifier", type=str, default=DEFAULT_CLASSIFIER_PATH,
                        help=f"Path to MaMa-MoEx classifier weights (default: {DEFAULT_CLASSIFIER_PATH})")
    parser.add_argument("--detect-conf", type=float, default=DEFAULT_DETECTION_CONFIDENCE,
                        help=f"YOLO detection confidence threshold (default: {DEFAULT_DETECTION_CONFIDENCE})")
    parser.add_argument("--classify-conf", type=float, default=DEFAULT_CLASSIFIER_CONFIDENCE,
                        help=f"Classifier confidence threshold in %% (default: {DEFAULT_CLASSIFIER_CONFIDENCE})")
    parser.add_argument("--entropy-thresh", type=float, default=DEFAULT_ENTROPY_THRESHOLD,
                        help=f"Entropy threshold (default: {DEFAULT_ENTROPY_THRESHOLD})")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("Warning: Running on CPU. Video processing will be much slower.")
    print(f"Using device: {device}")

    detector = load_detector(args.detector, device)
    classifier, num_classes = load_classifier(args.classifier, device)

    # Auto-select class names based on checkpoint
    dataset_name, class_names = get_class_names(num_classes)
    print(f"[Dataset]    Detected dataset: {dataset_name} ({num_classes} classes)")

    process_video(
        input_path=args.input,
        output_path=args.output,
        detector=detector,
        classifier=classifier,
        device=device,
        detect_thresh=args.detect_conf,
        classify_thresh=args.classify_conf,
        entropy_thresh=args.entropy_thresh,
        class_names=class_names,
    )
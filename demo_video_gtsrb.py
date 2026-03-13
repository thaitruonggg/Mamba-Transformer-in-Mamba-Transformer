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

# Import your custom model
from MaMa_MoEx import MaMa_MoEx_Ti

# ==========================================
# ⚙️ EASY CONFIGURATION: CHANGE THESE PATHS
# ==========================================
DETECTOR_PATH = "best.pt"  # Your trained YOLOv8/11 weights
CLASSIFIER_PATH = "mama_moex_model.pth"  # Your Stage 2 weights
INPUT_VIDEO_PATH = "TestImages/1.mp4"  # Your input video
OUTPUT_VIDEO_PATH = "result_video.mp4"  # Where to save the new video

# Separate Thresholds
DETECTION_CONFIDENCE = 0.2  # YOLO: Gatekeeper for finding boxes
CLASSIFIER_CONFIDENCE = 80.0  # MaMa-MoEx: Expert judge for saving to memory
# ==========================================

# GTSRB class names
gtsrb_class_names = [
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


def load_detector(model_path, device):
    """Loads your custom-trained YOLO detector."""
    print(f"🚀 Loading YOLO Detector from {model_path}...")
    detector = YOLO(model_path)
    detector.to(device)
    return detector


def load_classifier(model_path, device, num_classes=43):
    """Loads your custom MaMa-MoEx classification model."""
    print(f"🧠 Loading MaMa-MoEx Classifier from {model_path}...")
    classifier = MaMa_MoEx_Ti(pretrained=False)
    classifier.head = torch.nn.Linear(in_features=192, out_features=num_classes, bias=True)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    return classifier


def process_video(input_path, output_path, detector, classifier, device, detect_thresh, classify_thresh):
    """Reads a video, runs tracking + classification with memory, and exports."""
    if not os.path.exists(input_path):
        print(f"❌ Error: Video not found at {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {input_path}")
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
    ])

    print(f"\n🎬 Processing Video: {total_frames} frames at {fps} native FPS")

    # --- TRACKER MEMORY & ACADEMIC METRICS ---
    track_memory = {}
    total_latency_ms = 0.0
    valid_frames = 0
    pbar = tqdm(total=total_frames, desc="Exporting Video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Stage 1: Detection with ByteTrack
        results = detector.track(frame, conf=detect_thresh, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

        # Process each tracked box
        if results.boxes.id is not None:
            track_ids = results.boxes.id.int().cpu().tolist()
            boxes = results.boxes.xyxy.cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                if x2 <= x1 or y2 <= y1: continue

                # Stage 2: Classify with MaMa
                cropped_img = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                crop_tensor = classify_transform(cropped_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = classifier(crop_tensor)
                    probabilities = F.softmax(output[0], dim=0)
                    conf_score, class_idx = torch.max(probabilities, dim=0)

                conf_percent = conf_score.item() * 100
                class_name = gtsrb_class_names[class_idx.item()]

                # Update memory if MaMa is highly confident
                if conf_percent >= classify_thresh:
                    track_memory[track_id] = {"name": class_name, "conf": conf_percent}

                # Draw the box IF the tracker remembers what it is
                if track_id in track_memory:
                    mem = track_memory[track_id]
                    label_text = f"ID:{track_id} {mem['name']} ({mem['conf']:.1f}%)"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 170, 0), -1)
                    cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Stop Timer & Sync CUDA ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        # Calculate Frame Metrics
        latency_sec = end_time - start_time
        latency_ms = latency_sec * 1000.0
        current_fps = 1.0 / latency_sec if latency_sec > 0 else 0.0

        total_latency_ms += latency_ms
        valid_frames += 1

        # Embed Metrics on Video
        cv2.rectangle(frame, (10, 10), (280, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {latency_ms:.1f} ms", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out.write(frame)

        # Update Console
        pbar.set_postfix(FPS=f"{current_fps:.1f}", Latency=f"{latency_ms:.1f}ms")
        pbar.update(1)

    # Cleanup and Print Final Report
    pbar.close()
    cap.release()
    out.release()

    if valid_frames > 0:
        avg_latency = total_latency_ms / valid_frames
        avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        print("\n" + "=" * 45)
        print("📊 FINAL METRICS FOR YOUR GRADUATE THESIS:")
        print("=" * 45)
        print(f"Total Frames Analyzed: {valid_frames}")
        print(f"Average Latency:       {avg_latency:.2f} ms")
        print(f"Average FPS:           {avg_fps:.2f} FPS")
        print("=" * 45)

    print(f"\n✅ Done! Video saved successfully to: {output_path}")


# ==========================================
# SCRIPT EXECUTION STARTS HERE
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("⚠️ Warning: Running on CPU. Video processing will be much slower.")

    my_detector = load_detector(DETECTOR_PATH, device)
    my_classifier = load_classifier(CLASSIFIER_PATH, device, num_classes=43)

    process_video(INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, my_detector, my_classifier, device,
                  DETECTION_CONFIDENCE, CLASSIFIER_CONFIDENCE)
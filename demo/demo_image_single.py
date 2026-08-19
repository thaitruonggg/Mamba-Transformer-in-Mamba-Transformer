"""
Behavior:
  - YOLO detects all traffic signs in the image.
  - MaMa-MoEx classifies each detected crop.
  - If the classifier is confident -> label with the class name (green box).
  - If the classifier is NOT confident -> label as "Misc (Unknown Sign)" (orange box).
  - Every detected sign is always shown — nothing is hidden.
"""

import argparse
import os
import math
import warnings
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO

warnings.filterwarnings("ignore")

from MaMa_MoEx import MaMa_MoEx_Ti
#from MaMa import MaMa_Ti

# ==========================================
# DEFAULT CONFIGURATION
# ==========================================
DEFAULT_DETECTOR_PATH = "../yolo11_model/gtsdbbest.pt"
DEFAULT_CLASSIFIER_PATH = "pretrained_models/mama_moex_model.pth"
DEFAULT_IMAGE_PATH = "test_images/1.jpg"

# Thresholds
DEFAULT_DETECTION_CONFIDENCE = 0.2    # YOLO: minimum confidence to consider a detection (0.0–1.0)
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
    # Fallback: generate generic names
    print(f"Warning: Unknown dataset with {num_classes} classes. Using generic class names.")
    return ("Unknown", [f"Class_{i}" for i in range(num_classes)])

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
    Compute the Shannon entropy of a probability distribution.
    High entropy = model is unsure (probabilities spread across many classes).
    Low entropy  = model is focused on one class.
    """
    probs = probabilities.clamp(min=1e-9) # Clamp to avoid log(0)
    entropy = -(probs * probs.log()).sum().item()
    return entropy

def classify_crop(crop_image, classifier, transform, device, class_names):
    """
    Classify a single cropped image.

    Returns:
        class_idx  (int)   : predicted class index
        class_name (str)   : predicted class name
        confidence (float) : softmax confidence as a percentage (0–100)
        entropy    (float) : Shannon entropy of the softmax distribution
    """
    crop_tensor = transform(crop_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = classifier(crop_tensor)
        probabilities = F.softmax(output[0], dim=0)
        conf, class_idx = torch.max(probabilities, dim=0)

    confidence = conf.item() * 100.0
    entropy = compute_entropy(probabilities)
    class_name = class_names[class_idx.item()]

    return class_idx.item(), class_name, confidence, entropy

def run_pipeline(image_path, detector, classifier, device,
                 detect_thresh, classify_thresh, entropy_thresh, class_names):
    """
    Run the full detection → classification pipeline on a single image.

    Every YOLO detection is shown:
      - Classified signs  → green box + class name
      - Uncertain signs   → orange box + "Misc (Unknown Sign)"
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        return

    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"  Detection threshold  : {detect_thresh}")
    print(f"  Classifier threshold : {classify_thresh}%")
    print(f"  Entropy threshold    : {entropy_thresh}")
    print(f"{'='*60}")

    original_image = Image.open(image_path).convert('RGB')

    classify_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Stage 1: Detection (YOLO)
    results = detector(original_image, conf=detect_thresh, verbose=False)[0]
    num_detections = len(results.boxes)
    print(f"\n[Stage 1] YOLO detected {num_detections} sign(s)")

    if num_detections == 0:
        print("No signs detected. Try lowering --detect-conf.")
        # Still show the image even if nothing was detected
        fig = plt.figure(figsize=(14, 9), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(original_image)
        plt.show()
        return

    # Prepare the figure
    fig = plt.figure(figsize=(14, 9), frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(original_image)

    classified_count = 0
    misc_count = 0

    # Stage 2: Classification (MaMa-MoEx) per detection
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        det_conf = box.conf[0].item()

        # Crop the detected region
        cropped_img = original_image.crop((x1, y1, x2, y2))

        # Classify the crop
        class_idx, class_name, cls_conf, entropy = classify_crop(
            cropped_img, classifier, classify_transform, device, class_names
        )

        # Decision: Classified or Misc?
        is_confident = (cls_conf >= classify_thresh) and (entropy <= entropy_thresh)

        if is_confident:
            # Classifier is sure — show the class name
            classified_count += 1
            box_color = '#00FF00'      # green
            label_bg = '#00AA00'       # dark green
            label_text = f"{class_name}\n({cls_conf:.1f}%)"
            status = "CLASSIFIED"
        else:
            # Classifier is NOT sure — mark as Misc
            misc_count += 1
            box_color = '#FF8C00'      # orange
            label_bg = '#CC6600'       # dark orange
            label_text = f"Misc (Unknown Sign)\n({cls_conf:.1f}% | H={entropy:.2f})"
            status = "MISC"

        # Print per-detection info
        reason = ""
        if not is_confident:
            reasons = []
            if cls_conf < classify_thresh:
                reasons.append(f"conf {cls_conf:.1f}% < {classify_thresh}%")
            if entropy > entropy_thresh:
                reasons.append(f"entropy {entropy:.2f} > {entropy_thresh}")
            reason = f"  [{', '.join(reasons)}]"

        print(f"  [{i+1}/{num_detections}] {status}: "
              f"det={det_conf:.2f}, cls={cls_conf:.1f}%, H={entropy:.2f}, "
              f"best_guess=\"{class_name}\"{reason}")

        # Draw the bounding box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor=box_color, facecolor='none'
        )
        ax.add_patch(rect)

        # Draw the label
        ax.text(
            x1, y1 - 10, label_text,
            color='white', fontsize=9, fontweight='bold',
            bbox=dict(facecolor=label_bg, edgecolor='none', pad=3.0, alpha=0.85),
            verticalalignment='bottom'
        )

    # Summary
    print(f"\n[Summary] {num_detections} detected → "
          f"{classified_count} classified, {misc_count} misc/unknown")

    plt.show()

# List available images in the test folder.
def list_test_images(folder):
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    images = []
    if os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            if os.path.splitext(f)[1].lower() in valid_ext:
                images.append(os.path.join(folder, f))
    return images

def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic Sign Detection + Classification Demo (Single Image)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python demo_image_single.py
    python demo_image_single.py --image test_images/2.jpg
    python demo_image_single.py --image test_images/1.jpg --detect-conf 0.3 --classify-conf 90
    python demo_image_single.py --list
        """
    )
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE_PATH,
                        help=f"Path to a single test image (default: {DEFAULT_IMAGE_PATH})")
    parser.add_argument("--detector", type=str, default=DEFAULT_DETECTOR_PATH,
                        help=f"Path to YOLO detector weights (default: {DEFAULT_DETECTOR_PATH})")
    parser.add_argument("--classifier", type=str, default=DEFAULT_CLASSIFIER_PATH,
                        help=f"Path to MaMa-MoEx classifier weights (default: {DEFAULT_CLASSIFIER_PATH})")
    parser.add_argument("--detect-conf", type=float, default=DEFAULT_DETECTION_CONFIDENCE,
                        help=f"YOLO detection confidence threshold (default: {DEFAULT_DETECTION_CONFIDENCE})")
    parser.add_argument("--classify-conf", type=float, default=DEFAULT_CLASSIFIER_CONFIDENCE,
                        help=f"Classifier confidence threshold in %% (default: {DEFAULT_CLASSIFIER_CONFIDENCE})")
    parser.add_argument("--entropy-thresh", type=float, default=DEFAULT_ENTROPY_THRESHOLD,
                        help=f"Entropy threshold for rejecting uncertain classifications (default: {DEFAULT_ENTROPY_THRESHOLD})")
    parser.add_argument("--list", action="store_true",
                        help="List available images in the test_images folder and exit")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # If --list flag, just show available images and exit
    if args.list:
        images = list_test_images("test_images")
        if images:
            print("Available test images:")
            for img in images:
                print(f"  {img}")
        else:
            print("No images found in test_images/")
        exit(0)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models (once)
    detector = load_detector(args.detector, device)
    classifier, num_classes = load_classifier(args.classifier, device)

    # Auto-select class names based on checkpoint
    dataset_name, class_names = get_class_names(num_classes)
    print(f"[Dataset]    Detected dataset: {dataset_name} ({num_classes} classes)")

    # Run pipeline on the single image
    run_pipeline(
        image_path=args.image,
        detector=detector,
        classifier=classifier,
        device=device,
        detect_thresh=args.detect_conf,
        classify_thresh=args.classify_conf,
        entropy_thresh=args.entropy_thresh,
        class_names=class_names,
    )
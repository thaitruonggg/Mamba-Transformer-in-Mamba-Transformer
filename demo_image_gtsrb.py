import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO
import os
import warnings

warnings.filterwarnings("ignore")

from MaMa_MoEx import MaMa_MoEx_Ti
#from MaMa import MaMa_Ti

# ==========================================
# EASY CONFIGURATION: CHANGE THESE PATHS
# ==========================================
DETECTOR_PATH = "YOLO11 model/v11best.pt"
CLASSIFIER_PATH = "mama_moex_model.pth"
IMAGE_PATH = "TestImages/"

# Separate Thresholds
DETECTION_CONFIDENCE = 0.5  # YOLO: How sure it's a sign (0.0 to 1.0)
CLASSIFIER_CONFIDENCE = 85.0  # MaMa: How sure of the specific class (0.0 to 100.0)
# ==========================================

# GTSRB class names (43 classes)
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
    print(f"Loading YOLOv8 Detector from {model_path}...")
    detector = YOLO(model_path)
    detector.to(device)
    return detector


def load_classifier(model_path, device):
    print(f"Loading MaMa-MoEx Classifier from {model_path}...")
    classifier = MaMa_MoEx_Ti(pretrained=False)
    classifier.head = torch.nn.Linear(in_features=192, out_features=43, bias=True)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    return classifier


def run_pipeline(image_path, detector, classifier, device, detect_thresh, classify_thresh):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    original_image = Image.open(image_path).convert('RGB')
    classify_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    # Stage 1: Detection (YOLOv8)
    results = detector(original_image, conf=detect_thresh, verbose=False)[0]

    # Create a figure without a frame
    fig = plt.figure(figsize=(12, 8), frameon=False)
    # Create an axes that fills the entire figure
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(original_image)

    detections_drawn = 0

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        cropped_img = original_image.crop((x1, y1, x2, y2))
        crop_tensor = classify_transform(cropped_img).unsqueeze(0).to(device)

        # Stage 2: Classification (MaMa-MoEx)
        with torch.no_grad():
            output = classifier(crop_tensor)
            probabilities = F.softmax(output[0], dim=0)
            mama_conf, class_idx = torch.max(probabilities, dim=0)

        conf_percent = mama_conf.item() * 100

        if conf_percent >= classify_thresh:
            detections_drawn += 1
            class_name = gtsrb_class_names[class_idx.item()]

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=3, edgecolor='#00FF00', facecolor='none')
            ax.add_patch(rect)

            label_text = f"{class_name}\n({conf_percent:.1f}%)"
            ax.text(x1, y1 - 10, label_text, color='white', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='#00AA00', edgecolor='none', pad=3.0, alpha=0.8),
                    verticalalignment='bottom')

    print(f"Found {len(results.boxes)} boxes, displayed {detections_drawn} after filtering.")
    # No title is set
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    my_detector = load_detector(DETECTOR_PATH, device)
    my_classifier = load_classifier(CLASSIFIER_PATH, device)

    run_pipeline(IMAGE_PATH, my_detector, my_classifier, device,
                 DETECTION_CONFIDENCE, CLASSIFIER_CONFIDENCE)
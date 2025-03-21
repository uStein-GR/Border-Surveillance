import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from torchvision import models

# Load trained models
LIVING_NONLIVING_MODEL_PATH = r"D:/KMUTT/DLRL/Research/RGB Thermal Detection/Code/MoE/model/living_nonliving.pt"
LIVING_MODEL_PATH = r"D:/KMUTT/DLRL/Research/RGB Thermal Detection/Code/MoE/model/efficientnet_person_classifier_with_pre_train.pth"
NONLIVING_MODEL_PATH = r"D:/KMUTT/DLRL/Research/RGB Thermal Detection/Code/MoE/model/efficientnet_vehicle_classifier_with_pre_train.pth"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO model
living_nonliving_model = YOLO(LIVING_NONLIVING_MODEL_PATH).to(device)

# Load EfficientNet models
living_classifier = models.efficientnet_b0(weights=None).to(device)
living_classifier.classifier[1] = torch.nn.Linear(1280, 2).to(device)
living_classifier.load_state_dict(torch.load(LIVING_MODEL_PATH, map_location=device))
living_classifier.eval()

nonliving_classifier = models.efficientnet_b0(weights=None).to(device)
nonliving_classifier.classifier[1] = torch.nn.Linear(1280, 5).to(device)
nonliving_classifier.load_state_dict(torch.load(NONLIVING_MODEL_PATH, map_location=device))
nonliving_classifier.eval()

# Class mappings
LIVING_LABELS = ["Animal", "Person"]
NONLIVING_LABELS = ["Bus", "Car", "Motorbike", "Truck", "Van"]

# Define transformation for EfficientNet
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify_object(model, image, labels):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, class_idx].item()
    return labels[class_idx], confidence

# Video input and output
VIDEO_INPUT_PATH = r"D:\KMUTT\DLRL\Research\RGB Thermal Detection\Code\MoE\testing video\3632187975-preview.mp4"
VIDEO_OUTPUT_PATH = r"D:\KMUTT\DLRL\Research\RGB Thermal Detection\Code\MoE\testing video\video output\output_video2.mp4"

cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    annotated_frame = frame.copy()
    results = living_nonliving_model(source=frame, conf=0.2, imgsz=640)
    final_detections = []
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            label = "Living" if class_id == 0 else "Non-Living"
            cords = [round(x) for x in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = cords
            obj_crop = frame[y1:y2, x1:x2]
            obj_crop = cv2.cvtColor(obj_crop, cv2.COLOR_BGR2RGB)
            
            if label == "Living":
                class_label, class_conf = classify_object(living_classifier, obj_crop, LIVING_LABELS)
            else:
                class_label, class_conf = classify_object(nonliving_classifier, obj_crop, NONLIVING_LABELS)
            
            final_detections.append((cords, class_label, round(class_conf, 2)))
    
    for cords, label, conf in final_detections:
        x1, y1, x2, y2 = cords
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"{label} ({conf:.2f})", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    out.write(annotated_frame)
    cv2.imshow("Video Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

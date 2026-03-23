from ultralytics import YOLO
import cv2
import os

# -------------------------------
# LOAD YOLO MODEL
# -------------------------------
model = YOLO("yolov8s.pt")   # better accuracy
# (if slow → use yolov8n.pt)

# -------------------------------
# FOLDER SETUP
# -------------------------------
base_folder = "training_images"
folders = ["normal_days", "exam_special_days"]

print("\nRunning YOLO Crowd Detection (Optimized)...\n")

for folder_name in folders:
    folder_path = os.path.join(base_folder, folder_name)

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            print(f"Error loading image: {img_name}")
            continue

        # -------------------------------
        # YOLO DETECTION (IMPROVED 🔥)
        # -------------------------------
        results = model(image, conf=0.25, imgsz=640)

        count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                if cls == 0:  # person class
                    count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(image, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

        print(f"{img_name} | {folder_name} | People Count: {count}")

        # Display
        cv2.putText(image, f"Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("YOLO Crowd Detection", image)
        cv2.waitKey(0)

cv2.destroyAllWindows()
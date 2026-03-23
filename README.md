# Crowd Detection using Deep Learning

## 📌 Project Overview

This project presents a Deep Learning based crowd detection system that identifies and counts people in images using the YOLOv8 object detection model. The system is designed to analyze crowd density under different scenarios such as normal days and exam days.

## 🎯 Objective

The main objective of this project is to:

* Detect people in images using a deep learning model
* Count the number of individuals present
* Differentiate crowd levels based on contextual factors like normal days and exam days
* Build a foundation for future crowd prediction using machine learning

## 🧠 Technologies Used

* Python
* YOLOv8 (Ultralytics)
* OpenCV
* NumPy

## ⚙️ How It Works

1. Input images are provided from two categories:

   * Normal Days
   * Exam Special Days

2. The YOLOv8 model processes each image and:

   * Detects people
   * Draws bounding boxes
   * Counts the number of individuals

3. The output is displayed with:

   * Detected people highlighted
   * Total crowd count printed

## 📂 Project Structure

```
project/
│
├── training_images/
│     ├── normal_days/
│     └── exam_special_days/
│
├── main.py
├── requirements.txt
```

## ▶️ How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the program:

```
python main.py
```

## 🚀 Future Scope

* Convert image-based detection into real-time video processing
* Build dataset from detected counts
* Apply machine learning models (Decision Tree / LSTM) for crowd prediction
* Improve accuracy using advanced YOLO models

## 🧠 Conclusion

This project demonstrates how deep learning can be effectively used for real-world applications like crowd monitoring and analysis. It provides a scalable base for intelligent crowd prediction systems.

---

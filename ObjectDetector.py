import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget, QListWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO("yolo11n.pt")

# Function to perform object detection on an image using YOLOv8
# def detect_objects(image_path):
#     img = cv2.imread(image_path)  # Read the image
#     results = model(img)  # Perform detection

#     # YOLOv8 results are accessed differently
#     detected_objects = []
    
#     for result in results:
#         boxes = result.boxes  # Get the detected boxes
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Extract the bounding box coordinates
#             confidence = box.conf.item()  # Confidence score
#             label = box.cls.item()  # Class label
#             label_name = model.names[int(label)]  # Get label name

#             # Append the detection to the list
#             detected_objects.append((label_name, confidence, (x1, y1, x2, y2)))

#             # Draw the bounding boxes and labels on the image
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(img, f'{label_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     return img, detected_objects

from collections import defaultdict
import random

# Generate a unique color for each label
def get_color_for_label(label_name):
    random.seed(hash(label_name))  # Ensure same color for the same label
    return tuple(random.randint(0, 255) for _ in range(3))  # Generate an (R, G, B) tuple

# Function to perform object detection on an image using YOLOv8
def detect_objects(image_path):
    img = cv2.imread(image_path)  # Read the image
    results = model(img)  # Perform detection

    # Predefined colors for objects (you can adjust the colors or generate them dynamically)
    class_colors = {}
    
    # Dictionary to store the count of each detected object
    object_counts = {}

    detected_objects = []

    for result in results:
        boxes = result.boxes  # Get the detected boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Extract bounding box coordinates
            confidence = box.conf.item()  # Confidence score
            label = box.cls.item()  # Class label
            label_name = model.names[int(label)]  # Get label name

            # Assign a random color if the label is new
            if label_name not in class_colors:
                class_colors[label_name] = [random.randint(0, 255) for _ in range(3)]

            # Update the object counts
            if label_name in object_counts:
                object_counts[label_name] += 1
            else:
                object_counts[label_name] = 1

            # Append the detection to the list
            detected_objects.append((label_name, confidence, (x1, y1, x2, y2)))

            # Get the color for this label
            color = class_colors[label_name]

            # Draw bounding boxes and labels on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f'{label_name} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return img, detected_objects, object_counts





class ObjectDetectorApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Object Detector')
        self.setGeometry(100, 100, 800, 600)

        # Layouts
        layout = QVBoxLayout()

        # QLabel to display the image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # QPushButton to load an image
        self.load_image_btn = QPushButton('Load Image')
        self.load_image_btn.clicked.connect(self.load_image)
        layout.addWidget(self.load_image_btn)

        # QListWidget to display detected objects
        self.object_list = QListWidget()
        layout.addWidget(self.object_list)

        self.setLayout(layout)
        
        
    def load_image(self):
        # Open file dialog to select an image
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Open Image', '', 'Image Files (*.png *.jpg *.jpeg)')

        if file_path:
            # Perform object detection
            detected_image, objects, object_counts = detect_objects(file_path)

            # Update the image display
            q_img = self.convert_cv_qt(detected_image)
            self.image_label.setPixmap(q_img)

            # Update the list of detected objects with their counts
            self.object_list.clear()
            for obj_name, count in object_counts.items():
                self.object_list.addItem(f'{obj_name}: {count}')

    def convert_cv_qt(self, cv_img):
            """ Convert from OpenCV image to QPixmap for display in PyQt """
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convert_to_qt_format)
            return pixmap.scaled(640, 480, Qt.KeepAspectRatio)

# Main function to run the PyQt app
def main():
    app = QApplication([])
    window = ObjectDetectorApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()

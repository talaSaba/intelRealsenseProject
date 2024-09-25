import cv2
import numpy as np
import torch


class YoloDetector():
    #initializing the YOLO detector and loading the model and sets the device to GPU/CPU
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    #loads a spesified YOLOv5 model, if no model is provided load the pretrained YOLOv5 model
    def load_model(self, model_name):
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    #resize the frame and runs the model on it, return the detected labels and coordinates
    def score_frame(self, frame):
        self.model.to(self.device)  
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    #converts a class index to its corresponding label
    def class_to_label(self, x):
        return self.classes[int(x)]

    #plots bounding boxes on the frame for detected people and returns the detections and frame
    def plot_boxes(self, results, frame, height, width, confidence):
        labels, cord = results
        detections = []
        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]
            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                label = self.class_to_label(labels[i])
                #print(f"Detected: {label} with confidence: {row[4]}")
                if label == 'person':
                    tlwh = np.asarray([x1, y1, int(x2 - x1), int(y2 - y1)], dtype=np.float32)
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], row[4].item(), 'person'))
        return frame, detections

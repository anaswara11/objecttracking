import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class ObjectDetection:
    def __init__(self, weights_path="dnn_model/yolov4.weights", cfg_path="dnn_model/yolov4.cfg"):
        print("Loading Object Detection")
        print("Running opencv dnn with YOLOv4")
        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 608

        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path="dnn_model/classes.txt"):
        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(80, 3))
        return self.classes

    def detect(self, frame):
        return self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)

def main():
    od = ObjectDetection()
    cap = cv2.VideoCapture("classvideo3.mov")

    # Get screen resolution
    screen_width, screen_height = 1366, 768  # Default values, replace with your screen resolution

    # Initialize count
    count = 0

    # Dictionary to store the coordinates of the red dots for each box
    previous_dots = {}

    while True:
        ret, frame = cap.read()
        count += 1
        if not ret:
            break

        # Detect objects on frame
        (class_ids, scores, boxes) = od.detect(frame)

        # Draw red dots for each box
        for box in boxes:
            (x, y, w, h) = box
            center_x = x + w // 2
            center_y = y + h
            center = (center_x, center_y)

            # Draw red dot at the center of the lower edge of the box
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Store the coordinates of the red dot for the current box
            if count not in previous_dots:
                previous_dots[count] = {}
            previous_dots[count][tuple(box)] = center  # Convert numpy array to tuple

        # Draw red dots from previous frames for each box
        for frame_num, dots in previous_dots.items():
            if frame_num < count:
                for box, center in dots.items():
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # Calculate scaling factor
        scale_factor = min(screen_width / 3840, screen_height / 2160) * 0.8

        # Resize frame
        frame_resized = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

        cv2.imshow("Frame", frame_resized)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Plot the mapped points on a 2D graph
    plot_2d_graph(previous_dots)

def plot_2d_graph(dots):
    x_coords = []
    y_coords = []

    for frame_num, frame_dots in dots.items():
        for box, center in frame_dots.items():
            x_coords.append(center[0])
            y_coords.append(center[1])

    plt.scatter(x_coords, y_coords, color='red')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D Graph of Detected Points')
    plt.show()

if __name__ == "__main__":
    main()

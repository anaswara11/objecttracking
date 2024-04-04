import cv2
import numpy as np

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

def generate_heatmap(dots, screen_width, screen_height):
    heatmap = np.zeros((screen_height, screen_width), dtype=np.uint8)

    for frame_num, frame_dots in dots.items():
        for box, center in frame_dots.items():
            x, y = center
            if y < screen_height and x < screen_width:  # Ensure the point is within the screen boundaries
                heatmap[int(y), int(x)] += 1  # Increment the value at the pixel corresponding to the detected point

    # Normalize the heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    # Apply colormap for visualization
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

    return heatmap

def main():
    od = ObjectDetection()
    cap = cv2.VideoCapture("classvideo3.mov")

    # Get screen resolution
    screen_width, screen_height = 1366, 768  # Default values, replace with your screen resolution

    # Dictionary to store the coordinates of the red dots for each box
    previous_dots = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects on frame
        (class_ids, scores, boxes) = od.detect(frame)

        # Store the coordinates of the red dot for each box
        for box in boxes:
            (x, y, w, h) = box
            center_x = x + w // 2
            center_y = y + h
            center = (center_x, center_y)

            if int(center_y) < screen_height and int(center_x) < screen_width:  # Ensure the point is within the screen boundaries
                if count not in previous_dots:
                    previous_dots[count] = {}
                previous_dots[count][tuple(box)] = center  # Convert numpy array to tuple

    cap.release()

    # Generate heatmap
    heatmap = generate_heatmap(previous_dots, screen_width, screen_height)

    cv2.imshow("Heatmap", heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

"""
detector.py

Provides a Detector class for finding people in each video frame.
Supports either OpenCV’s DNN module or Ultralytics YOLOv8 under the hood.
"""

import cv2  # OpenCV for image processing and DNN inference

# Try to import the YOLO class from Ultralytics if available
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # If not installed, we’ll disable YOLO mode

class Detector:
    def __init__(self, method='yolo', model_path=None, conf_threshold=0.5):
        """
        Set up the person detector.

        :param method: 'yolo' to use YOLOv8, or 'opencv' to use OpenCV DNN
        :param model_path:
            - For 'yolo': path to a .pt weights file (default: 'yolov8n.pt')
            - For 'opencv': dict with keys 'prototxt' (model definition) and 'model' (weights)
        :param conf_threshold: minimum confidence (0.0–1.0) to accept a detection
        """
        self.method = method
        self.conf_threshold = conf_threshold

        if method == 'opencv':
            # Load Caffe-style model for CPU-based detection
            proto = model_path.get('prototxt')
            weights = model_path.get('model')
            self.net = cv2.dnn.readNetFromCaffe(proto, weights)

        elif method == 'yolo':
            # Ensure the ultralytics package is installed
            if YOLO is None:
                raise ImportError(
                    "YOLOv8 support requires 'pip install ultralytics'"
                )
            # Load the YOLOv8 model (will download weights if needed)
            weights = model_path or 'yolov8n.pt'
            self.model = YOLO(weights)

        else:
            # If the user passed something else, we can’t proceed
            raise ValueError(
                "Unsupported detection method: choose 'yolo' or 'opencv'."
            )

    def detect(self, frame):
        """
        Run person detection on a single frame.

        :param frame: the image array in BGR color format
        :return: list of tuples (x1, y1, x2, y2, confidence) for each person
        """
        results = []

        if self.method == 'opencv':
            # Prepare the frame for OpenCV DNN: resize & normalize
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                frame, 0.007843, (300, 300), 127.5
            )
            # Run the model
            self.net.setInput(blob)
            detections = self.net.forward()

            # Loop over all detections
            for i in range(detections.shape[2]):
                score = float(detections[0, 0, i, 2])
                cls_id = int(detections[0, 0, i, 1])
                # Class ID 15 is 'person' in COCO for this model
                if cls_id == 15 and score >= self.conf_threshold:
                    # Convert relative coords to absolute pixel values
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    x1, y1, x2, y2 = box.astype(int)
                    results.append((x1, y1, x2, y2, score))

        else:  # YOLOv8 path
            # Perform inference; results[0] holds detections for this frame
            detections = self.model(frame)[0]
            for b in detections.boxes:
                cls_id = int(b.cls)       # detected class index
                conf   = float(b.conf)    # confidence score
                # Class 0 is 'person' in COCO for YOLO
                if cls_id == 0 and conf >= self.conf_threshold:
                    # b.xyxy is a tensor [[x1,y1,x2,y2]]; convert to ints
                    x1, y1, x2, y2 = (
                        b.xyxy[0].cpu().numpy().astype(int)
                    )
                    results.append((x1, y1, x2, y2, conf))

        return results  # return all person boxes and confidences


def main():
    """
    Quick command-line test: open your webcam and draw detections live.
    Press 'q' to quit.
    """
    import argparse
    from capture import test_webcam

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['yolo', 'opencv'], default='yolo',
                        help='Which detection backend to use')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Minimum confidence to display')
    parser.add_argument('--mirror', action='store_true',
                        help='Flip the image horizontally')
    parser.add_argument('--source', type=int, default=0,
                        help='Camera index to use (e.g. 0 or 1)')
    args = parser.parse_args()

    # Create the detector with chosen options
    det = Detector(
        method=args.method,
        model_path=None if args.method == 'yolo'
                   else {'prototxt':'path/to/prototxt',
                         'model':'path/to/caffemodel'},
        conf_threshold=args.conf
    )

    # Open the webcam directly for testing
    cap = cv2.VideoCapture(args.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # stop if we lost the camera feed

        if args.mirror:
            frame = cv2.flip(frame, 1)  # mirror view if requested

        # Run detection on the frame
        detections = det.detect(frame)
        for x1, y1, x2, y2, conf in detections:
            # Draw a green rectangle around each person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Label it with the confidence score
            cv2.putText(
                frame, f"{conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        # Show the annotated frame in a window
        cv2.imshow('Detections', frame)
        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up when done
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

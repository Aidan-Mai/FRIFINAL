#!/usr/bin/env python3
"""
app.py

Main application tying together video capture, person detection, tracking, and counting.
Each line and key command is commented to explain its purpose in simple terms.
"""

# --- IMPORTS ---
import argparse             # for parsing command-line arguments
import cv2                  # OpenCV library for video I/O and drawing
import numpy as np         # NumPy for numerical operations (not heavily used here)
import time                 # for timing the flashing alert

from capture import VideoStream           # custom module to handle camera input
from detector import Detector             # custom module for person detection
from tracker import CentroidTracker       # custom module for tracking detected people
from counter import PeopleCounter         # custom module for counting entries/exits

# --- FLASHING ALERT CONFIGURATION ---
BLINK_ON_DURATION  = 1.5  # how many seconds the red alert stays visible
BLINK_OFF_DURATION = 1.5  # how many seconds the red alert is hidden
CYCLE_DURATION     = BLINK_ON_DURATION + BLINK_OFF_DURATION  # total flash cycle

def parse_args():
    """
    Define and parse command-line arguments, returning the populated namespace.
    """
    parser = argparse.ArgumentParser(description="People counting application")
    parser.add_argument(
        '--source', type=str, default='0',
        help='Camera index (int) or path to video file'
    )
    parser.add_argument(
        '--width', type=int, default=640,
        help='Width of the captured frame'
    )
    parser.add_argument(
        '--height', type=int, default=480,
        help='Height of the captured frame'
    )
    parser.add_argument(
        '--mirror', action='store_true',
        help='Flip the frame horizontally (mirror view)'
    )
    parser.add_argument(
        '--method', choices=['yolo', 'opencv'], default='yolo',
        help='Detection backend: "yolo" for Ultralytics YOLO, "opencv" for OpenCV DNN'
    )
    parser.add_argument(
        '--conf', type=float, default=0.5,
        help='Minimum confidence for detections (0.0 to 1.0)'
    )
    parser.add_argument(
        '--threshold', type=int, default=5,
        help='Number of people allowed before triggering alert'
    )
    parser.add_argument(
        '--line', type=int, nargs=4, default=[100, 240, 540, 240],
        help='Coordinates for counting line: x1 y1 x2 y2'
    )
    return parser.parse_args()


def preprocess_boxes(boxes, frame_shape, iou_thresh=0.3, min_area=2000):
    """
    Clamp boxes to frame boundaries, drop tiny boxes, and remove duplicates.

    :param boxes: list of raw bounding boxes [(x1,y1,x2,y2), ...]
    :param frame_shape: shape of the current frame (height, width, channels)
    :param iou_thresh: threshold above which two boxes are considered duplicates
    :param min_area: minimum box area (in pixels) to keep
    :return: filtered list of cleaned bounding boxes
    """
    h, w = frame_shape[:2]   # extract frame height and width
    processed = []

    # 1) Clamp and area filter
    for (x1, y1, x2, y2) in boxes:
        # ensure coordinates lie within [0, width-1] and [0, height-1]
        x1c = max(0, min(x1, w - 1))
        y1c = max(0, min(y1, h - 1))
        x2c = max(0, min(x2, w - 1))
        y2c = max(0, min(y2, h - 1))
        area = (x2c - x1c) * (y2c - y1c)  # compute box area
        if area < min_area:
            continue  # skip boxes smaller than min_area
        processed.append((x1c, y1c, x2c, y2c))

    # 2) Deduplicate by Intersection over Union (IoU)
    def iou(a, b):
        """Compute IoU between two boxes a and b."""
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        interW = max(0, xB - xA); interH = max(0, yB - yA)
        interArea = interW * interH
        unionArea = ( (a[2]-a[0])*(a[3]-a[1]) +
                      (b[2]-b[0])*(b[3]-b[1]) - interArea )
        return interArea / unionArea if unionArea > 0 else 0

    keep = []
    for box in processed:
        # keep box if it doesn't overlap too much with any already kept box
        if not any(iou(box, kept) > iou_thresh for kept in keep):
            keep.append(box)

    return keep


def main():
    """
    Main entry point: sets up modules, reads frames, and processes each frame.
    """
    # Parse command-line arguments
    args = parse_args()

    # Determine camera source: convert to int if it's a digit, else use as file path
    src = int(args.source) if args.source.isdigit() else args.source

    # Initialize the detection module with chosen method and confidence threshold
    det = Detector(
        method=args.method,
        model_path=None if args.method == 'yolo' else {
            'prototxt': 'path/to/prototxt',   # replace with actual paths
            'model': 'path/to/caffemodel'
        },
        conf_threshold=args.conf
    )

    # Initialize the centroid tracker with a max disappearance tolerance
    ct = CentroidTracker(maxDisappeared=40)

    # Define the counting line endpoints from parsed arguments
    line = ((args.line[1], args.line[0]), (args.line[3], args.line[2]))

    # Initialize the PeopleCounter with the line and occupancy threshold
    counter = PeopleCounter(line=line, threshold=args.threshold)

    # Open the video stream (webcam or file)
    vs = VideoStream(source=src, width=args.width, height=args.height)
    vs.open()

    # --- FRAME PROCESSING LOOP ---
    while True:
        # 1. Grab a frame
        ret, frame = vs.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        # 2. Optionally mirror the frame for a 'selfie' view
        if args.mirror:
            frame = cv2.flip(frame, 1)

        # 3. Draw the counting line in red
        cv2.line(frame, line[0], line[1], (0, 0, 255), 2)

        # 4. Detect people: returns boxes + confidence
        detections = det.detect(frame)
        raw_boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in detections]

        # 5. Clean up the boxes (clamp, filter, dedupe)
        boxes = preprocess_boxes(raw_boxes, frame.shape)

        # 6. Track detections and assign IDs
        objects = ct.update(boxes)

        # 7. Count entries/exits and get current count + alert flag
        count, alert = counter.update(objects)

        # 8. Overlay the current count in green text
        cv2.putText(frame, f"Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 9. If threshold exceeded, flash a red alert on/off
        if alert:
            cycle = time.time() % CYCLE_DURATION
            if cycle < BLINK_ON_DURATION:
                h, w = frame.shape[:2]
                # draw a thick red border
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 20)
                # draw centered alert message
                cv2.putText(frame, "!!! ROOM FULL !!!",
                            (w // 2 - 200, h // 2),
                            cv2.FONT_HERSHEY_TRIPLEX, 2.0,
                            (255, 255, 255), 4, cv2.LINE_AA)

        # 10. Draw tracked objects: ID label, centroid dot, and bounding box
        for objectID, (centroid, bbox) in objects.items():
            x, y = centroid
            # label with the object ID
            cv2.putText(frame, f"ID {objectID}", (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # small circle at the centroid
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
            # bounding box around the person
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 11. Display the frame in a window named "People Counter"
        cv2.imshow("People Counter", frame)

        # 12. Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- CLEANUP ---
    vs.release()             # close the video stream
    cv2.destroyAllWindows()  # close all OpenCV windows


if __name__ == '__main__':
    main()  # run the main function when script is executed

"""
counter.py

Implements entry/exit counting logic for tracked people crossing a defined line.
Tracks people as they enter or leave and raises an alert when occupancy ≥ threshold.
"""

import numpy as np  # for vector math and numeric operations

class PeopleCounter:
    def __init__(self, line, threshold=1):
        """
        Initialize the counter.

        :param line: tuple of two points defining the counting line ((x1, y1), (x2, y2))
        :param threshold: number of people that triggers the alert
        """
        # store line endpoints as NumPy arrays for easy math
        self.p1 = np.array(line[0], dtype=float)
        self.p2 = np.array(line[1], dtype=float)

        # occupancy threshold and current count
        self.threshold = threshold
        self.count = 0

        # flag to know if we've already alerted
        self.alerted = False

        # keep track of which side of the line each object was on last frame
        # key: objectID, value: side (>0 or <0)
        self._prev_sides = {}

    def _get_side(self, centroid):
        """
        Determine which side of the line a point lies on.

        We use the sign of the 2D cross product:
          cross > 0 => one side
          cross < 0 => other side
          cross == 0 => exactly on the line

        :param centroid: (x, y) tuple for the object's center
        :return: -1, 0, or +1 indicating side
        """
        c = np.array(centroid, dtype=float)    # convert to NumPy array
        v_line = self.p2 - self.p1             # direction vector of the line
        v_point = c - self.p1                  # vector from line start to point

        # 2D cross product (scalar) = x1*y2 - y1*x2
        cross = v_line[0] * v_point[1] - v_line[1] * v_point[0]
        return np.sign(cross)  # -1, 0, or +1

    def update(self, objects):
        """
        Update the occupancy count based on tracked objects.

        :param objects: dict mapping objectID -> (centroid, bbox)
        :return: (current_count, alert_flag)
        """
        # IDs currently in view
        current_ids = set(objects.keys())
        # IDs seen in previous update
        prev_ids = set(self._prev_sides.keys())

        # --- Handle new objects appearing this frame ---
        for objectID in current_ids - prev_ids:
            centroid, _ = objects[objectID]
            side = self._get_side(centroid)       # which side it started on
            if side > 0:
                # first appears on the “inside” side → count as entry
                self.count += 1
            # record its side for next frame
            self._prev_sides[objectID] = side

        # --- Handle objects that persist across frames ---
        for objectID in current_ids & prev_ids:
            centroid, _ = objects[objectID]
            side = self._get_side(centroid)       # new side
            prev = self._prev_sides[objectID]     # side last frame
            if side != prev and side != 0:
                # crossed the line this frame
                if side > prev:
                    # moved onto “inside” → entry
                    self.count += 1
                else:
                    # moved onto “outside” → exit
                    self.count = max(self.count - 1, 0)
            # update stored side
            self._prev_sides[objectID] = side

        # --- Handle objects that disappeared this frame ---
        lost_ids = prev_ids - current_ids
        for objectID in lost_ids:
            # simply remove from tracking history; do NOT decrement the count
            del self._prev_sides[objectID]

        # --- Determine whether to raise or clear the alert ---
        alert = False
        if self.count >= self.threshold and not self.alerted:
            # threshold reached
            alert = True
            self.alerted = True
        elif self.count < self.threshold and self.alerted:
            # count fell back below threshold
            self.alerted = False

        return self.count, alert


if __name__ == "__main__":
    # Quick demo to test the counter in isolation
    import cv2
    from detector import Detector
    from tracker import CentroidTracker

    # define a vertical line at x=320 from y=0 to y=480
    line = ((320, 0), (320, 480))
    threshold = 2                  # alert when 2 or more people inside
    counter = PeopleCounter(line, threshold)

    # set up detection and tracking
    det = Detector(method='yolo', conf_threshold=0.5)
    ct = CentroidTracker(maxDisappeared=40)

    # open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # draw the counting line in red
        cv2.line(frame, line[0], line[1], (0, 0, 255), 2)

        # detect people and extract bounding boxes
        boxes = [(x1, y1, x2, y2)
                 for x1, y1, x2, y2, _ in det.detect(frame)]
        # track them and get centroids
        objects = ct.update(boxes)
        # update count and get alert flag
        count, alert = counter.update(objects)

        # show count in green at top-left
        cv2.putText(frame, f"Count: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # if alert is active, show text in red below count
        if alert:
            cv2.putText(frame, "Threshold reached!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # draw each tracked object's ID, centroid, and box
        for oid, (centroid, bbox) in objects.items():
            c = tuple(centroid.astype(int))
            cv2.circle(frame, c, 4, (255, 0, 0), -1)
            cv2.putText(frame, f"ID {oid}", (c[0] - 10, c[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # display the demo window
        cv2.imshow("Counter Demo", frame)
        # press 'q' to exit the demo
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

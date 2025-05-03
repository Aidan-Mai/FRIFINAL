"""
tracker.py

Tracks detected objects across frames using a simple centroid-based tracker.
"""

import numpy as np

class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        """
        :param maxDisappeared: number of consecutive frames an object can be missing
        before deregistering
        """
        # next object ID to assign
        self.nextObjectID = 0
        # objectID -> centroid (x, y)
        self.objects = {}
        # objectID -> number of consecutive frames disappeared
        self.disappeared = {}
        # objectID -> bounding box
        self.bboxes = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid, bbox=None):
        """Register a new object with a centroid and optional bounding box."""
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        if bbox is not None:
            self.bboxes[self.nextObjectID] = bbox
        self.nextObjectID += 1

    def deregister(self, objectID):
        """Remove an object from tracking."""
        del self.objects[objectID]
        del self.disappeared[objectID]
        if objectID in self.bboxes:
            del self.bboxes[objectID]

    def update(self, rects):
        """
        Update tracked objects based on new bounding boxes.

        :param rects: list of (x1, y1, x2, y2) tuples
        :return: dict mapping objectID -> (centroid, bbox)
        """
        # if no detections, mark existing objects as disappeared
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return {objID: (self.objects[objID], self.bboxes.get(objID))
                    for objID in self.objects}

        # compute input centroids
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if no existing objects, register all
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
        else:
            # grab existing object IDs and centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = np.array(list(self.objects.values()))

            # compute distance matrix between object centroids and input centroids
            D = np.linalg.norm(objectCentroids[:, np.newaxis] - inputCentroids, axis=2)
            # find smallest value in each row, then sort row indices
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()
            # match object IDs to input centroids
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bboxes[objectID] = rects[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            # check unmatched existing objects
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            # check unmatched new detections
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # if objects >= detections, some disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                # new objects appeared
                for col in unusedCols:
                    self.register(inputCentroids[col], rects[col])

        # return updated objects
        return {objID: (self.objects[objID], self.bboxes.get(objID))
                for objID in self.objects}


if __name__ == "__main__":
    # Quick interactive test
    import cv2
    from capture import test_webcam
    from detector import Detector

    ct = CentroidTracker(maxDisappeared=40)
    det = Detector(method='yolo', conf_threshold=0.5)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = [(x1, y1, x2, y2) for (x1, y1, x2, y2, _) in det.detect(frame)]
        objects = ct.update(boxes)

        for objectID, (centroid, bbox) in objects.items():
            text = f"ID {objectID}"
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            (x1, y1, x2, y2) = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imshow("Tracked", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

"""
capture.py

Handles video capture from a camera device or video file.
Provides VideoStream class and test_webcam for quick tests.
"""

import cv2  # OpenCV library for video operations


class VideoStream:
    """
    Wraps OpenCV's VideoCapture for easy frame grabbing.
    """

    def __init__(self, source=1, width=640, height=480):
        """
        Store the video source and desired resolution.

        :param source: camera index (int) or path to a video file (str)
        :param width: width of each frame (pixels)
        :param height: height of each frame (pixels)
        """
        self.source = source      # where to read frames from
        self.width = width        # desired frame width
        self.height = height      # desired frame height
        self.cap = None           # will hold the VideoCapture object

    def open(self):
        """
        Open the video source and set the frame size.
        Raises an error if the source cannot be opened.
        """
        # Create the capture object
        self.cap = cv2.VideoCapture(self.source)
        # Check if it opened successfully
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source}")
        # Set the desired frame width and height
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        """
        Grab a single frame from the open video source.

        :return: tuple (ret, frame)
                 ret   : True if frame was read correctly
                 frame : the image array if ret is True
        """
        if self.cap is None:
            # Protect against calling read() before open()
            raise RuntimeError("Video source is not opened. Call open() first.")
        return self.cap.read()  # returns (success_flag, image)

    def release(self):
        """
        Close the video source and free resources.
        """
        if self.cap is not None:
            self.cap.release()  # release the camera or file
            self.cap = None     # reset to indicate it's closed


def test_webcam(device_index=0, width=640, height=480, mirror=False):
    """
    Quickly test a webcam feed in a window.

    :param device_index: which camera to open (usually 0 or 1)
    :param width: width of the test window (pixels)
    :param height: height of the test window (pixels)
    :param mirror: if True, flip the frame left-to-right
    """
    # Open the webcam directly
    cap = cv2.VideoCapture(device_index)
    # Set the capture size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Check for successful opening
    if not cap.isOpened():
        print(f"Error: cannot open webcam index {device_index}")
        return

    # Loop until the user quits
    while True:
        ret, frame = cap.read()  # grab a frame
        if not ret:
            print("Failed to grab frame")
            break  # end loop on failure

        # Optionally mirror the image (good for selfie view)
        if mirror:
            frame = cv2.flip(frame, 1)

        # Overlay instructions on the frame
        cv2.putText(frame, "Press 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the frame in a window
        cv2.imshow("Webcam Test", frame)

        # Wait for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up: close camera and window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # If run as a script, do a default webcam test on camera 0
    test_webcam(device_index=0, width=640, height=480, mirror=False)

from traitlets import Any, Integer, Bool
from traitlets.config.configurable import SingletonConfigurable, Config
import atexit
import cv2
import threading
import numpy as np
import traceback



class Camera(SingletonConfigurable):

    value           = Any()
    _running        = Bool()
    width           = Integer(default_value=224).tag(config=True)
    height          = Integer(default_value=224).tag(config=True)
    fps             = Integer(default_value=21).tag(config=True)
    capture_width   = Integer(default_value=816).tag(config=True)
    capture_height  = Integer(default_value=616).tag(config=True)

    def __init__(self, config=None, **kwargs):
        if config:
            self.update_config(config)
        super().__init__(**kwargs)
        self.value = np.empty((self.height, self.width, 3), dtype=np.uint8)

        try:
            self.cap = cv2.VideoCapture(self._gst_str(), cv2.CAP_GSTREAMER)
            re, image = self.cap.read()
            if not re:
                traceback.print_exc()
                self.stop()
                raise RuntimeError('Could not read image from camera.')
            self.value = image
            self.start()
        except:
            self.stop()
            raise RuntimeError(
                'Could not initialize camera.  Please see error trace.')

        atexit.register(self.stop)

    def _capture_frames(self):
        while self._running:
            re, image = self.cap.read()
            if re:
                self.value = image
            else:
                break

    def _gst_str(self):
        return ('nvarguscamerasrc ! '  # source for capturing video from a camera
        'video/x-raw(memory:NVMM), width=(int){}, height=(int){}, format=(string)NV12, framerate=(fraction){}/1 ! '
        'nvvidconv ! '  # video converter
        'video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! '  # Conversion settings
        'videoconvert ! '  # Converts video from one colorspace to another
        'queue max-size-buffers=1 leaky=downstream ! '
        'appsink '  # sink element that receives video frames
        'drop=true max-buffers=1'
       ).format(self.capture_width, self.capture_height, self.fps, self.width, self.height)

    def start(self):
        self._running = True
        if not self.cap.isOpened():
            self.cap.open(self._gst_str(), cv2.CAP_GSTREAMER)
        if not hasattr(self, 'thread') or not self.thread.isAlive():
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.start()

    def stop(self):
        self._running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'thread'):
            self.thread.join()

    def restart(self):
        self.stop()
        self.start()


def camera_config(width=None, height=None, fps=None):
    """
    Create and return a configuration object for a camera with resolution set to width and height.
    Example:
        >>> camera = Camera(config=capture(width=816, height=616, fps=21))
        >>> camera.capture_width, camera.capture_height, camera.fps
        816, 616, 21
    """
    c = Config()
    if width is not None:
        c.Camera.capture_width = width
    if height is not None:
        c.Camera.capture_height = height
    if fps is not None:
        c.Camera.fps = fps
    return c


if __name__ == "__main__":
    # Running this file directly will save a sample image from the camera:
    camera = Camera()
    cv2.imwrite('output_image.png', camera.value)
    camera.stop()
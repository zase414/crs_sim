"""
A window showing the simulated camera feed: every image the script grabs.

    window = CameraWindow()
    robot.camera.on_grab.append(window.show)

The window is an OpenCV window in a separate process (this file run as a
script) fed through a pipe, so it needs neither the main thread nor
``cv2.waitKey`` calls in the script, and it does not interfere with the
script's own OpenCV windows. ``show`` never blocks: frames the
window has not taken yet are replaced by newer ones. Closing the window just
stops the feed.

The images are shown in full resolution, scaled to the window: it opens
``width`` pixels wide, resize it to see more, or zoom in with Ctrl + mouse
wheel (or the toolbar) to see single pixels.
"""
from __future__ import annotations

import atexit
import os
import struct
import subprocess
import sys
import threading
import time

import numpy as np

# frame header: height, width, frame number, exposure time [s]
_HEADER = struct.Struct("<IIId")


class CameraWindow:
    """
    :param title: window title
    :param width: initial window width [px], the height follows the image
        aspect ratio
    """
    def __init__(self, title: str = "crs_sim camera", width: int = 800) -> None:
        self._process = subprocess.Popen(
            [sys.executable, __file__, title, str(width)],
            stdin=subprocess.PIPE)
        self._frame: bytes | None = None
        self._count = 0
        self._t0 = time.monotonic()
        self._ready = threading.Condition()
        self._closed = False
        threading.Thread(target=self._send, daemon=True,
                         name="crs_sim camera window").start()
        atexit.register(self.close)

    def show(self, image: np.ndarray, t_exp: float | None = None) -> None:
        """Show an HxWx3 uint8 BGR image, exposed at t_exp (time.monotonic())."""
        if self._closed:
            return
        image = np.ascontiguousarray(image[:, :, :3], dtype=np.uint8)
        self._count += 1
        t = float("nan") if t_exp is None else t_exp - self._t0
        frame = _HEADER.pack(*image.shape[:2], self._count, t) + image.tobytes()
        with self._ready:
            self._frame = frame
            self._ready.notify()

    def close(self) -> None:
        """Close the window."""
        with self._ready:
            self._closed = True
            self._ready.notify()
        try:
            self._process.stdin.close()
        except OSError:
            pass

    def _send(self) -> None:
        while True:
            with self._ready:
                while self._frame is None and not self._closed:
                    self._ready.wait()
                if self._closed:
                    return
                frame, self._frame = self._frame, None
            try:
                self._process.stdin.write(frame)
                self._process.stdin.flush()
            except (OSError, ValueError):  # the window was closed
                self._closed = True
                return


def _run_window(title: str, width: int) -> None:
    import cv2

    # opencv-python points Qt to fonts it does not ship, use the system ones
    if (not os.path.isdir(os.environ.get("QT_QPA_FONTDIR", ""))
            and os.path.isdir("/usr/share/fonts")):
        os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"
    # and it has no Wayland plugin, Qt would warn and fall back to X11
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    latest: list = [None]
    done = threading.Event()

    def read() -> None:
        stdin = sys.stdin.buffer
        while True:
            header = stdin.read(_HEADER.size)
            if len(header) < _HEADER.size:
                break
            h, w, n, t = _HEADER.unpack(header)
            data = stdin.read(h * w * 3)
            if len(data) < h * w * 3:
                break
            latest[0] = (np.frombuffer(data, np.uint8).reshape(h, w, 3), n, t)
        done.set()

    threading.Thread(target=read, daemon=True).start()
    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
                    | cv2.WINDOW_GUI_EXPANDED)
    shown = None
    while not done.is_set():
        frame = latest[0]
        if frame is not None and frame is not shown:
            image, n, t = frame
            if shown is None:
                cv2.resizeWindow(title, width, round(width * image.shape[0]
                                                     / image.shape[1]))
            shown = frame
            cv2.imshow(title, image)
            # in the status bar, text drawn into the image would be scaled
            # with it
            cv2.displayStatusBar(title, f"frame {n}   exposed {t:.2f} s after start")
        cv2.waitKey(30)
        if shown is not None and cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_window(sys.argv[1], int(sys.argv[2]))

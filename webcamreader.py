import platform
import warnings
from collections import deque
from subprocess import DEVNULL, PIPE, CREATE_NO_WINDOW, Popen
from threading import Thread, Condition, Lock
from time import perf_counter

import numpy as np


class WebCamReader:
    def __init__(self, input_format, size, input_device, ffmpeg_bin='ffmpeg'):
        self.pipe = None
        self.worker = None

        self.width = size[0]
        self.height = size[1]
        self.frame_shape = size[1], size[0], 3

        cmd = [ffmpeg_bin,
               '-hide_banner',
               '-loglevel', 'error',
               '-f', input_format,
               # '-rtbufsize', '64M',
               '-video_size', f'{size[0]}x{size[1]}',
               '-i', f'video={input_device}',
               '-f', 'image2pipe',
               '-pix_fmt', 'bgr24',
               '-c:v', 'rawvideo',
               '-']
        self.nbytes = size[0] * size[1] * 3
        popen_kwargs = {
            'bufsize': self.nbytes * 3,
            'stdin': DEVNULL,
            'stdout': PIPE,
            'stderr': PIPE,
        }
        if platform.system() == 'Windows':
            popen_kwargs['creationflags'] = CREATE_NO_WINDOW

        self.pipe = Popen(cmd, **popen_kwargs)

        self.running = True
        self.d = deque(maxlen=10)
        self.lock = Lock()
        self.not_empty = Condition(self.lock)
        self.worker = Thread(target=self.update, args=())
        self.worker.start()

    def __del__(self):
        self.close()

    def close(self):
        if self.worker:
            self.running = False
            self.worker.join()
            self.worker = None
        if self.pipe:
            self.pipe.stdout.close()
            self.pipe.stderr.close()
            self.pipe.wait()
            self.pipe = None

    def update(self):
        while self.running:
            raw_bytes = self.pipe.stdout.read(self.nbytes)
            if not raw_bytes:
                error_msg = self.pipe.stderr.read().decode()
                raise OSError(error_msg)

            if len(raw_bytes) != self.nbytes:
                warnings.warn(f'Warning: {self.nbytes} bytes wanted but {len(raw_bytes)} bytes read.')

            frame = np.frombuffer(raw_bytes, np.uint8)
            frame.shape = self.frame_shape
            self.d.append(frame)
            with self.not_empty:
                self.not_empty.notify()

    def read_frame(self):
        if not len(self.d):
            with self.not_empty:
                self.not_empty.wait()
        return self.d.popleft()


class WebCamReader2:
    def __init__(self, input_format, size, input_device, ffmpeg_bin='ffmpeg'):
        self.pipe = None
        self.thread = None

        self.width = size[0]
        self.height = size[1]
        self.frame_shape = size[1], size[0], 3

        cmd = [ffmpeg_bin,
               '-hide_banner',
               '-loglevel', 'error',
               '-f', input_format,
               '-video_size', f'{size[0]}x{size[1]}',
               '-i', f'video={input_device}',
               '-f', 'image2pipe',
               '-pix_fmt', 'bgr24',
               '-c:v', 'rawvideo',
               '-']
        self.nbytes = size[0] * size[1] * 3
        popen_kwargs = {
            'bufsize': self.nbytes * 3,
            'stdin': DEVNULL,
            'stdout': PIPE,
            'stderr': PIPE,
        }
        if platform.system() == 'Windows':
            popen_kwargs['creationflags'] = CREATE_NO_WINDOW

        self.pipe = Popen(cmd, **popen_kwargs)

        self.running = True
        self.cv = Condition()
        self.last_frame = None
        self.thread = Thread(target=self.update, args=())
        self.thread.start()

    def __del__(self):
        self.close()

    def close(self):
        if self.thread:
            self.running = False
            self.thread.join()
            self.thread = None
        if self.pipe:
            self.pipe.stdout.close()
            self.pipe.stderr.close()
            self.pipe.wait()
            self.pipe = None

    def update(self):
        while self.running:
            t_start = perf_counter()
            raw_bytes = self.pipe.stdout.read(self.nbytes)
            if not raw_bytes:
                error_msg = self.pipe.stderr.read().decode()
                raise OSError(error_msg)
            if len(raw_bytes) != self.nbytes:
                warnings.warn(f'Warning: {self.nbytes} bytes wanted but {len(raw_bytes)} bytes read.')

            frame = np.frombuffer(raw_bytes, np.uint8)
            frame.shape = self.frame_shape
            with self.cv:
                self.last_frame = frame
                self.cv.notify()
            t_end = perf_counter()
            print(f'{(t_end - t_start) * 1000:.3} ms')

    def read_frame(self):
        with self.cv:
            self.cv.wait()
            return self.last_frame

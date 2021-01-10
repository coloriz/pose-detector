import pickle
import platform
from abc import ABC, abstractmethod
from pathlib import Path
from subprocess import Popen, PIPE, DEVNULL, CREATE_NO_WINDOW
from time import perf_counter


class IDataRecorder(ABC):
    @abstractmethod
    def write_data(self, frame, keypoints, state, current_pose_index, angle, confidence, score):
        pass

    @abstractmethod
    def close(self):
        pass


class DummyDataRecorder(IDataRecorder):
    def write_data(self, frame, keypoints, state, current_pose_index, angle, confidence, score):
        ...

    def close(self):
        ...


class DataRecorder(IDataRecorder):
    def __init__(self, size, fps, output, ffmpeg_bin='ffmpeg', codec='libx264', pix_fmt='yuv420p', crf=23):
        self.output_path = Path(output)
        self.output_path.mkdir(parents=True, exist_ok=True)

        cmd = [ffmpeg_bin,
               '-hide_banner',
               '-loglevel', 'error',
               '-f', 'rawvideo',
               '-s', f'{size[0]}x{size[1]}',
               '-r', f'{fps}',
               '-pix_fmt', 'bgr24',
               '-i', '-',
               '-an',
               '-c:v', codec,
               '-pix_fmt', pix_fmt,
               '-crf', f'{crf}',
               '-y',
               f'{self.output_path / "screen.mp4"}']
        popen_kwargs = {
            'stdin': PIPE,
            'stdout': DEVNULL,
            'stderr': PIPE,
        }
        if platform.system() == 'Windows':
            popen_kwargs['creationflags'] = CREATE_NO_WINDOW

        self.pipe = Popen(cmd, **popen_kwargs)
        self.timestamp_list = []
        self.keypoints_list = []
        self.state_list = []
        self.current_pose_list = []
        self.angle_list = []
        self.confidence_list = []
        self.score_list = []

    def write_data(self, frame, keypoints, state, current_pose_index, angle, confidence, score):
        try:
            self.pipe.stdin.write(frame.tobytes())
            self.timestamp_list.append(perf_counter())
            self.keypoints_list.append(keypoints)
            self.state_list.append(state)
            self.current_pose_list.append(current_pose_index)
            self.angle_list.append(angle)
            self.confidence_list.append(confidence)
            self.score_list.append(score)
        except OSError as e:
            _, err_stream = self.pipe.communicate()
            print(err_stream.decode())
            raise e

    def close(self):
        with open(self.output_path / 'data.pkl', 'wb') as f:
            pickle.dump({
                'timestamp': self.timestamp_list,
                'keypoints': self.keypoints_list,
                'state': self.state_list,
                'current_pose': self.current_pose_list,
                'angle': self.angle_list,
                'confidence': self.confidence_list,
                'score': self.score_list
            }, f)
        self.pipe.stderr.close()
        self.pipe.stdin.flush()
        self.pipe.stdin.close()
        self.pipe.wait()

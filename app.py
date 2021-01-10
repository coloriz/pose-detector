from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import deque
from enum import Enum, auto
from pathlib import Path
from time import perf_counter, time
from typing import List, Iterator

import cv2 as cv
import numpy as np

from datarecorder import DataRecorder, DummyDataRecorder
from posture import (
    Posture, Pose_01, Pose_02, Pose_03, Pose_04, Pose_05, Pose_06, Pose_07, Pose_08, Pose_09, Pose_10
)
from ui import InformationLayer
from utils import put_text, Assets, generate_info_layers


class State(Enum):
    Ready = auto()
    PoseReady = auto()
    PoseMeasuring = auto()
    Finish = auto()


class StateMachine:
    def __init__(self, opt):
        self.session_id = opt.session_id
        # Initialize camera and window
        self.cap = cv.VideoCapture(opt.input_device, eval(f'cv.{opt.input_api}'))
        assert self.cap.isOpened(), 'Failed to initialize video capture!'

        self.width, self.height = opt.video_width, opt.video_height
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        self.window_name = 'tutorial'
        if opt.fullscreen:
            cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
            cv.setWindowProperty(self.window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        else:
            cv.namedWindow(self.window_name)
        self.width_qr = self.width // 4
        self.width_half = self.width // 2
        self.width_3qr = self.width // 4 * 3
        self.border_margin = round(max(self.width, self.height) * 0.025)

        # Initialize openpose library
        import os
        import sys
        sys.path.append(os.fspath(opt.op_path/'python/openpose/Release'))
        os.environ['PATH'] += f';{os.fspath(opt.op_path/"x64/Release")}'
        os.environ['PATH'] += f';{os.fspath(opt.op_path/"bin")}'

        try:
            import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found.')
            raise e

        self._op_wrapper = op.WrapperPython()
        self._op_wrapper.configure({
            'model_folder': os.fspath(opt.op_path/'../models/'),
            'model_pose': 'BODY_25',
            'number_people_max': 1
        })
        self._op_wrapper.start()
        self._datum = op.Datum()

        # Raw data recorder
        if not opt.no_save:
            self.recorder = DataRecorder((self.width, self.height), 24, f'records/{opt.session_id}/{int(time())}')
        else:
            self.recorder = DummyDataRecorder()

        # Handler of each state
        self.handlers = {
            State.Ready: self.handle_ready,
            State.PoseReady: self.handle_pose_ready,
            State.PoseMeasuring: self.handle_pose_measuring,
            State.Finish: self.handle_finish,
        }

        # Load assets
        self.assets = Assets('assets/')
        self.poses: List[Posture] = [Pose_01, Pose_02, Pose_03, Pose_04, Pose_05, Pose_06, Pose_08, Pose_09, Pose_10]
        self.poses_ui: List[InformationLayer] = generate_info_layers(self.poses)

        self.running: bool = True
        self.state: State = State.Ready
        self.pose_index_iter: Iterator[int] = iter(range(len(self.poses)))
        self.current_pose_i: int = 0
        self.t_start: float = perf_counter()
        self.fail_counter = deque([False] * opt.fail_tolerance, maxlen=opt.fail_tolerance)
        self.angle: float = np.pi
        self.confidence: float = 0
        self.score: int = 0

        self.keypoints: np.ndarray = np.array([], np.float32)
        self.frame: np.ndarray = np.array([], np.uint8)

    def close(self):
        self.recorder.close()
        self._op_wrapper.stop()
        cv.destroyAllWindows()
        self.cap.release()

    def check_failed(self) -> bool:
        return all(self.fail_counter)

    @property
    def keypoints_detected(self) -> bool:
        return self.keypoints.ndim == 3

    def run(self):
        datum = self._datum

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            datum.cvInputData = frame
            self._op_wrapper.emplaceAndPop([datum])
            self.keypoints = datum.poseKeypoints
            self.frame = datum.cvOutputData

            self.handlers[self.state]()

            put_text(self.frame, self.state.name, (self.border_margin, self.border_margin), (0, 255, 0))
            cv.imshow(self.window_name, self.frame)
            key = cv.waitKey(1) & 0xff
            self.handle_input(key)

            self.recorder.write_data(self.frame, self.keypoints, self.state.name, self.poses[self.current_pose_i].name, self.angle, self.confidence, self.score)

        self.close()

    def __display_figure(self):
        """왼쪽 화면의 절반을 figure로 채움"""
        self.frame[:, self.width_half:] = self.frame[:, self.width_qr:self.width_3qr]
        self.frame[:, :self.width_half] = self.poses[self.current_pose_i].figure[:self.height, :self.width]

    def handle_input(self, key: int):
        if key == 27:  # esc
            self.running = False
        elif key == 99:  # c
            np.save('pose.npy', self.keypoints)
        elif key == 97:  # a
            cv.imwrite(f'{int(time())}.jpg', self.frame)

    def handle_ready(self) -> None:
        """준비 화면 띄우고 3초 뒤 측정 시작 화면으로 전환"""
        self.frame = self.assets.are_you_ready
        if (perf_counter() - self.t_start) >= 3:
            self.state = State.PoseReady
            self.current_pose_i = next(self.pose_index_iter)

    def handle_pose_ready(self) -> None:
        self.__display_figure()
        self.poses_ui[self.current_pose_i].alpha_composite(self.frame, 60, self.height - 220, 0)
        pose = self.poses[self.current_pose_i]
        self.angle = np.pi
        self.confidence = 0
        self.score = 0

        if not self.keypoints_detected:
            return

        person = self.keypoints[0]
        self.angle = pose.angle_difference(person)
        self.confidence = pose.average_confidence(person)
        self.score = pose.compute_matching_score(person)
        put_text(self.frame, f'angle: {self.angle:.3f} / score: {self.score:.3f} / conf: {self.confidence:.3f}', (self.border_margin, self.height - self.border_margin), (255, 255, 255))
        if self.score >= pose.threshold:
            self.t_start = perf_counter()
            self.state = State.PoseMeasuring

    def handle_pose_measuring(self) -> None:
        self.__display_figure()
        elapsed = perf_counter() - self.t_start
        self.poses_ui[self.current_pose_i].alpha_composite(self.frame, 60, self.height - 220, elapsed)
        pose = self.poses[self.current_pose_i]

        if not self.keypoints_detected:
            self.state = State.PoseReady
            return

        person = self.keypoints[0]
        self.angle = pose.angle_difference(person)
        self.confidence = pose.average_confidence(person)
        self.score = pose.compute_matching_score(person)
        put_text(self.frame, f'angle: {self.angle:.3f} / score: {self.score:.3f} / conf: {self.confidence:.3f}', (self.border_margin, self.height - self.border_margin), (255, 255, 255))
        # 만약 측정 도중 자세가 틀어지면 다시 PoseReady로
        self.fail_counter.append(self.score < pose.threshold)
        if self.check_failed():
            self.state = State.PoseReady

        if elapsed >= pose.duration:
            try:
                self.current_pose_i = next(self.pose_index_iter)
                self.state = State.PoseReady
            except StopIteration:
                self.state = State.Finish

    def handle_finish(self) -> None:
        self.running = False


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('session_id')
    parser.add_argument('--input-api', default='CAP_DSHOW', help='preferred Capture API backends to use.')
    parser.add_argument('--input-device', default=0, type=int, help='id of the video capturing device to open.')
    parser.add_argument('--video-width', default=1280, type=int, help='width of the frames in the video stream.')
    parser.add_argument('--video-height', default=720, type=int, help='height of the frames in the video stream.')
    parser.add_argument('-f', '--fullscreen', action='store_true', help='run the app in full screen mode.')
    parser.add_argument('-n', '--no-save', action='store_true', help='do not save the session.')
    parser.add_argument('--fail-tolerance', default=5, type=int, help='consecutive count to be evaluated as a failure.')
    parser.add_argument('--op-path', type=Path, required=True, help='OpenPose build path')
    opt = parser.parse_args()

    fsm = StateMachine(opt)
    fsm.run()


if __name__ == '__main__':
    main()

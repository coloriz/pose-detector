from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping, Union, List

import numpy as np
from numpy.linalg import norm

from keypoint import Keypoint

CONFIDENCE_THRESHOLD = 0.1
PAIRS = np.array(
    [(1, 0, 2), (1, 0, 5), (1, 0, 8), (1, 2, 5),  # 0 1 2 3
     (2, 1, 3),  # 4
     (3, 2, 4),  # 5
     (5, 1, 6),  # 6
     (6, 5, 7),  # 7
     (8, 1, 9), (8, 1, 12), (8, 9, 12),  # 8 9 10
     (9, 8, 10),  # 11
     (10, 9, 11),  # 12
     (12, 8, 13),  # 13
     (13, 12, 14)]  # 14
)
EPS = np.finfo(np.float32).eps


class PreCondition(ABC):
    def _check_confidences(self, person: np.ndarray) -> bool:
        confidences = person[self.indices, 2]
        return np.all(confidences >= CONFIDENCE_THRESHOLD)

    def __call__(self, person: np.ndarray) -> bool:
        if not self._check_confidences(person):
            return False
        return self.call(person)

    @property
    @abstractmethod
    def indices(self) -> List[int]:
        pass

    @abstractmethod
    def call(self, person: np.ndarray) ->  bool:
        pass


class AlwaysTrue(PreCondition):
    def __init__(self):
        self._indices = []

    @property
    def indices(self) -> List[int]:
        return self._indices

    def call(self, person: np.ndarray) -> bool:
        return True


class TwoArmsUpperThanNose(PreCondition):
    def __init__(self):
        self._indices = [Keypoint.Nose, Keypoint.RWrist, Keypoint.LWrist]

    @property
    def indices(self) -> List[int]:
        return self._indices

    def call(self, person: np.ndarray) -> bool:
        nose_y = person[Keypoint.Nose, 1]
        hands_y = person[[Keypoint.RWrist, Keypoint.LWrist], 1]

        return np.all(hands_y < nose_y)


class StandingWithLeftLeg(PreCondition):
    def __init__(self, ratio=0.6):
        self._indices = [Keypoint.Neck, Keypoint.MidHip, Keypoint.LAnkle, Keypoint.RAnkle]
        self.ratio = ratio

    @property
    def indices(self) -> List[int]:
        return self._indices

    def call(self, person: np.ndarray) -> bool:
        torso_length = norm(person[Keypoint.Neck, :2] - person[Keypoint.MidHip, :2])
        ankle_diff = person[Keypoint.LAnkle, 1] - person[Keypoint.RAnkle, 1]

        return torso_length * self.ratio < ankle_diff


class StandingWithRightLeg(PreCondition):
    def __init__(self, ratio=0.6):
        self._indices = [Keypoint.Neck, Keypoint.MidHip, Keypoint.LAnkle, Keypoint.RAnkle]
        self.ratio = ratio

    @property
    def indices(self) -> List[int]:
        return self._indices

    def call(self, person: np.ndarray) -> bool:
        torso_length = norm(person[Keypoint.Neck, :2] - person[Keypoint.MidHip, :2])
        ankle_diff = person[Keypoint.RAnkle, 1] - person[Keypoint.LAnkle, 1]

        return torso_length * self.ratio < ankle_diff


class StandingLookRight(PreCondition):
    def __init__(self, threshold=20):
        self._indices = [Keypoint.LShoulder, Keypoint.LHip, Keypoint.LAnkle, Keypoint.LBigToe]
        self.threshold = threshold

    @property
    def indices(self) -> List[int]:
        return self._indices

    def call(self, person: np.ndarray) -> bool:
        leftside_x = person[[Keypoint.LShoulder, Keypoint.LHip, Keypoint.LAnkle], 0]
        return np.std(leftside_x) <= self.threshold and person[Keypoint.LBigToe, 0] < person[Keypoint.LAnkle, 0]


class LeftArmStretchedOut(PreCondition):
    def __init__(self, threshold=20):
        self._indices = [Keypoint.LShoulder, Keypoint.LElbow, Keypoint.LWrist]
        self.threshold = threshold

    @property
    def indices(self) -> List[int]:
        return self._indices

    def call(self, person: np.ndarray) -> bool:
        leftarm_y = person[[Keypoint.LShoulder, Keypoint.LElbow, Keypoint.LWrist], 1]
        return np.std(leftarm_y) <= self.threshold


class RightWristAtLeftmost(PreCondition):
    def __init__(self):
        self._indices = [Keypoint.RWrist]

    @property
    def indices(self) -> List[int]:
        return self._indices

    def call(self, person: np.ndarray) -> bool:
        rightwrist_x = person[Keypoint.RWrist, 0]
        valid_keypoints_x = person[person[..., 2] >= CONFIDENCE_THRESHOLD, 0]
        return np.all(valid_keypoints_x <= rightwrist_x)


class LeftWristAtRightmost(PreCondition):
    def __init__(self):
        self._indices = [Keypoint.LWrist]

    @property
    def indices(self) -> List[int]:
        return self._indices

    def call(self, person: np.ndarray) -> bool:
        leftwrist_x = person[Keypoint.LWrist, 0]
        valid_keypoints_x = person[person[..., 2] >= CONFIDENCE_THRESHOLD, 0]
        return np.all(valid_keypoints_x >= leftwrist_x)


PRECONDITIONS = np.array(
    [AlwaysTrue(),
     TwoArmsUpperThanNose(),
     StandingWithLeftLeg(),
     StandingWithRightLeg(),
     StandingLookRight(),
     LeftArmStretchedOut(),
     RightWristAtLeftmost(),
     LeftWristAtRightmost()]
)


class Posture:
    def __init__(self, path: Union[str, Path]):
        data: Mapping = np.load(path)
        # 자세 사진 (640x720)
        self.figure: np.ndarray = data['figure']
        # 사전 조건
        self.preconditions: np.ndarray = PRECONDITIONS[data['preconditions']]
        # 각도 계산에 사용되는 PAIRS
        self.pairs: np.ndarray = PAIRS[data['pair_indices']]
        # 자세 계산에 필요한 인덱스
        self.indices: np.ndarray = np.unique(self.pairs)
        # 키포인트 좌표 (debug)
        self.points: np.ndarray = data['points']
        # PAIRS의 각도
        self.angles: np.ndarray = data['angles']
        # 각도 임계값
        self.threshold: float = float(data['threshold'])
        # 자세 지속 요구 시간
        self.duration: int = int(data['duration'])
        # 자세 이름
        self.name: str = str(data['name'])

    def cosine_similarity(self, person: np.ndarray):
        points = person[:, :2]
        return np.mean(np.sum(self.points * points, axis=1) / (norm(self.points, axis=1) * norm(points, axis=1)))

    def angle_difference_bak(self, person: np.ndarray):
        """계산에 필요한 인덱스 중 하나라도 threshold값을 못 넘으면 np.pi 리턴"""
        confidences = person[self.indices, 2]
        if np.any(confidences < CONFIDENCE_THRESHOLD):
            return np.pi
        points = person[:, :2]
        v1s = points[self.pairs[:, 1]] - points[self.pairs[:, 0]]
        v2s = points[self.pairs[:, 2]] - points[self.pairs[:, 0]]
        angles = np.arccos(np.sum(v1s * v2s, axis=1) / (norm(v1s, axis=1) * norm(v2s, axis=1)))
        return np.mean(np.abs(self.angles - angles))

    def angle_difference(self, person: np.ndarray):
        # if not all(map(lambda cond: cond(person), self.preconditions)):
        #     return np.pi
        points = person[:, :2]
        v1 = points[self.pairs[:, 1]] - points[self.pairs[:, 0]]
        v2 = points[self.pairs[:, 2]] - points[self.pairs[:, 0]]
        d = norm(v1, axis=1) * norm(v2, axis=1)
        d[d < EPS] = EPS
        cosine = np.clip(np.sum(v1 * v2, axis=1) / d, -1, 1)
        angles = np.arccos(cosine)
        differences = self.angles - angles
        # self.pairs의 한 벡터 집합에서 하나의 인덱스라도 confidence가 CONFIDENCDE_THRESHOLD보다
        # 낮은 인덱스가 있으면 최종 차이값에서 그 부분의 angle은 최대값인 np.pi로 대체
        for i in range(PAIRS.shape[1]):
            confidences = person[self.pairs[:, i], 2]
            differences[confidences < CONFIDENCE_THRESHOLD] = np.pi

        return np.mean(np.abs(differences))

    def compute_matching_score(self, person: np.ndarray):
        """0 ~ 1"""
        angle = self.angle_difference(person)
        n_conditions = len(self.preconditions)
        n = sum(map(lambda cond: cond(person), self.preconditions))
        penalty_condition = (n / n_conditions * (n_conditions - 1) + 1) / n_conditions
        return np.clip(self._normalize_angle(angle) * penalty_condition, 0, 1)

    def average_confidence(self, person: np.ndarray):
        confidences = person[self.indices, 2]
        return np.mean(confidences)

    @staticmethod
    def _normalize_angle(angle):
        return (np.pi - angle) / np.pi


Pose_01 = Posture('posture_data/pose_01.npz')
Pose_02 = Posture('posture_data/pose_02.npz')
Pose_03 = Posture('posture_data/pose_03.npz')
Pose_04 = Posture('posture_data/pose_04.npz')
Pose_05 = Posture('posture_data/pose_05.npz')
Pose_06 = Posture('posture_data/pose_06.npz')
Pose_07 = Posture('posture_data/pose_07.npz')
Pose_08 = Posture('posture_data/pose_08.npz')
Pose_09 = Posture('posture_data/pose_09.npz')
Pose_10 = Posture('posture_data/pose_10.npz')

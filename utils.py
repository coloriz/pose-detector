from pathlib import Path
from typing import Tuple, TypeVar, List

import cv2 as cv
import numpy as np

from posture import Posture
from ui import InformationLayer


def put_text(img, text, org, color, align='left', font=cv.FONT_HERSHEY_SIMPLEX, scale=None, thickness=None):
    font_scale = scale or 0.8
    font_thickness = thickness or 2
    shadow_offset = 2
    text_size, baseline = cv.getTextSize(text, font, font_scale, font_thickness)
    baseline += font_thickness
    if align == 'left':
        final_position = (org[0], org[1] + text_size[1] // 2)
    elif align == 'center':
        final_position = (org[0] - text_size[0] // 2, org[1] + text_size[1] // 2)
    elif align == 'right':
        final_position = (org[0] - text_size[0], org[1] + text_size[1] // 2)
    else:
        raise ValueError('alignment should be one of ["left", "center", "right"].')

    cv.putText(img, text, (final_position[0] + shadow_offset, final_position[1] + shadow_offset),
               font, font_scale, (0, 0, 0, 255), font_thickness, cv.LINE_AA)
    cv.putText(img, text, final_position, font, font_scale, color, font_thickness, cv.LINE_AA)


T = TypeVar('T')
Rectangle = Tuple[T, T, T, T]


def get_bounding_rect(person: np.ndarray, threshold: float) -> Rectangle:
    threshed = person[person[:, 2] > threshold]
    return cv.boundingRect(threshed[np.newaxis, :, :2]) if threshed.size > 0 else (0, 0, 0, 0)


def get_keypoints_rectangle(keypoints: np.ndarray, threshold: float = 0.1) -> List[Rectangle]:
    if keypoints.ndim != 3:
        return []

    return [get_bounding_rect(person, threshold) for person in keypoints]


def draw_rectangles(img: np.ndarray, rectangles: List[Rectangle], color, thickness=None):
    for rect in rectangles:
        if rect[2] * rect[3] > 0:  # w * h > 0
            cv.rectangle(img, rect, color, thickness)


def normalize_points(points: np.ndarray) -> np.ndarray:
    """y방향의 ptp로 scaling"""
    return 2 * (points - np.min(points, axis=0)) / np.ptp(points[:, 1], axis=0) - 1


class Assets:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.are_you_ready = self.__load_asset('are-you-ready.png')

    def __load_asset(self, filename: str):
        return cv.imread(str(self.base_path / filename))


def angle_to_score(angle: float) -> int:
    """angle: 0 ~ pi"""
    return int(np.around((np.pi - angle) / np.pi * 100))


def generate_info_layers(poses: List[Posture]):
    total = len(poses)
    return [InformationLayer(pose.name, i + 1, total, pose.duration) for i, pose in enumerate(poses)]

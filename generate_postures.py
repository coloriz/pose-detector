from io import BytesIO
from pathlib import Path
from typing import NamedTuple, List

import cv2 as cv
import grpc
import numpy as np
from numpy.linalg import norm

import pose_pb2
import pose_pb2_grpc
from posture import PAIRS


def get_keypoints_and_rendered_image(img: np.ndarray):
    _, data = cv.imencode('.jpg', img)
    with grpc.insecure_channel('localhost:50051') as channel, BytesIO() as buf:
        np.save(buf, data)
        stub = pose_pb2_grpc.PoseStub(channel)
        response = stub.GetKeypoints(pose_pb2.Image(data=buf.getvalue()))

    with BytesIO(response.keypoints) as buf1, BytesIO(response.painted) as buf2:
        keypoints = np.load(buf1)
        rendered = cv.imdecode(np.load(buf2), cv.IMREAD_COLOR)

    return keypoints, rendered


class RawPose(NamedTuple):
    filename: str
    preconditions: List[int]
    pair_indices: List[int]
    threshold: float
    duration: int
    name: str


def generate_postures():
    raw_poses = [
        RawPose('raws/pose_01.png', [0], [7, 8, 9, 10, 11, 12, 13, 14], 0.92, 5, '왼쪽무릎'),
        RawPose('raws/pose_02.png', [0], [5, 8, 9, 10, 11, 12, 13, 14], 0.92, 5, '오른쪽무릎'),
        RawPose('raws/pose_03.png', [0, 6], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 0.9, 5, '오른쪽허리'),
        RawPose('raws/pose_04.png', [0, 7], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 0.9, 5, '왼쪽허리'),
        RawPose('raws/pose_05.png', [0, 4, 5], [7, 14], 0.95, 5, '팔뻗기정면'),
        RawPose('raws/pose_06.png', [0, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 0.95, 5, '팔뻗기위로'),
        RawPose('raws/pose_07.png', [0], [3, 4, 5, 6, 7], 1.0, 5, '몸굽히기'),
        RawPose('raws/pose_08.png', [0, 3], [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12], 0.93, 5, '왼무릎들기'),
        RawPose('raws/pose_09.png', [0, 2], [0, 1, 2, 3, 4, 5, 6, 7, 9, 13, 14], 0.93, 5, '오른무릎들기'),
        RawPose('raws/pose_10.png', [0], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 0.96, 5, '숨고르기'),
    ]

    for pose in raw_poses:
        img = cv.imread(pose.filename)
        img = cv.resize(img, (1280, 720))
        keypoints, rendered = get_keypoints_and_rendered_image(img)
        figure = img[:720, 320:960]
        pair_indices = np.array(pose.pair_indices)
        pairs = PAIRS[pair_indices]

        points = keypoints[0, :, :2]
        v1s = points[pairs[:, 1]] - points[pairs[:, 0]]
        v2s = points[pairs[:, 2]] - points[pairs[:, 0]]
        angles = np.arccos(np.sum(v1s * v2s, axis=1) / (norm(v1s, axis=1) * norm(v2s, axis=1)))
        print(angles)
        stem = Path(pose.filename).stem
        np.savez_compressed(f'posture_data/{stem}.npz',
                            figure=figure,
                            preconditions=pose.preconditions,
                            pair_indices=pair_indices,
                            points=points,
                            angles=angles,
                            threshold=pose.threshold,
                            duration=pose.duration,
                            name=pose.name)

        cv.imshow('rendered', rendered)
        cv.imshow('figure', figure)
        cv.waitKey(1)


if __name__ == '__main__':
    generate_postures()

import os
import sys
from argparse import ArgumentParser
from concurrent.futures.thread import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import cv2 as cv
import grpc
import numpy as np

import pose_pb2
import pose_pb2_grpc


class Pose(pose_pb2_grpc.PoseServicer):
    def __init__(self):
        super().__init__()

        params = {
            'model_folder': os.fspath(opt.op_path/'../models/'),
            # 'render_pose': 0,
            'model_pose': 'BODY_25',
            'number_people_max': 1,
        }
        self.op_wrapper = op.WrapperPython()
        self.op_wrapper.configure(params)
        self.op_wrapper.start()
        self.datum = op.Datum()

    def GetKeypoints(self, request, context):
        with BytesIO(request.data) as buf:
            encoded = np.load(buf)
        img = cv.imdecode(encoded, cv.IMREAD_COLOR)
        self.datum.cvInputData = img
        self.op_wrapper.emplaceAndPop([self.datum])
        keypoints = self.datum.poseKeypoints
        painted = self.datum.cvOutputData

        # with BytesIO() as buf:
        #     np.save(buf, keypoints)
        #     return pose_pb2.Keypoints(keypoints=buf.getvalue())

        with BytesIO() as buf1, BytesIO() as buf2:
            np.save(buf1, keypoints)
            _, data = cv.imencode('.jpg', painted)
            np.save(buf2, data)
            return pose_pb2.Keypoints(keypoints=buf1.getvalue(), painted=buf2.getvalue())


def serve():
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    pose_pb2_grpc.add_PoseServicer_to_server(Pose(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--op-path', type=Path, required=True, help='OpenPose build path')
    opt = parser.parse_args()

    sys.path.append(os.fspath(opt.op_path / 'python/openpose/Release'))
    os.environ['PATH'] += f';{os.fspath(opt.op_path / "x64/Release")}'
    os.environ['PATH'] += f';{os.fspath(opt.op_path / "bin")}'

    try:
        import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found.')
        raise e

    serve()

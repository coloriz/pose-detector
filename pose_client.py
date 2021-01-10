from io import BytesIO

import cv2 as cv
import grpc
import numpy as np

import pose_pb2
import pose_pb2_grpc


def run():
    cap = cv.VideoCapture(0)
    assert cap.isOpened(), 'Failed to initialize video capture!'
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    window_name = 'tutorial'
    cv.namedWindow(window_name)

    with grpc.insecure_channel('localhost:50051') as channel:
        stub = pose_pb2_grpc.PoseStub(channel)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, data = cv.imencode('.jpg', frame)
            with BytesIO() as buf:
                np.save(buf, data)
                response = stub.GetKeypoints(pose_pb2.Image(data=buf.getvalue()))
            with BytesIO(response.keypoints) as buf1, BytesIO(response.painted) as buf2:
                keypoints = np.load(buf1)
                frame = cv.imdecode(np.load(buf2), cv.IMREAD_COLOR)

            cv.imshow(window_name, frame)
            if cv.waitKey(1) & 0xff == 27:
                break


if __name__ == '__main__':
    run()

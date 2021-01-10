import pickle
from argparse import ArgumentParser, FileType
from collections import OrderedDict

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


def plot_session(data):
    font = fm.FontProperties(fname='assets/NanumSquareRoundR.ttf', size=10)

    # with open('records/10/1605236896/data.pkl', 'rb') as f:
    #     data = pickle.load(f)

    length = len(data['timestamp'])
    t_start = data['timestamp'][0]
    timeseries = [t - t_start for t in data['timestamp']]
    pose_to_color = OrderedDict.fromkeys(data['current_pose'])
    for i, key in enumerate(pose_to_color.keys()):
        pose_to_color[key] = f'C{i}'

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(timeseries[0], timeseries[-1])
    ax.set_ylim(0, 1)

    for pose, color in pose_to_color.items():
        x = [i for i in range(length) if data['state'][i] != 'Ready' and data['current_pose'][i] == pose]
        # To make it discontinuous
        x = x[1:]
        # x = [i for i in range(length) if data['state'][i] == 'PoseMeasuring' and data['current_pose'][i] == pose]
        y = [data['score'][i] for i in x]
        ax.plot([timeseries[i] for i in x], y, color, label=pose)
    ax.legend(prop=font)

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('infile', type=FileType('rb'), help='path to data.pkl')
    opt = parser.parse_args()
    data = pickle.load(opt.infile)
    plot_session(data)

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns


def load_json_logs(json_logs):
    # load and convert json_logs to log_dict, key is epoch, value is a sub dict
    # keys of sub dict is different metrics, e.g. memory, bbox_mAP
    # value of sub dict is a list of corresponding values of all iterations
    log_dicts = [dict() for _ in json_logs]
    for json_log, log_dict in zip(json_logs, log_dicts):
        with open(json_log, 'r') as log_file:
            for line in log_file:
                log = json.loads(line.strip())
                # skip lines without `epoch` field
                if 'epoch' not in log:
                    continue
                epoch = log.pop('epoch')
                if epoch not in log_dict:
                    log_dict[epoch] = defaultdict(list)
                for k, v in log.items():
                    log_dict[epoch][k].append(v)
    return log_dicts


def print_loss(log_dicts_train, parameter):
    train_loss = []
    epochs = log_dicts_train[0].keys()
    for epoch in epochs:
        train_loss += log_dicts_train[0][epoch][parameter]

    # for i, log_dicts_test in enumerate(log_dicts_test):
    #     epochs += log_dicts_train[i]["epoch"]
    #     test_loss += log_dicts_train[i]["epoch"]

    # Visualize loss history
    return epochs, train_loss


def plot_loss_cls():
    faster_rcnn_train = ["/Users/iotta/Master/Code/mmdetection/logs/fasterrcnn/faster_rcnn_train_24.json"]
    repppoints_train = ["/Users/iotta/Master/Code/mmdetection/logs/repppoints/repppoints_train_24.json"]
    # json_logs_train = ["/Users/iotta/Master/Code/mmdetection/logs/filtered/faster_rcnn_train_112.json"]
    # json_logs_test = ["/Users/iotta/Master/Code/mmdetection/logs/filtered/faster_rcnn_test_112.json"]
    log_dicts_train = load_json_logs(faster_rcnn_train)
    log_dicts_test = load_json_logs(repppoints_train)

    epochs_fasterrcnn, train_loss_fasterrcnn = print_loss(log_dicts_train, "loss_cls")
    epochs_repppoints, train_loss_repppoints = print_loss(log_dicts_test, "loss_cls")
    # Visualize loss history
    plt.plot(epochs_fasterrcnn, train_loss_fasterrcnn, '#EB1E4E', '-')
    plt.plot(epochs_repppoints, train_loss_repppoints, '#118AB2', '-')
    # plt.plot(epochs, test_loss, 'b-')
    plt.legend(['FasterRCNN', 'RepPoints'])  # , 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(top=1)  # ymax is your value
    plt.ylim(bottom=0)  # ymin is your value
    plt.title('Loss classification over ' + str(len(epochs_fasterrcnn)) + ' epochs')
    plt.savefig(
        "/Users/iotta/Master/Code/mmdetection/logs/plots/loss_cls_" + str(len(epochs_fasterrcnn)) + "_epochs.jpg")
    plt.show()


def plot_loss():
    faster_rcnn_train = ["/Users/iotta/Master/Code/mmdetection/logs/fasterrcnn/faster_rcnn_train_24.json"]
    repppoints_train = ["/Users/iotta/Master/Code/mmdetection/logs/repppoints/repppoints_train_24.json"]
    # json_logs_train = ["/Users/iotta/Master/Code/mmdetection/logs/filtered/faster_rcnn_train_112.json"]
    # json_logs_test = ["/Users/iotta/Master/Code/mmdetection/logs/filtered/faster_rcnn_test_112.json"]
    log_dicts_train = load_json_logs(faster_rcnn_train)
    log_dicts_test = load_json_logs(repppoints_train)

    epochs_fasterrcnn, train_loss_fasterrcnn = print_loss(log_dicts_train, "loss")
    epochs_repppoints, train_loss_repppoints = print_loss(log_dicts_test, "loss")
    # Visualize loss history
    plt.plot(epochs_fasterrcnn, train_loss_fasterrcnn, '#EB1E4E', '-')
    plt.plot(epochs_repppoints, train_loss_repppoints, '#118AB2', '-')
    # plt.plot(epochs, test_loss, 'b-')
    plt.legend(['FasterRCNN', 'RepPoints'])  # , 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(top=1)  # ymax is your value
    plt.ylim(bottom=0)  # ymin is your value
    plt.title('Loss over ' + str(len(epochs_fasterrcnn)) + ' epochs')
    plt.savefig(
        "/Users/iotta/Master/Code/mmdetection/logs/plots/loss_cls_" + str(len(epochs_fasterrcnn)) + "_epochs.jpg")
    plt.show()


def plot_mean_avarage(param):
    # faster_rcnn_train = ["/Users/iotta/Master/Code/mmdetection/logs/fasterrcnn/faster_rcnn_test.json"]
    # repppoints_train = ["/Users/iotta/Master/Code/mmdetection/logs/repppoints/repppoints_test.json"]
    faster_rcnn_train = ["/Users/iotta/Master/Code/mmdetection/logs/fasterrcnn/fasterrcnn_val_24.json"]
    repppoints_train = ["/Users/iotta/Master/Code/mmdetection/logs/repppoints/reppoints_val_24.json"]
    log_dicts_train = load_json_logs(faster_rcnn_train)
    log_dicts_test = load_json_logs(repppoints_train)

    epochs_fasterrcnn, train_loss_fasterrcnn = print_loss(log_dicts_train, param)
    epochs_repppoints, train_loss_repppoints = print_loss(log_dicts_test, param)
    # Visualize loss history
    plt.plot(epochs_fasterrcnn, train_loss_fasterrcnn, '#EB1E4E', '-')
    plt.plot(epochs_repppoints, train_loss_repppoints, '#118AB2', '--')
    # plt.plot(epochs, test_loss, 'b-')
    plt.legend(['FasterRCNN', 'RepPoints'])  # , 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel(param)
    plt.title(param + ' ,' + str(len(epochs_fasterrcnn)) + ' epochs')
    plt.savefig(
        "/Users/iotta/Master/Code/mmdetection/logs/plots/" + param + "_" + str(len(epochs_fasterrcnn)) + "_epochs.jpg")
    plt.show()


def main():
    # plot_loss_cls()
    plot_mean_avarage("bbox_mAP_l")


if __name__ == '__main__':
    main()

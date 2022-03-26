# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
from collections import defaultdict
import numpy as np
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
    faster_rcnn_train = [
        "/Users/iotta/Master/Code/mmdetection/logs/reppoints_augmented/repppoints_train_1350_iter.json"]
    repppoints_train = ["/Users/iotta/Master/Code/mmdetection/logs/reppoints/repppoints_train_300_iter.json"]
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
    plt.legend(['RepPoints with augmented data', 'RepPoints'])  # , 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(top=1)  # ymax is your value
    plt.ylim(bottom=0)  # ymin is your value
    plt.title('Loss classification over ' + str(len(epochs_fasterrcnn)) + ' epochs')
    plt.savefig(
        "/Users/iotta/Master/Code/mmdetection/logs/plots/augmented/loss_cls_" + str(
            len(epochs_fasterrcnn)) + "_epochs.jpg")
    plt.show()


def plot_loss(param):
    faster_rcnn_train = ["/Users/iotta/Master/Code/mmdetection/logs/fasterrcnn/fasterrcnn_train_300_iter_48.json"]
    repppoints_train = ["/Users/iotta/Master/Code/mmdetection/logs/repppoints/repppoints_train_300_iter.json"]
    log_dicts_train = load_json_logs(faster_rcnn_train)
    log_dicts_test = load_json_logs(repppoints_train)

    epochs_fasterrcnn, train_loss_fasterrcnn = print_loss(log_dicts_train, param)
    epochs_repppoints, train_loss_repppoints = print_loss(log_dicts_test, param)

    fig, ax1 = plt.subplots()
    color = '#EB1E4E'
    ax1.set_xlabel('num of epochs')
    ax1.set_ylabel(param)
    d1, = ax1.plot(epochs_fasterrcnn, train_loss_fasterrcnn, color=color, label="FasterRCNN")
    # plt.legend(['FasterRCNN', 'RepPoints'])
    color = '#118AB2'
    d2, = ax1.plot(epochs_repppoints, train_loss_repppoints, color=color, label="RepPoints")
    plt.legend(handles=[d1, d2])
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(param + ' ,' + str(len(epochs_fasterrcnn)) + ' epochs')
    # plt.axis([1, 50, 0, 1])
    plt.xlim(1, 50)
    plt.ylim(0, 1)
    plt.savefig(
        "/Users/iotta/Master/Code/mmdetection/logs/plots/" + param + "_" + str(len(epochs_fasterrcnn)) + "_epochs.jpg")
    plt.show()


def plot_mean_avarage(param, log1, log2, name1, name2, plot_path):
    faster_rcnn_train = log1
    repppoints_train = log2
    log_dicts1 = load_json_logs([faster_rcnn_train])
    log_dicts2 = load_json_logs([repppoints_train])

    epochs1, train_data1 = print_loss(log_dicts1, param)
    epochs2, train_data2 = print_loss(log_dicts2, param)

    fig, ax1 = plt.subplots()
    color = '#EB1E4E'
    ax1.set_xlabel('num of epochs')
    ax1.set_ylabel(param)
    d1, = ax1.plot(epochs1, train_data1, color=color, label=name1)
    # plt.legend(['FasterRCNN', 'RepPoints'])
    color = '#118AB2'
    d2, = ax1.plot(epochs2, train_data2, color=color, label=name2)
    plt.legend(handles=[d1, d2])
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(param + ' ,' + str(len(epochs1)) + ' epochs')
    plt.axis([1, 50, 0, 1])
    plt.xlim(1, 50)
    plt.ylim(0, 0.8)
    plt.savefig(plot_path + param + "_" + str(len(epochs1)) + "_epochs.jpg")
    plt.show()


def plot_reppoints_loss():
    repppoints_train = ["/Users/iotta/Master/Code/mmdetection/logs/repppoints/repppoints_train_300_iter.json"]
    log_dicts_test = load_json_logs(repppoints_train)

    epochs, loss = print_loss(log_dicts_test, "loss")
    epochs, loss_cls = print_loss(log_dicts_test, "loss_cls")
    epochs, loss_pts_init = print_loss(log_dicts_test, "loss_pts_init")
    epochs, loss_pts_refine = print_loss(log_dicts_test, "loss_pts_refine")

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('num of epochs')
    # ax1.set_ylabel(param)
    color1 = '#EB1E4E'
    d1, = ax1.plot(epochs, loss, color=color1, label="loss")
    # plt.legend(['FasterRCNN', 'RepPoints'])
    color2 = '#118AB2'
    d2, = ax1.plot(epochs, loss_cls, color=color2, label="loss_cls")
    color3 = '#049F76'
    d3, = ax1.plot(epochs, loss_pts_init, color=color3, label="loss_pts_init")
    color4 = '#073B4C'
    d4, = ax1.plot(epochs, loss_pts_refine, color=color4, label="loss_pts_refine")

    plt.legend(handles=[d1, d2, d3, d4])
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('RepPoints loss functions')
    # plt.axis([1, 50, 0, 1])
    # plt.xlim(1, 50)
    # plt.ylim(0, 0.6)
    plt.savefig(
        "/Users/iotta/Master/Code/mmdetection/logs/plots/" + 'repppoints_loss' + "_" + str(
            len(epochs)) + "_epochs.jpg")
    plt.show()


def plot_reppoints_map():
    repppoints_train = ["/Users/iotta/Master/Code/mmdetection/logs/repppoints/reppoints_val.json"]
    log_dicts_test = load_json_logs(repppoints_train)

    epochs, loss = print_loss(log_dicts_test, "bbox_mAP_s")
    epochs, loss_cls = print_loss(log_dicts_test, "bbox_mAP_m")
    epochs, loss_pts_init = print_loss(log_dicts_test, "bbox_mAP_l")

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('num of epochs')
    # ax1.set_ylabel(param)
    color1 = '#EB1E4E'
    d1, = ax1.plot(epochs, loss, color=color1, label="bbox_mAP_s")
    # plt.legend(['FasterRCNN', 'RepPoints'])
    color2 = '#118AB2'
    d2, = ax1.plot(epochs, loss_cls, color=color2, label="bbox_mAP_m")
    color3 = '#049F76'
    d3, = ax1.plot(epochs, loss_pts_init, color=color3, label="bbox_mAP_l")

    plt.legend(handles=[d1, d2, d3])
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('RepPoints mAP functions')
    # plt.axis([1, 50, 0, 1])
    # plt.xlim(1, 50)
    # plt.ylim(0, 0.6)
    plt.savefig(
        "/Users/iotta/Master/Code/mmdetection/logs/plots/" + 'repppoints_map_size' + "_" + str(
            len(epochs)) + "_epochs.jpg")
    plt.show()


def plot_reppoints_map(log_file, destination_graph_path, name):
    repppoints_train = [log_file]
    log_dicts_test = load_json_logs(repppoints_train)

    epochs, loss_m = print_loss(log_dicts_test, "bbox_mAP")
    epochs, loss_m50 = print_loss(log_dicts_test, "bbox_mAP_50")
    epochs, loss_m75 = print_loss(log_dicts_test, "bbox_mAP_75")
    epochs, loss = print_loss(log_dicts_test, "bbox_mAP_s")
    epochs, loss_cls = print_loss(log_dicts_test, "bbox_mAP_m")
    epochs, loss_pts_init = print_loss(log_dicts_test, "bbox_mAP_l")

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('num of epochs')
    # ax1.set_ylabel(param)
    color4 = '#FFD166'
    d4, = ax1.plot(epochs, loss_m, color=color4, label="bbox_mAP")
    # plt.legend(['FasterRCNN', 'RepPoints'])
    color5 = '#073B4C'
    d5, = ax1.plot(epochs, loss_m50, color=color5, label="bbox_mAP_50")
    color6 = '#460615'
    d6, = ax1.plot(epochs, loss_m75, color=color6, label="bbox_mAP_75")
    color1 = '#EB1E4E'
    d1, = ax1.plot(epochs, loss, color=color1, label="bbox_mAP_s")
    # plt.legend(['FasterRCNN', 'RepPoints'])
    color2 = '#118AB2'
    d2, = ax1.plot(epochs, loss_cls, color=color2, label="bbox_mAP_m")
    color3 = '#049F76'
    d3, = ax1.plot(epochs, loss_pts_init, color=color3, label="bbox_mAP_l")

    plt.legend(handles=[d4, d5, d6, d1, d2, d3])
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('RepPoints mAP functions')
    # plt.axis([1, 50, 0, 1])
    # plt.xlim(1, 50)
    # plt.ylim(0, 0.6)
    plt.savefig(destination_graph_path + name + "_" + str(len(epochs)) + "_epochs.jpg")
    plt.show()


def main():
    # plot_loss_cls()
    plot_mean_avarage("bbox_mAP_l")
    # plot_loss("loss_cls")
    # plot_reppoints_loss()
    # plot_reppoints_map()


def plot_single_log_graphs():
    # plot_loss_cls()
    # plot_mean_avarage("bbox_mAP_l")
    # plot_loss("loss_cls")""]"/Users/iotta/Master/Code/mmdetection/logs/plots/"
    # plot_reppoints_loss()
    # plot_reppoints_map()
    plot_reppoints_map("/Users/iotta/Master/Code/mmdetection/logs/reppoints_augmented_training/reppoints_val.json",
                       "/Users/iotta/Master/Code/mmdetection/logs/reppoints_augmented_training/plots/", "map")


if __name__ == '__main__':
    # plot_single_log_graphs()
    plot_mean_avarage("bbox_mAP_50", "/Users/iotta/Master/Code/mmdetection/logs/reppoints_augmented_training/reppoints_val.json",
                      "/Users/iotta/Master/Code/mmdetection/logs/reppoints/reppoints_val.json",
                      "RepPoints with augmentation", "RepPoints", "/Users/iotta/Master/Code/mmdetection/logs/reppoints_augmented_training/plots/")

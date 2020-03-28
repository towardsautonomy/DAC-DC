import csv
import configparser
import matplotlib.pyplot as plt

from src.configs import *

# function to plot metrics
def plot_metrics():
    # assemble metric parameters
    epoch = []
    xy_loss, wh_loss, iou, conf_loss, iou_loss, total_loss = [], [], [], [], [], []
    val_xy_loss, val_wh_loss, val_iou, val_conf_loss, val_iou_loss, val_total_loss = [], [], [], [], [], []
    with open(CSV_LOG_FILE) as file:
        reader = csv.DictReader( file )
        for line in reader:
            epoch.append(int(line['epoch']))
            # training losses and iou
            xy_loss.append(float(line['xy_loss']))
            wh_loss.append(float(line['wh_loss']))
            conf_loss.append(float(line['conf_loss']))
            iou_loss.append(float(line['iou_loss']))
            total_loss.append(float(line['total_loss']))
            iou.append(float(line['mean_iou']))
            # validation losses and iou
            val_xy_loss.append(float(line['val_xy_loss']))
            val_wh_loss.append(float(line['val_wh_loss']))
            val_conf_loss.append(float(line['val_conf_loss']))
            val_iou_loss.append(float(line['val_iou_loss']))
            val_total_loss.append(float(line['val_total_loss']))
            val_iou.append(float(line['val_mean_iou']))

    # plot metrics
    # losses
    fig, ax1 = plt.subplots(2, figsize=(12,12))
    ax1[0].set_title('Training Losses and IOU', fontweight='bold')
    # training losses and iou
    ax1[0].grid(linestyle='-', linewidth='0.2', color='gray')
    ax1[0].plot(epoch, xy_loss)
    ax1[0].plot(epoch, wh_loss)
    ax1[0].plot(epoch, conf_loss)
    ax1[0].plot(epoch, iou_loss)
    ax1[0].plot(epoch, total_loss)
    ax1[0].legend(['xy_loss','wh_loss','conf_loss','iou_loss','total_loss'], loc='upper left', fancybox=True, framealpha=1., shadow=True, borderpad=1)
    ax1[0].set_ylabel('Losses', fontweight='bold')
    # iou
    ax2_0 = ax1[0].twinx()
    ax2_0.plot(epoch, iou)
    ax2_0.legend(['mean_iou'], loc='upper right', fancybox=True, framealpha=1., shadow=True, borderpad=1)
    ax2_0.set_ylabel('IOU', fontweight='bold')
    # validation data
    ax1[1].set_title('Validation Losses and IOU', fontweight='bold')
    ax1[1].grid(linestyle='-', linewidth='0.2', color='gray')
    ax1[1].plot(epoch, val_xy_loss)
    ax1[1].plot(epoch, val_wh_loss)
    ax1[1].plot(epoch, val_conf_loss)
    ax1[1].plot(epoch, val_iou_loss)
    ax1[1].plot(epoch, val_total_loss)
    ax1[1].legend(['xy_loss','wh_loss','conf_loss','iou_loss','total_loss'], loc='upper left', fancybox=True, framealpha=1., shadow=True, borderpad=1)
    ax1[1].set_xlabel('Epochs', fontweight='bold')
    ax1[1].set_ylabel('Losses', fontweight='bold')
    # iou
    ax2_1 = ax1[1].twinx()
    ax2_1.plot(epoch, val_iou)
    ax2_1.legend(['mean_iou'], loc='upper right', fancybox=True, framealpha=1., shadow=True, borderpad=1)
    ax2_1.set_ylabel('IOU', fontweight='bold')
    # plot metrics and save
    plt.savefig(MODELS_FOLDER+'metrics.png')
    plt.show()

# main function
if __name__ == '__main__':
    plot_metrics()
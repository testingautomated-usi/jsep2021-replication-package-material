# Copyright 2021 Testing Automated @ Università della Svizzera italiana (USI)
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
from pathlib import Path

import tensorflow
from scipy.stats import stats

import utils
from config import Config
from utils import *

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))
    print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    cfg.SIMULATION_NAME = 'gauss-journal-track1-nominal'

    print("Script to compare offline vs online (within Udacity's) uncertainty values")

    # load the online uncertainty from csv
    path = os.path.join(cfg.TESTING_DATA_DIR,
                        cfg.SIMULATION_NAME,
                        'driving_log.csv')
    data_df = pd.read_csv(path)

    data_df = data_df[data_df["crashed"] == 0]

    online_uncertainty = data_df["uncertainty"]
    online_cte = abs(data_df["cte"])
    online_steering_angles = data_df["steering_angle"]
    center_images = data_df["center"]
    print("loaded %d images from file" % len(center_images))
    print("loaded %d steering_angle values" % len(online_steering_angles))
    print("loaded %d uncertainty values" % len(online_uncertainty))
    print("loaded %d CTE values" % len(online_cte))

    min_idx_unc = np.argmin(online_uncertainty)
    max_idx_unc = np.argmax(online_uncertainty)

    min_idx_cte = np.argmin(online_cte)
    max_idx_cte = np.argmax(online_cte)

    plt.figure(figsize=(80, 20))
    # display original
    ax = plt.subplot(1, 4, 1)
    plt.imshow(mpimg.imread(center_images[min_idx_unc]).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title("min uncertainty \n steering angle=%s \n uncertainty=%s" % (round(online_steering_angles[min_idx_unc], 5),
                                                                          online_uncertainty[min_idx_unc]),
              fontsize=50)

    # display reconstruction
    ax = plt.subplot(1, 4, 2)
    plt.imshow(mpimg.imread(center_images[max_idx_unc]).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title("max uncertainty \n steering angle=%s \n uncertainty=%s" % (round(online_steering_angles[max_idx_unc], 5),
                                                                          round(online_uncertainty[max_idx_unc], 20)),
              fontsize=50)

    ax = plt.subplot(1, 4, 3)
    plt.imshow(mpimg.imread(center_images[min_idx_cte]).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title("min CTE \n steering angle=%s \n CTE=%s" % (round(online_steering_angles[min_idx_cte], 5),
                                                          round(online_cte[min_idx_cte], 2)),
              fontsize=50)

    # display reconstruction
    ax = plt.subplot(1, 4, 4)
    plt.imshow(mpimg.imread(center_images[max_idx_cte]).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title("max CTE \n steering angle=%s \n CTE=%s" % (round(online_steering_angles[max_idx_cte], 5),
                                                          round(online_cte[max_idx_cte], 2)),
              fontsize=50)

    plt.savefig("plots/steering-uncertainty-cte.png")
    plt.show()
    plt.close()

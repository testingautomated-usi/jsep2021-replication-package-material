# Copyright 2021 Testing Automated @ Università della Svizzera italiana (USI)
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
import gc
import os

import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.model_selection import train_test_split

from selforacle import utils_vae
from config import Config
from utils import load_all_images
from utils import plot_reconstruction_losses, load_improvement_set
from selforacle.utils_vae import load_vae, load_data_for_vae_training
from vae_evaluate import load_or_compute_losses, get_threshold, get_scores
from selforacle.vae_train import train_vae_model


def evaluate_class_imbalance(cfg):
    # remove old files
    if os.path.exists('likely_false_positive_uncertainty.npy'):
        os.remove('likely_false_positive_uncertainty.npy')
    if os.path.exists('likely_false_positive_cte.npy'):
        os.remove('likely_false_positive_cte.npy')
    if os.path.exists('likely_false_positive_common.npy'):
        os.remove('likely_false_positive_common.npy')

    ''' 
        1. compute reconstruction error on nominal images
        and compute the likely false positives
    '''
    dataset = load_all_images(cfg)
    vae, name = load_vae(cfg, load_vae_from_disk=True)
    original_losses = load_or_compute_losses(vae, dataset, name, delete_cache=True)
    threshold_nominal = get_threshold(original_losses, conf_level=0.95)
    likely_fps_uncertainty, likely_fps_cte, _ = get_scores(cfg,
                                                           name,
                                                           original_losses,
                                                           original_losses,
                                                           threshold_nominal)

    assert len(likely_fps_uncertainty) > 0
    assert len(likely_fps_cte) > 0

    # save the likely false positive
    np.save('likely_false_positive_uncertainty.npy', likely_fps_uncertainty)
    np.save('likely_false_positive_cte.npy', likely_fps_cte)

    for mode in ['UNC', 'CTE']:

        if mode == 'UNC':
            lfps = likely_fps_uncertainty
        elif mode == 'CTE':
            lfps = likely_fps_cte

        ''' 
            2. compute improvement set
        '''
        x_train, x_test = load_data_for_vae_training(cfg, sampling=15)
        improvement_set = load_improvement_set(cfg, lfps)

        print("Old training data_nominal set: " + str(len(x_train)) + " elements")
        print("Improvement data_nominal set: " + str(len(improvement_set)) + " elements")

        initial_improvement_set = improvement_set

        for improvement_ratio in [2]:
            print("Using improvement ratio: " + str(improvement_ratio))
            for i in range(improvement_ratio - 1):
                temp = initial_improvement_set[:]
                improvement_set = np.concatenate((temp, improvement_set), axis=0)

            x_train_improvement_set, x_test_improvement_set = train_test_split(improvement_set,
                                                                               test_size=cfg.TEST_SIZE,
                                                                               random_state=0)

            x_train = np.concatenate((x_train, x_train_improvement_set), axis=0)
            x_test = np.concatenate((x_test, x_test_improvement_set), axis=0)

            print("New training data_nominal set: " + str(len(x_train)) + " elements")

            ''' 
                3. retrain using RDR's configuration
            '''
            weights = None

            newname = name + '-RDR-' + str(improvement_ratio) + "X-" + mode
            train_vae_model(cfg,
                            vae,
                            newname,
                            x_train,
                            x_test,
                            delete_model=True,
                            retraining=True,
                            sample_weights=weights)

            vae = utils_vae.load_vae_by_name(newname)

            path = os.path.join(cfg.TESTING_DATA_DIR,
                                cfg.SIMULATION_NAME,
                                'driving_log.csv')
            data_df = pd.read_csv(path)

            ''' 
                4. evaluate retrained model (RDR)  
            '''
            new_losses = load_or_compute_losses(vae, dataset, newname, delete_cache=True)
            plot_reconstruction_losses(original_losses, new_losses, newname, threshold_nominal, None, data_df)
            get_scores(cfg, newname, original_losses, new_losses, threshold_nominal)

        ''' 
            5. load data_nominal for retraining
        '''
        x_train, x_test = load_data_for_vae_training(cfg, sampling=1)
        improvement_set = load_improvement_set(cfg, lfps)

        print("Old training data_nominal set: " + str(len(x_train)) + " elements")
        print("Improvement data_nominal set: " + str(len(improvement_set)) + " elements")

        initial_improvement_set = improvement_set

        temp = initial_improvement_set[:]
        improvement_set = np.concatenate((temp, improvement_set), axis=0)

        x_train_improvement_set, x_test_improvement_set = train_test_split(improvement_set,
                                                                           test_size=cfg.TEST_SIZE,
                                                                           random_state=0)

        x_train = np.concatenate((x_train, x_train_improvement_set), axis=0)
        x_test = np.concatenate((x_test, x_test_improvement_set), axis=0)

        ''' 
            6. retrain using CWR's configuration
        '''
        # weights the frame using the reconstruction loss
        weights = np.array(original_losses)

        vae, name = load_vae(cfg, load_vae_from_disk=True)
        newname = name + '-CWR-' + mode
        train_vae_model(cfg,
                        vae,
                        newname,
                        x_train,
                        x_test,
                        delete_model=True,
                        retraining=True,
                        sample_weights=weights)

        vae = utils_vae.load_vae_by_name(newname)

        ''' 
            7. evaluate retrained (CWR) 
        '''
        new_losses = load_or_compute_losses(vae, dataset, newname, delete_cache=True)
        plot_reconstruction_losses(original_losses, new_losses, newname, threshold_nominal, None, data_df)
        get_scores(cfg, newname, original_losses, new_losses, threshold_nominal)

        # remove old files
        if os.path.exists('likely_false_positive_uncertainty.npy'):
            os.remove('likely_false_positive_uncertainty.npy')
        if os.path.exists('likely_false_positive_cte.npy'):
            os.remove('likely_false_positive_cte.npy')
        if os.path.exists('likely_false_positive_common.npy'):
            os.remove('likely_false_positive_common.npy')

        # del vae
        K.clear_session()
        gc.collect()


def main():
    os.chdir(os.getcwd().replace('script', ''))
    print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    evaluate_class_imbalance(cfg)


if __name__ == '__main__':
    main()

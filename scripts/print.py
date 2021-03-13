# Copyright 2021 Testing Automated @ Università della Svizzera italiana (USI)
# All rights reserved.
# This file is part of the project SelfOracle, a misbehaviour predictor for autonomous vehicles,
# developed within the ERC project PRECRIME
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.
import tensorflow
from scipy.stats import ttest_rel
from tqdm import tqdm

import utils
from config import Config
from utils import *
from selforacle.vae import normalize_and_reshape, VAE

WHAT = '-latent16-centerimg-nocrop'

if __name__ == '__main__':
    os.chdir(os.getcwd().replace('scripts', ''))
    print(os.getcwd())

    cfg = Config()
    cfg.from_pyfile("config_my.py")

    drive = utils.get_driving_styles(cfg)

    cfg.SIMULATION_NAME = 'gauss-journal-track3-nominal'
    data_nominal = load_all_images(cfg)

    encoder_mse = tensorflow.keras.models.load_model('sao/encoder-track3-MSEloss' + WHAT + '-CI-RETRAINED-2X-UNC')
    decoder_mse = tensorflow.keras.models.load_model('sao/decoder-track3-MSEloss' + WHAT + '-CI-RETRAINED-2X-UNC')
    vae_mse = VAE(model_name="encoder_mse",
                  loss="MSE",
                  latent_dim=cfg.SAO_LATENT_DIM,
                  encoder=encoder_mse,
                  decoder=decoder_mse)
    vae_mse.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001))

    encoder_vae = tensorflow.keras.models.load_model('sao/encoder-track3-VAEloss' + WHAT + '-CI-RETRAINED-2X-UNC')
    decoder_vae = tensorflow.keras.models.load_model('sao/decoder-track3-VAEloss' + WHAT + '-CI-RETRAINED-2X-UNC')
    vae_vae = VAE(model_name="encoder_vae",
                  loss="VAE",
                  latent_dim=cfg.SAO_LATENT_DIM,
                  encoder=encoder_vae,
                  decoder=decoder_vae)
    vae_vae.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001))

    i = 0
    list_original = []
    list_reconstructed_mse = []
    list_reconstructed_vae = []

    for x in tqdm(data_nominal):
        i += 1

        x = utils.resize(x)
        x = normalize_and_reshape(x)

        list_original.append(x.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
        # z_mean, z_log_var, z = vae_mse.encoder.predict(x)
        # decoded = vae_mse.decoder.predict(z)

        reconstructed_mse = vae_mse.predict(x)
        reconstructed_vae = vae_vae.predict(x)

        list_reconstructed_mse.append(
            reconstructed_mse.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
        list_reconstructed_vae.append(
            reconstructed_vae.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))

        # if i % 400 == 0:
        #     plt.figure(figsize=(80, 16))
        #     # display original
        #     ax = plt.subplot(1, 3, 1)
        #     plt.imshow(x.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
        #     ax.get_xaxis().set_visible(False)
        #     ax.get_yaxis().set_visible(False)
        #     plt.title("Original", fontsize=60)
        #
        #     # display reconstruction
        #     ax = plt.subplot(1, 3, 2)
        #     plt.imshow(reconstructed_mse.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
        #     ax.get_xaxis().set_visible(False)
        #     ax.get_yaxis().set_visible(False)
        #     plt.title("MSE: %.4f" % vae_mse.test_on_batch(x)[2], fontsize=60)
        #
        #     ax = plt.subplot(1, 3, 3)
        #     plt.imshow(reconstructed_vae.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
        #     ax.get_xaxis().set_visible(False)
        #     ax.get_yaxis().set_visible(False)
        #     plt.title("MSE: %.4f" % vae_vae.test_on_batch(x)[2], fontsize=60)
        #
        #     plt.savefig("plots/example-" + WHAT + "-" + str(i) + ".png")
        #     plt.show()
        #     plt.close()

    del data_nominal
    cfg.SIMULATION_NAME = 'gauss-journal-track3-rain'
    data_unseen = load_all_images(cfg)

    i = 0
    list_original_unseen = []
    list_reconstructed_mse_unseen = []
    list_reconstructed_vae_unseen = []

    for x in tqdm(data_unseen):
        i += 1

        x = utils.resize(x)
        x = normalize_and_reshape(x)

        list_original_unseen.append(x.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
        # z_mean, z_log_var, z = vae_mse.encoder.predict(x)
        # decoded = vae_mse.decoder.predict(z)

        reconstructed_mse = vae_mse.predict(x)
        reconstructed_vae = vae_vae.predict(x)

        list_reconstructed_mse_unseen.append(
            reconstructed_mse.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
        list_reconstructed_vae_unseen.append(
            reconstructed_vae.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))

        # if i % 100 == 0:
            # plt.figure(figsize=(80, 16))
            # # display original
            # ax = plt.subplot(1, 3, 1)
            # plt.imshow(x.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)
            # plt.title("Original", fontsize=60)
            #
            # # display reconstruction
            # ax = plt.subplot(1, 3, 2)
            # plt.imshow(reconstructed_mse.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)
            # plt.title("MSE: %.4f" % vae_mse.test_on_batch(x)[2], fontsize=60)
            #
            # ax = plt.subplot(1, 3, 3)
            # plt.imshow(reconstructed_vae.reshape(RESIZED_IMAGE_HEIGHT, RESIZED_IMAGE_WIDTH, IMAGE_CHANNELS))
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)
            # plt.title("MSE: %.4f" % vae_vae.test_on_batch(x)[2], fontsize=60)
            #
            # plt.savefig("plots/example-" + WHAT + "-" + str(i) + ".png")
            # plt.show()
            # plt.close()


    lpl_original = laplacian_variance(list_original)
    lpl_var_mse = laplacian_variance(list_reconstructed_mse)
    lpl_var_vae = laplacian_variance(list_reconstructed_vae)

    res = ttest_rel(lpl_var_mse, lpl_var_vae)
    print(f'T-score = {res.statistic:.2f}, p-value = {res.pvalue:.2f}')

    plt.hist(lpl_original, bins=50, alpha=0.2, label='nominal images')
    plt.hist(lpl_var_mse, bins=50, alpha=0.2, label='unseen images generated by VAE (MSE loss)')
    plt.hist(lpl_var_vae, bins=50, alpha=0.2, label='unseen images generated by VAE (VAE loss)')
    plt.legend(loc='upper right')
    plt.xlabel('Laplacian variance')
    plt.title(f'T-score = {res.statistic:.2f}, p-value = {res.pvalue:.2f}')
    plt.savefig("plots/laplacian-" + WHAT + ".png")
    plt.show()


    lpl_original = laplacian_variance(list_original)
    lpl_var_mse = laplacian_variance(list_reconstructed_mse_unseen)
    lpl_var_vae = laplacian_variance(list_reconstructed_vae_unseen)

    res = ttest_rel(lpl_var_mse, lpl_var_vae)
    print(f'T-score = {res.statistic:.2f}, p-value = {res.pvalue:.2f}')

    plt.hist(lpl_original, bins=50, alpha=0.2, label='nominal images')
    plt.hist(lpl_var_mse, bins=50, alpha=0.2, label='unseen images generated by VAE (MSE loss)')
    plt.hist(lpl_var_vae, bins=50, alpha=0.2, label='unseen images generated by VAE (VAE loss)')
    plt.legend(loc='upper right')
    plt.xlabel('Laplacian variance')
    plt.title(f'T-score = {res.statistic:.2f}, p-value = {res.pvalue:.2f}')
    plt.savefig("plots/laplacian-" + WHAT + ".png")
    plt.show()



import os
import pickle
from os.path import join, dirname, exists

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def savefig(fname, show_figure=True):
    if not exists(dirname(fname)):
        os.makedirs(dirname(fname))
    plt.tight_layout()
    plt.savefig(fname)
    if show_figure:
        plt.show()


def get_data_dir():
    return join('.', 'data')


def load_pickled_data(fname, include_labels=False):
    with open(fname, 'rb') as f:
        data = pickle.load(f)

    train_data, test_data = data['train'], data['test']
    if 'mnist.pkl' in fname or 'shapes.pkl' in fname:
        # Binarize MNIST and shapes dataset
        train_data = (train_data > 127.5).astype('uint8')
        test_data = (test_data > 127.5).astype('uint8')
    if 'celeb.pkl' in fname:
        train_data = train_data[:, :, :, [2, 1, 0]]
        test_data = test_data[:, :, :, [2, 1, 0]]
    if include_labels:
        return train_data, test_data, data['train_labels'], data['test_labels']
    return train_data, test_data


def show_samples(samples, fname=None, nrow=10, title='Samples'):
    samples = (torch.FloatTensor(samples) / 255).permute(0, 3, 1, 2)
    grid_img = make_grid(samples, nrow=nrow)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')

    if fname is not None:
        savefig(fname)
    else:
        plt.show()


def visualize_cifar10():
    data_dir = get_data_dir()
    train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))
    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs]
    show_samples(images, title='CIFAR10 Samples')


def save_training_plot(train_losses, test_losses, title, fname):
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_losses, label='train loss')
    plt.plot(x_test, test_losses, label='test loss')
    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('NLL')
    savefig(fname)


def save_results(dset_id, fn):
    assert dset_id in [1, 2]
    data_dir = get_data_dir()
    if dset_id == 1:
        train_data, test_data = load_pickled_data(join(data_dir, 'svhn.pkl'))
    else:
        train_data, test_data = load_pickled_data(join(data_dir, 'cifar10.pkl'))

    vqvae_train_losses, vqvae_test_losses, pixelcnn_train_losses, pixelcnn_test_losses, samples, reconstructions = fn(
        train_data, test_data, dset_id)
    samples, reconstructions = samples.astype('float32'), reconstructions.astype('float32')
    print(f'VQ-VAE Final Test Loss: {vqvae_test_losses[-1]:.4f}')
    print(f'PixelCNN Prior Final Test Loss: {pixelcnn_test_losses[-1]:.4f}')
    save_training_plot(vqvae_train_losses, vqvae_test_losses, f'Q2 Dataset {dset_id} VQ-VAE Train Plot',
                       f'results/q2_dset{dset_id}_vqvae_train_plot.png')
    save_training_plot(pixelcnn_train_losses, pixelcnn_test_losses, f'Q3 Dataset {dset_id} PixelCNN Prior Train Plot',
                       f'results/q2_dset{dset_id}_pixelcnn_train_plot.png')
    show_samples(samples, title=f'Q2 Dataset {dset_id} Samples',
                 fname=f'results/q2_dset{dset_id}_samples.png')
    show_samples(reconstructions, title=f'Q3 Dataset {dset_id} Reconstructions',
                 fname=f'results/q2_dset{dset_id}_reconstructions.png')

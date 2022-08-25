import numpy as np
from tensorflow import one_hot

fft_size = 256
width = 48
num_channels = 3
num_classes = 26


def extract_features(img):
    feats = img.reshape(img.shape[0], fft_size, width, num_channels).astype('>f4')
    return feats


def restructure_spectrogram(spectrogram):
    example_spectrogram = spectrogram.flatten()
    res = np.reshape(example_spectrogram, (48, 256)).swapaxes(0, 1)
    return res


def restructure_datapoints(spectrograms):
    return np.array([restructure_spectrogram(spectrogram) for spectrogram in spectrograms])


def extract_features_restructured(dataset):
    """
    My own implementation, flattens the spectrograms first and changes the structure of the spectrograms.

    :param img:
    :return:
    """
    a = np.array([restructure_datapoints(datapoint) for datapoint in dataset])
    a = np.moveaxis(a, 1, -1)
    return a


def encode_labels(y):
    return one_hot(y, num_classes)

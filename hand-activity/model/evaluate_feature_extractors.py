from tensorflow.keras.models import load_model
from preprocessing import extract_features_restructured
from svm import get_feature_extractor, extract_features_for_svm
import numpy as np
import tensorflow as tf

# model to be used for feature extraction
model_path = "/mnt/data/hand-activity-data/models/extraction_models/"
data_path = '/mnt/data/hand-activity-data/'
data_path_y_split = '/mnt/data/hand-activity-data/split/'


def load_data_user(round, user):
    x = np.load(data_path + 'split/round{}_user{}_X.npy'.format(round, user))
    return x

accuracies = []
for user in range(1, 13):
    print(f'Starting extraction for feature_extractor {user}')
    model = load_model(f'{model_path}model_user_{user}.h5')
    x = load_data_user(3, user)
    y = np.load(data_path_y_split + f'round{3}_user{user}_Y.npy')

    x = extract_features_restructured(x)
    y = tf.one_hot(y, 26)

    _, accuracy = model.evaluate(x, y)
    accuracies.append(accuracy)

print(accuracies)
accuracy_dnn = np.average(accuracies)
sd_dnn = np.std(accuracies)
print(f'Accuracy: {accuracy_dnn}; SD: {sd_dnn}')

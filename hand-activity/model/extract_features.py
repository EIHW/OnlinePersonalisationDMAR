from tensorflow.keras.models import load_model
from preprocessing import extract_features_restructured
from svm import get_feature_extractor, extract_features_for_svm
import numpy as np

# model to be used for feature extraction
# model_path = "/mnt/data/hand-activity-data/extraction_models/"
model_path = "/mnt/data/hand-activity-data/models/"
data_path = '/mnt/data/hand-activity-data/'


def load_data_user(round, user):
    x_eval = np.load(data_path + 'split/round{}_user{}_X.npy'.format(round, user))
    return x_eval


for n_feature_extractor in range(1, 13):
    print(f'Starting extraction for feature_extractor {n_feature_extractor}')
    model = load_model(f'{model_path}model_user{n_feature_extractor}.h5')
    feature_extractor = get_feature_extractor(model)
    for round in range(1, 5):
        for user in range(1, 13):
            print(f'Extracting round {round} of user {user}.')
            x_train = load_data_user(round, user)

            if len(x_train) == 0:
                train_features_for_svm = []
            else:
                x_train = extract_features_restructured(x_train)
                train_features_for_svm = extract_features_for_svm(feature_extractor, x_train)
            path = f'./extracted_features_for_user{n_feature_extractor}/svm_features_round{round}_user{user}'
            np.save(path, train_features_for_svm)
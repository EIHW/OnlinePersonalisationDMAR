from tensorflow.keras.models import load_model
from svm import get_feature_extractor, extract_features_for_svm
import numpy as np
from load_data import leave_user_out_data

# model to be used for feature extraction
# model_path = "/mnt/data/hand-activity-data/extraction_models/"
model_path = "/mnt/data/hand-activity-data/models/"
data_path = '/mnt/data/hand-activity-data/'

model = load_model('model.h5')
feature_extractor = get_feature_extractor(model)
x_train, y_train, x_test, y_test = leave_user_out_data(5)


x_train = x_train.swapaxes(1, 2)
x_train = np.nan_to_num(x_train)
x_test = x_test.swapaxes(1, 2)
x_test = np.nan_to_num(x_test)
model.evaluate(x_test, y_test)

train_features_for_svm = extract_features_for_svm(feature_extractor, x_train)
test_features_for_svm = extract_features_for_svm(feature_extractor, x_test)

path = './train2.npy'
np.save(path, train_features_for_svm)

path = './test2.npy'
np.save(path, test_features_for_svm)

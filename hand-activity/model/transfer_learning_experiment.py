from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import extract_features_restructured, encode_labels

model_path = "data/hand-activity-data/models/model_user1.h5"
# model_path = "/home/sysgen/workspace/hand-activity-data/models/model_user1.h5"

data_path = 'data/hand-activity-data/users/features/'
data_path_y = 'data/hand-activity-data/users/features/'

data_path_split = 'data/hand-activity-data/split/'
data_path_y_split = 'data/hand-activity-data/split/'


def load_test_data(user):
    test_rounds = [4]  # , 4]
    online_round = 3

    x_test = np.concatenate([np.load(data_path_split + 'round{}_user{}_X.npy'.format(r, user)) for r in test_rounds])
    y_test = np.concatenate([np.load(data_path_y_split + 'round{}_user{}_Y.npy'.format(r, user)) for r in test_rounds])

    x_online = np.load(data_path_split + 'round{}_user{}_X.npy'.format(online_round, user))
    y_online = np.load(data_path_y_split + 'round{}_user{}_Y.npy'.format(online_round, user))

    online_learning_perc = 0.4
    x_online, _, y_online, _ = train_test_split(x_online, y_online, test_size=1 - online_learning_perc, random_state=2, stratify=y_online)

    return x_online, y_online, x_test, y_test


x_online, y_online, x_test, y_test = load_test_data(1)
x_online = extract_features_restructured(x_online)
x_test = extract_features_restructured(x_test)
y_online = encode_labels(y_online)
y_test = encode_labels(y_test)
print('x_online: {}'.format(x_online.shape))
print('x_test: {}'.format(x_test.shape))

model = load_model(model_path)
for layer in model.layers:
    if layer.name == 'dense_3':
        layer.trainable = True
    else:
        layer.trainable = False
model.summary()

# model.evaluate(x=x_test, y=y_test)

H = model.fit(
    x=x_online,
    y=y_online,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=128
)

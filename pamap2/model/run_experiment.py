from load_data import leave_user_out_data
from get_model import load_model
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import save_model

x_train, y_train, x_test, y_test = leave_user_out_data(5)

x_train = x_train.swapaxes(1, 2)
x_train = np.nan_to_num(x_train)
x_test = x_test.swapaxes(1, 2)
x_test = np.nan_to_num(x_test)

y_train = tf.one_hot(y_train, 12)
y_test = tf.one_hot(y_test, 12)

model = load_model()

H = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=128,
)

save_model(model, 'pamap2/model/model_tmp.h5', save_format='h5')

# np.set_printoptions(threshold=sys.maxsize)
#
# labels = model.predict(x_train)
# print(np.argmax(labels, axis=1))
# labels = model.predict(x_test)
# print(np.argmax(labels, axis=1))

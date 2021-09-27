import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 載入 MNIST 手寫數字資料

# 將寬高各 28 個像素的圖壓縮成一維陣列
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

print(y_train[0])

# One-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train[0])

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(y_train[:10])
# X = x_train[0, :, :]
# X = X.reshape(28, 28)
# plt.imshow(X, cmap="gray")
# plt.show()

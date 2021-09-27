import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 載入 MNIST 手寫數字資料

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(y_train[:10])

X = x_train[0, :, :]
X = X.reshape(28, 28)
plt.imshow(X, cmap="gray")
plt.show()

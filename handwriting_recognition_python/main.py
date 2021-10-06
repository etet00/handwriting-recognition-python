from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 載入 MNIST 手寫數字資料

# 將寬高各 28 個像素的圖壓縮成一維陣列
x_train_flat = x_train.reshape(60000, 784)
x_test_flat = x_test.reshape(10000, 784)

# One-hot encoding
y_train_encoding = to_categorical(y_train)
y_test_encoding = to_categorical(y_test)

# 建立模型，使用兩層隱藏層
neuron_num = 500
model = Sequential()    # 標準一層一層傳遞的神經網路成為 Sequential
model.add(Dense(units=neuron_num, input_dim=784, activation="sigmoid"))     # neuron_num 表示此層使用多少的神經元
model.add(Dense(units=neuron_num, activation="sigmoid"))
model.add(Dense(units=10, activation="softmax"))    # softmax 使輸出結果總和為 1

# 需要進行 compile 把神經網路建立好
# 需設定參數，損失函數(loss function)、優化器(optimizer)及效能衡量指標(metrics)的類別
model.compile(loss="mse", optimizer=SGD(learning_rate=0.087), metrics=["accuracy"])
model.summary()

# 執行模型訓練
history = model.fit(x_train_flat, y_train_encoding, batch_size=100, epochs=20)

# 印出最終訓練準確率
scores = model.evaluate(x_test_flat, y_test_encoding)
print("準確率", scores[1])

model.save("handwriting_model.h5")
model.save_weights("handwriting_weight.h5")

plt.figure(figsize=(8, 8))
plt.plot(history.history["accuracy"], "r", label="訓練準確率")
# plt.show()
plt.savefig("accuracy.pdf")

# 預測
predictions = model.predict_classes(x_test_flat)


# model.fit(x_train, y_train, batch_size=10, validation_split=0.2, epochs=20)

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(y_train[:10])
# X = x_train[0, :, :]
# X = X.reshape(28, 28)
# plt.imshow(X, cmap="gray")
# plt.show()

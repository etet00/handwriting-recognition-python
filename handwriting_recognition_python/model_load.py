import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from show_image import show_image


def probability2num(predictions):
    pre_out = []
    for prediction in predictions:
        pre_out.append(np.argmax(prediction))
    return pre_out


# mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 載入 MNIST 手寫數字資料

# 將寬高各 28 個像素的圖壓縮成一維陣列
x_train_flat = x_train.reshape(60000, 784)
x_test_flat = x_test.reshape(10000, 784)

# One-hot encoding
y_train_encoding = to_categorical(y_train)
y_test_encoding = to_categorical(y_test)

# 載入訓練完畢之模型
model = load_model("handwriting_model.h5")

# 預測結果
predictions = model.predict(x_test_flat)
predictions = probability2num(predictions)

# 顯示結果
show_image(x_test, y_test, predictions, start_id=9, num=10)

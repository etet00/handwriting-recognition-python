import numpy as np
import os
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from show_image import show_image
from probability2num import probability2num

test_feature = []
test_label = []

# 讀取自行準備之圖檔
file_path = os.path.join(".", "figs")
file_list = os.listdir(path=file_path)

for file in file_list:
    test_label.append(int(file.strip(".jpg")))
    fig = cv2.imread(os.path.join(file_path, file))
    fig = cv2.resize(fig, dsize=(28, 28))
    fig = cv2.cvtColor(fig, cv2.COLOR_BGR2GRAY)     # 將圖片轉成灰階
    test_feature.append(fig)

num = len(test_label)

# 將清單傳換成陣列
test_feature = np.abs(255 - np.array(test_feature))  # 順便執行顏色反轉
test_label = np.array(test_label)

# 將寬高各 28 個像素的圖壓縮成一維陣列
test_feature_flat = test_feature.reshape(num, 784)

# One-hot encoding
test_label_encoding = to_categorical(test_label)

# 載入訓練完畢之模型
model = load_model("handwriting_model.h5")

# 預測結果
predictions = model.predict(test_feature_flat)
predictions = probability2num(predictions)

# 顯示結果
show_image(test_feature, test_label, predictions, start_id=0, num=num)

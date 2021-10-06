import matplotlib.pyplot as plt


def show_image(images, labels, predictions=None, start_id=0, num=1):
    plt.gcf().set_size_inches(8, 5)     # 設定圖片大小
    if num > 10:                    # 最多顯示 20 張圖片
        num = 10
    for i in range(num):
        ax = plt.subplot(2, 5, i+1)
        ax.imshow(images[start_id], cmap="gray")

        # 如果有模型預測結果，在標題上面顯示預測以及真實數值
        if predictions:
            title = f"ai = {str(predictions[start_id])}"
            if predictions[start_id] == labels[start_id]:
                title += "(o)"
            else:
                title += "(x)"
            title += f"\nlabel = {str(labels[start_id])}"
        # 無預測結果，僅顯示真實數值
        else:
            title = f"label = {str(labels[start_id])}"

        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        start_id += 1
    plt.show()

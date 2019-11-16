# 调用模型预测结果,
import os
import cv2
import numpy as np

from PIL import Image
from sklearn.externals import joblib
from image_split import ImagePreprocess


class PredictImage:
    def __init__(self, name):
        self.name = name

    def transform(self):
        image_path = f'jupyter_project/ML/04项目实战/T01验证码识别/datas/{self.name}.jpg'
        image = cv2.imread(image_path)    # rgb方式读取图片
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)  # 灰度处理
        self.x = [np.array(img.reshape(-1)) / 255]

    def predict(self):
        # 模型加载
        model_path = f'jupyter_project/ML/04项目实战/T01验证码识别/SVM/clf_{self.name}.model'
        clf = joblib.load(model_path)
        return clf.predict(self.x)[0]


def calculate(rlt):
    """
    计算识别结果
    return: int
    """
    if rlt[1] == 1:
        return int(rlt[0]) + int(rlt[2])
    else:
        return int(rlt[0]) - int(rlt[2])


def main(name):
    # 图片分割
    image_path = 'jupyter_project/ML/04项目实战/T01验证码识别/datas/test_images'
    save_path = 'jupyter_project/ML/04项目实战/T01验证码识别/datas/'

    # 实例化图片处理对象
    image_pre = ImagePreprocess()
    image = Image.open(os.path.join(image_path, name))
    two_image = image_pre.two_values(image)     # 二值化处理
    no_noise_image = image_pre.reduce_noise(two_image)     # 领域将噪

    # 分割图片
    image_pre.split_image(no_noise_image, save_path)
    print(f'{name}分割完成。')

    # 加载模型预测结果
    result = []
    for i in range(3):
        image_predict = PredictImage(str(i + 1))
        image_predict.transform()
        result.append(image_predict.predict())
    print('验证码计算结果：\n', calculate(result))


if __name__ == '__main__':
    for i in range(900, 915):
        main(f'login{i}.jpg')

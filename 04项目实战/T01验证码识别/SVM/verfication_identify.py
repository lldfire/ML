import cv2
import numpy as np

from PIL import Image
from sklearn.externals import joblib    # 模型保存与导入


class Verfication:
    """验证码识别，验证码的预处理，识别结果计算"""
    def __init__(self):
        pass

    def two_values(self, image, threshold):
        """
        采用阈值分割法进行二值化处理
            image: image object
            threshold: 二值化阈值，默认140
        return: 二值化后的图片
        """
        image_gray = image.convert('L')      # 灰度处理
        pixdata = image_gray.load()
        width, hight = image_gray.size       # 图片的尺寸（像素点）

        # 遍历像素点，将大于阈值的设为白色，小于阈值的设黑色
        for x in range(width):
            for y in range(hight):
                if pixdata[x, y] < threshold:
                    pixdata[x, y] = 0
                else:
                    pixdata[x, y] = 255

        return image_gray

    def reduce_noise(self, image_gray, threshold=185, n=6):
        """
        采用8邻域降噪法对二值化处理后的图片进行降噪
            image: image object
            threshold: 像素阈值，默认185
            n: 满足阈值的数量
        return: image object
        """
        pix = image_gray.load()
        width, hight = image_gray.size
        for i in range(10):
            for x in range(1, width-1):
                for y in range(1, hight-1):
                    count = 0
                    if pix[x, y-1] > threshold:   # 上
                        count += 1
                    if pix[x, y+1] > threshold:   # 下
                        count += 1
                    if pix[x-1, y] > threshold:   # 左
                        count += 1
                    if pix[x+1, y] > threshold:   # 右
                        count += 1
                    if pix[x-1, y-1] > threshold:   # 左上
                        count += 1
                    if pix[x-1, y+1] > threshold:   # 左下
                        count += 1
                    if pix[x+1, y-1] > threshold:   # 右上
                        count += 1
                    if pix[x+1, y+1] > threshold:   # 右下
                        count += 1
                    if count > n:     # 如果某个像素点周围的的像素点为白色，则认为改像素点为噪点
                        pix[x, y] = 255
        return image_gray

    def split_image(self, image_gray):
        """
        将灰度和二值化后的图片进行分割，并将结果返回以供识别
            image: image object
            name: 图片名
        return: None
        """
        image_gray.crop([2, 1, 22, 19]).save('./image/1.jpg')
        image_gray.crop([31, 5, 42, 17]).save('./image/2.jpg')
        image_gray.crop([42, 2, 56, 17]).save('./image/3.jpg')

    def conversion_image(self, path):
        """
        读取图片并将图片信息转化为一维数组
        return: x, 特征属性
        """
        image = cv2.imread(path)    # RGB3维 数据
        image_g = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)  # 灰度处理
        return [image_g.reshape(-1)]     # 将数据转化为一维, 特征

    def calculate(self, rlt):
        """
        计算识别结果
        return: int
        """
        if rlt[1] == '+':
            return int(rlt[0]) + int(rlt[2])
        else:
            return int(rlt[0]) - int(rlt[2])

    def predict(self, path, threshold=185, noise=False):
        """
        调用模型，预测结果，path: 图片临时存储位置
        return: 计算结果
        """
        result = []
        image = Image.open(path)   # 读取图片文件
        t_image = self.two_values(image, threshold)   # 二值化处理

        if noise:
            t_image = self.reduce_noise(t_image, threshold, 7)   # 降噪处理
        self.split_image(t_image)     # 分割图片并保存

        for idx in [1, 2, 3]:
            x_test = np.array(self.conversion_image('./image/%d.jpg' % idx))
            clf = joblib.load('./models/clf_%d.model' % idx)

            result.append(clf.predict(x_test)[0])

        return self.calculate(result)

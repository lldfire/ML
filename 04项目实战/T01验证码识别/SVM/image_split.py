# 验证码二值化、降噪
import os
from PIL import Image


class ReadShowImage():
    """
    读取、查看图片
    """

    def __init__(self,):
        pass

    def read_image(self):
        pass


class ImagePreprocess:
    """
    图片的预处理，主要是灰度处理和二值化
    threshold: 二值化和降噪处理时的阈值
    """
    
    def __init__(self, threshold=120):
        self.threshold = threshold

    def two_values(self, image):
        """
        采用阈值分割法进行二值化处理
            image: image object
        return: 二值化后的图片
        """
        image_gray = image.convert('L')      # 灰度处理
        pixdata = image_gray.load()
        width, hight = image_gray.size       # 图片的尺寸（像素点）
        # 遍历像素点，将大于阈值的设为白色，小于阈值的设黑色
        for x in range(width):
            for y in range(hight):
                if pixdata[x, y] < self.threshold:
                    pixdata[x, y] = 0
                else:
                    pixdata[x, y] = 255
        return image_gray

    def reduce_noise(self, image, n=8):
        """
        采用8邻域降噪法对图片进行降噪
            image: image object
            threshold: 像素阈值，默认185
            n: 满足阈值的数量
        return: iamge object
        """
        image_gray = image.convert('L')
        pix = image_gray.load()
        width, hight = image.size
        for x in range(1, width-1):
            for y in range(1, hight-1):
                count = 0
                if pix[x, y-1] > self.threshold:   # 上
                    count += 1
                if pix[x, y+1] > self.threshold:   # 下
                    count += 1
                if pix[x-1, y] > self.threshold:   # 左
                    count += 1
                if pix[x+1, y] > self.threshold:   # 右
                    count += 1
                if pix[x-1, y-1] > self.threshold:   # 左上
                    count += 1
                if pix[x-1, y+1] > self.threshold:   # 左下
                    count += 1
                if pix[x+1, y-1] > self.threshold:   # 右上
                    count += 1
                if pix[x+1, y+1] > self.threshold:   # 右下
                    count += 1
                if count > n:     # 如果某个像素点周围的的像素点为白色，则认为改像素点为噪点
                    pix[x, y] = 255
        return image_gray

    def split_image(self, image, path):
        """
        将灰度和二值化后的图片进行分割，并将结果返回以供识别
        return: None
        """
        split_point = [[2, 1, 22, 19], [31, 5, 42, 17], [42, 2, 56, 17]]
        for i, point in enumerate(split_point):
            path_ = os.path.join(path, f'{i + 1}.jpg')
            image.crop(point).save(path_)

    def split_save_image(self, image, idx, name, path):
        """
        将灰度和二值化后的图片进行分割，并将分割结果分类保存
            image: image object
            name: 图片标签值
            idx: 样本数量
        """
        # path = 'jupyter_project/ML/04项目实战/T01验证码识别/datas/split_images'
        split_point = [[2, 1, 22, 19], [31, 5, 42, 17], [42, 2, 56, 17]]
        np = [(0, 2), (2, 3), (3, 4)]
        for i, point in enumerate(split_point):
            # 存储路径和名称
            path_ = path + f'/{i + 1}'
            name_ = f'{idx}_1_{name[np[i][0]:np[i][1]]}.jpg'
            now_path = os.path.join(path_, name_)
            
            # 分割图片，分别保存至对应目录下
            image.crop(point).save(now_path)


def main():
    image_path = 'jupyter_project/ML/04项目实战/T01验证码识别/datas/train_images'
    save_path = 'jupyter_project/ML/04项目实战/T01验证码识别/datas/split_images'
    image_list = os.listdir(image_path)
    # image = Image.open(os.path.join(image_path, image_list[0]))

    # 批量分割图片
    # 实例化图片处理对象
    image_pre = ImagePreprocess()
    for idx, name in enumerate(image_list):
        image = Image.open(os.path.join(image_path, name))
        two_image = image_pre.two_values(image)     # 二值化处理
        no_noise_image = image_pre.reduce_noise(two_image)     # 领域将噪

        # 分割图片
        image_pre.split_save_image(no_noise_image, idx, name, save_path)

        if idx % 1000 == 0:
            print(f'已处理完层{idx+1} 张验证码')
    else:
        print('验证码分割完成！')

    # path = 'jupyter_project/ML/04项目实战/T01验证码识别/datas'
    # image_pre.split_image(no_noise_image, path)

if __name__ == '__main__':
    main()

# 创建高斯朴素贝叶斯分类器
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


class GaussianNB:
    """
    高斯朴素贝叶斯分类器
    """

    def __init__(self):
        self.n_class = None    # 类别数量
        self.avgs = None       # 每个类别，每个特征的平均值
        self.vars = None       # 每个类别，每个特征的方差
        self.prior = None      # 先验概率

    def _get_prior(self, targets):
        """
        计算先验概率
        targets: 样本的标签
        """
        target_size = len(targets)     # 样本标签数量
        target_count = Counter(targets)
        # 计算每个类别的先验概率
        prior = np.array(
            [target_count[i] / target_size for i in target_count]
        )
        return prior

    def _get_avgs(self, data, target):
        """
        计算训练样本均值，每个类别中的每个特征分别计算
        target == i 找到每个类别标签的索引，取出对应的特征数据
        """
        return np.array(
            [data[target == i].mean(axis=0) for i in range(self.n_class)]
        )

    def _get_vars(self, data, target):
        """计算训练样本方差，每个类别中的每个特征的方差"""
        return np.array(
            [data[target == i].var(axis=0) for i in range(self.n_class)]
        )

    def _get_factor(self, row):
        """根据高斯公式计算似然概率值"""
        return ((1 / (self.vars * np.sqrt(2 * np.pi))) * np.exp(
            (-(row - self.avgs) ** 2) / (2 * self.vars ** 2)
        )).prod(axis=1)

    def fit(self, data: np.array, target: np.array):
        self.prior = self._get_prior(target)
        self.n_class = len(self.prior)
        self.avgs = self._get_avgs(data, target)
        self.vars = self._get_vars(data, target)

    def prodict_proba(self, data):
        """先验概率乘以调整因子，得到后验概率"""
        factors = np.apply_along_axis(
            lambda x: self._get_factor(x), axis=1, arr=data)
        probs = self.prior * factors

        # 归一化
        probs_sum = probs.sum(axis=1)
        return probs / probs_sum[:, None]

    def prodict(self, data):
        return self.prodict_proba(data).argmax(axis=1)


# 效果评估
def main():
    print('使用鸢尾花⚜️数据集测试分类器')
    iris = load_iris()
    x = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=9)

    print('构建模型')
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    prodict = clf.prodict(x_test)
    # print(np.around(clf.prodict_proba(x_test), 4))
    print(prodict)
    print(y_test)
    print(f'模型准确率{accuracy_score(prodict, y_test) * 100:.2f}%')


if __name__ == '__main__':
    main()

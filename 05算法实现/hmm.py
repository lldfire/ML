import numpy as np


# 创建HMM 模型
class HMM:
    def __init__(self, Ann, Bnm, pi, O):
        self.A = np.array(Ann, np.float)    # 状态概率转移矩阵
        self.B = np.array(Bnm, np.float)    # 观测状态转移概率矩阵
        self.Pi = np.array(pi, np.float)    # 初始状态概率矩阵
        self.O = np.array(O, np.int)    # 观测序列 二维矩阵
        self.N = self.A.shape[0]    # 状态的数量
#         self.M = self.O.shape[1]    # 观测矩阵长度

    def forward(self):
        """前向算法"""
        T = len(self.O)
        alpha = np.zeros((T, self.N), np.float)

        # 计算初值，每个初始状态*由此状态到观测值的转移概率
        for i in range(self.N):
            alpha[0, i] = self.Pi[i] * self.B[i, self.O[0]]

        # 递归计算
        for t in range(T-1):
            # t 时刻 所有可能的状态序列到 t+1时刻的状态的概率
            for i in range(self.N):
                summation = 0
                for j in range(self.N):
                    # t时刻 由状态j 变为状态 i 的概率
                    summation += alpha[t, j] * self.A[j, i]
                # t+1时刻状态i的概率等于 所有可能的状态组合
                alpha[t+1, i] = summation * self.B[i, self.O[t+1]]
        # 终值,alpha 矩阵中的最后一行
        Polambda = np.sum(alpha[T-1])
        return Polambda, alpha

    def backword(self):
        T = len(self.O)
        beta = np.zeros((T, self.N), np.float)
        for i in range(self.N):
            beta[T-1, i] = 1.0     # 矩阵中最后一行的值

        # 从后向遍历,每个时刻
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                summation = 0.0
                for j in range(self.N):
                    summation += self.A[i, j] * self.B[j, self.O[t+1]] \
                        * beta[t+1, j]
                beta[t, i] = summation
        proba = 0.0
        for i in range(self.N):
            proba += self.Pi[i] * self.B[i, self.O[0]] * beta[0, i]

        return proba, beta

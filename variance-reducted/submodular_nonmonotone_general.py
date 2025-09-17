import numpy as np
import math
import cvxopt
import matplotlib.pyplot as plt
from matplotlib import font_manager


# the first experiment about the offline QP
def loadtxtmethod(filename):
    data = np.loadtxt(filename, dtype=int, delimiter=None, skiprows=2)
    return data


class Submodular_ex:
    noise = None

    def __init__(self, data):
        self.p = p
        self.M = np.zeros((n, n))
        self.data = data

    def generate(self, data):
        m = np.array(self.data).shape[0]  # m=数据的行数
        for i in range(m):
            self.M[np.array(self.data)[i, 0] - 1][np.array(self.data)[i, 1] - 1] = 100 * np.array(self.data)[i][2]
        # M[np.array(data)[0, 0] - 1][np.array(data)[0, 1] - 1] = np.array(data)[0][2]
        # M[np.array(data)[1, 0] - 1][np.array(data)[1, 1] - 1] = np.array(data)[1][2]
        self.M = (self.M + self.M.T)

    def exact_value_oracle(self, x):
        y = 0
        for i in range(n):
            for j in range(n):
                if j < i or j > i:
                    y = y + self.M[i][j] * (1 - (1 - self.p) ** x[i]) * (1 - self.p) ** x[j]
        return y

    def stochastic_gradient_oracle(self, x):
        # return a stochastic gradient g=\\\\nable f(x)+\\\\deta*U[-1,1] for E(g)=\\\\nable f(x)=Hx+h\\n\",\n",
        noise = Submodular_ex.noise * np.random.randn(n, 1)  # 读取矩阵H的行数（shape[0]）,随机生成n*1的（高斯噪声）向量
        grad = [0] * self.M.shape[0]
        for t in range(n):
            for j in range(n):
                if j < t or j > t:
                    grad[t] = grad[t] - 2 * self.M[t][j] * (1 - self.p) ** x[j] * (1 - self.p) ** x[t] * math.log(
                        1 - self.p) + \
                              self.M[j][t] * (1 - self.p) ** x[t] * math.log(1 - self.p)
        return grad + noise

    @staticmethod
    def set_noise(delta):
        Submodular_ex.noise = delta  # 给噪声常数

    # @staticmethod
    # def set_demension(data):
    #         Submodular_ex.noise = delta  # 给噪声常数
    #         r = np.array(data).shape[1]
    #         n = np.max(np.array(data)[:, 0:(r - 1)])


# # first method: traditional gradient ascent
# def projected_gradient_ascent(sub: Submodular_ex, T, batch):
#     np.random.seed(T)
#     outcome = [0] * T
#     point = np.zeros((n, 1))  # 生成n行1列的矩阵
#     # setting the projection QP as 1/2x^{T}Px+q^{T}x s.t Gx \\\\le g, Cx=c\\n\",\n",
#     G = cvxopt.matrix(np.concatenate([np.ones((1, n)), -np.ones((1, n)), -np.eye(n)]))  # 拼接三个矩阵的，按行拼接 np.eye(n)生成n维单位向量
#     g = cvxopt.matrix(np.concatenate([np.array([[1], [-0.25]]), np.zeros((n, 1))]))  # 对应三个约束 Ax\leb,x\le1,-x\le0
#     P = cvxopt.matrix(2 * np.eye(n))  # 构造目标函数系数，\|x-point\|^{2}~ x^{2}-2point*x
#     for i in range(T):
#         gradient = sum(sub.stochastic_gradient_oracle(point) for j in range(batch)) / batch  # 小批量多次采样求平均
#         point += gradient / math.sqrt(i + 1)  # +=: point=point+gradient...
#         q = cvxopt.matrix(-2 * point)  # 构造目标函数终一次项系数 -2point
#         sol = cvxopt.solvers.qp(P, q, G, g)  # 求解二次规划问题：投影步
#         point = np.array(sol['x'])
#         point = point.reshape(n, 1)
#         outcome[i] = sub.exact_value_oracle(point)
#     return outcome


# the method: frankwolfe
def fw(H, T):
    # global point1
    np.random.seed(k)
    # m = Submodular_QP.constraints.shape[0]
    # n = Submodular_QP.constraints.shape[1]
    outcome = [0] * (T)
    point = np.zeros((n, 1))
    G = cvxopt.matrix(np.concatenate([np.ones((1, n)), -np.ones((1, n)), -np.eye(n)]))  # 拼接三个矩阵的，按行拼接 np.eye(n)生成n维单位向量
    g = cvxopt.matrix(np.concatenate([np.array([[1], [-0.25]]), np.zeros((n, 1))]))  # 对应三个约束 Ax\leb,x\le1,-x\le0
    for i in range(T):
        outcome[i] = H.exact_value_oracle(point)
        gradient = H.stochastic_gradient_oracle(point)
        sol = cvxopt.solvers.lp(cvxopt.matrix(-gradient), G, g)
        alpha = (math.log(3) / 2) / ((i + 1) * (sum(1 / (t + 1) for t in range(T))))
        point = (1 - alpha) * point + alpha * np.array(sol['x'])
    return outcome


#
#
# #
# #
# the method: fw2 of du
def fw2(H, T=500):
    np.random.seed(k)
    # m = Submodular_QP.constraints.shape[0]
    # n = Submodular_QP.constraints.shape[1]
    outcome = [0] * (T)
    point = np.zeros((n, 1))
    G = cvxopt.matrix(np.concatenate([np.ones((1, n)), -np.ones((1, n)), -np.eye(n)]))  # 拼接三个矩阵的，按行拼接 np.eye(n)生成n维单位向量
    g = cvxopt.matrix(np.concatenate([np.array([[1], [-0.25]]), np.zeros((n, 1))]))  # 对应三个约束 Ax\leb,x\le1,-x\le0
    for i in range(T):
        outcome[i] = H.exact_value_oracle(point)
        gradient = H.stochastic_gradient_oracle(point)
        sol = cvxopt.solvers.lp(cvxopt.matrix(-gradient), G, g)
        alpha = ((1 / T) * (1 + i / T)) / ((1 + (i+1) / T) ** 2)
        point = (1 - alpha) * point + alpha * np.array(sol['x'])
    return outcome


#
#
# our method: SPIDER_SFW
def spider_stochastic_fw(sub: Submodular_ex, T, batch):
    np.random.seed(k)
    # m = Submodular_QP.constraints.shape[0]
    # n = Submodular_QP.constraints.shape[1]
    outcome = [0] * (T)
    point = np.zeros((n, 1))
    # point[0][0] = 1
    G = cvxopt.matrix(np.concatenate([np.ones((1, n)), -np.ones((1, n)), -np.eye(n)]))  # 拼接三个矩阵的，按行拼接 np.eye(n)生成n维单位向量
    g = cvxopt.matrix(np.concatenate([np.array([[1], [-0.25]]), np.zeros((n, 1))]))  # 对应三个约束 Ax\leb,x\le1,-x\le0
    g_t = sum(sub.stochastic_gradient_oracle(point) for j in range(epoch ** 2)) / (epoch ** 2)
    for i in range(T):
        outcome[i] = sub.exact_value_oracle(point)
        sol = cvxopt.solvers.lp(cvxopt.matrix(-g_t), G, g)
        point1 = point
        alpha = ((1 / T) * (1 + i / T)) / ((1 + (i+1) / T) ** 2)
        point = (1 - alpha) * point + alpha * np.array(sol['x'])
        g_t = g_t + sum(sub.stochastic_gradient_oracle(point) for j in range(batch)) / batch - sum(
            sub.stochastic_gradient_oracle(point1) for j in range(batch)) / batch
    return outcome


#
#
# our method: Hybrid_SPIDER_SFW
def hybrid_spider_sfw(H, T, batch):
    # global point1
    np.random.seed(k)
    # m = Submodular_QP.constraints.shape[0]
    # n = Submodular_QP.constraints.shape[1]
    outcome = [0] * (T)
    point = np.zeros((n, 1))
    G = cvxopt.matrix(np.concatenate([np.ones((1, n)), -np.ones((1, n)), -np.eye(n)]))  # 拼接三个矩阵的，按行拼接 np.eye(n)生成n维单位向量
    g = cvxopt.matrix(np.concatenate([np.array([[1], [-0.25]]), np.zeros((n, 1))]))  # 对应三个约束 Ax\leb,x\le1,-x\le0
    g_t = sum(H.stochastic_gradient_oracle(point) for j in range(batch)) / (batch)
    for i in range(T):
        outcome[i] = H.exact_value_oracle(point)
        sol = cvxopt.solvers.lp(cvxopt.matrix(-g_t), G, g)
        point1 = point
        alpha = ((1 / T) * (1 + i / T)) / ((1 + (i+1) / T) ** 2)
        point = (1 - alpha) * point + alpha * np.array(sol['x'])
        rho = 8 / (i + 8)
        g_t = sum(H.stochastic_gradient_oracle(point) for j in range(batch)) / batch + (1 - rho) * (g_t - sum(
            H.stochastic_gradient_oracle(point1) for j in range(batch)) / batch)
    return outcome


if __name__ == "__main__":
    delta = 2
    data = loadtxtmethod('train.txt')
    p = 0.002
    # print(data)
    r = np.array(data).shape[1]
    n = np.max(np.array(data)[:, 0:(r - 1)])
    # M = np.zeros((n, n))
    # # M = sub_matrix(n)
    epoch = 200
    K =10
    outcome2_K = 0
    outcome3_K = 0
    outcome4_K = 0
    outcome6_K = 0
    outcome7_K = 0


    for k in range(K):
        sub = Submodular_ex(data)
        sub.set_noise(delta=delta)
        sub.generate(data)

        my_font = font_manager.FontProperties(fname=
                                              "C:/Windows/Fonts/msyh.ttc")
        plt.xlabel(u'Iteration Index', fontproperties=my_font)
        plt.ylabel(u'Objective value', fontproperties=my_font)

        # outcome1 = sum(np.array(projected_gradient_ascent(sub, T=epoch, batch=1)) for k in range(K)) / K
        outcome2 = np.array(spider_stochastic_fw(sub, T=epoch, batch=epoch))
        outcome3 = np.array(fw(sub, T=epoch))
        outcome4 = np.array(fw2(sub, T=epoch))
        # outcome5 = np.array(hybrid_spider_sfw(sub, T=epoch, batch=1))
        outcome6 = np.array(hybrid_spider_sfw(sub, T=epoch, batch=100))
        outcome7 = np.array(hybrid_spider_sfw(sub, T=epoch, batch=20))

        my_font = font_manager.FontProperties(fname=
                                              "C:/Windows/Fonts/msyh.ttc")
        plt.xlabel(u'Iteration Index', fontproperties=my_font)
        plt.ylabel(u'Objective value', fontproperties=my_font)
        # plt.plot(range(len(outcome1)), outcome1, label='PGA')
        plt.plot(range(len(outcome2)), outcome2, label='spider_FW')
        # plt.plot(range(len(outcome5)), outcome5, label='hybrid_spider_FW(1)')
        plt.plot(range(len(outcome6)), outcome6, label='hybrid_spider_FW(100)')
        plt.plot(range(len(outcome7)), outcome7, label='hybrid_spider_FW(20)')
        plt.plot(range(len(outcome3)), outcome3, label='non_monotone_fw1')
        plt.plot(range(len(outcome4)), outcome4, label='non_monotone_fw2')
        plt.legend()
        plt.savefig("D:/2023文件/实验图像/20230702/non-monotone/temp{}.png".format(k))  # 输入地址，并利用format函数修改图片名称
        plt.clf()  # 需要重新更新画布，否则会出现同一张画布上绘制多张图片

        # outcome1 = sum(np.array(projected_gradient_ascent(sub, T=epoch, batch=1)) for k in range(K)) / K
        outcome2_K = outcome2_K + outcome2
        outcome3_K = outcome3_K + outcome3
        outcome4_K = outcome4_K + outcome4
        outcome6_K = outcome6_K + outcome6
        outcome7_K = outcome7_K + outcome7

        # outcome3 = sum(np.array(fw(sub, T=epoch)) for k in range(K)) / K
        # outcome4 = sum(np.array(fw2(sub, T=epoch)) for k in range(K)) / K
        # outcome5 = sum(np.array(hybrid_spider_sfw(sub, T=epoch, batch=epoch)) for k in range(K)) / K

    my_font = font_manager.FontProperties(fname=
                                          "C:/Windows/Fonts/msyh.ttc")
    plt.xlabel(u'Iteration Index', fontproperties=my_font)
    plt.ylabel(u'Objective value', fontproperties=my_font)
    # plt.plot(range(len(outcome1)), outcome1, label='PGA')
    plt.plot(range(len(outcome2_K)), outcome2_K / K, label='spider_FW')
    plt.plot(range(len(outcome6_K)), outcome6_K / K, label='hybrid_spider_FW(100)')
    plt.plot(range(len(outcome7_K)), outcome7_K / K, label='hybrid_spider_FW(20)')
    plt.plot(range(len(outcome3_K)), outcome3_K / K, label='non_monotone_fw1')
    plt.plot(range(len(outcome4_K)), outcome4_K / K, label='non_monotone_fw2')
    plt.legend()
    plt.savefig("D:/2023文件/实验图像/20230702/non-monotone/temp{average}.png")  # 输入地址，并利用format函数修改图片名称
    plt.show()

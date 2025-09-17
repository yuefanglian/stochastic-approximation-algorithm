import numpy as np
import math
import cvxopt
import matplotlib.pyplot as plt
from matplotlib import font_manager
from numpy import ndarray


# the first experiment about the offline QP
class Submodular_QP:
    noise = None
    # we use the Submodular_QP.constraint to record the constraint matrix A, i.e., Ax\\\\le b\n",
    constraints = None

    # we take QP as f(x)=1/2x^{T}Hx+h^{T}x\
    def __init__(self):
        self.H = None
        self.h = None

    def generate(self, n):
        # n is the dimension of x=(x_{1},...,x_{n})
        self.H = np.random.rand(n, n)  # 生成随机n维向量，每个元素服从01均匀分布
        #self.H = np.identity(n)  # 生成随机n维向量，每个元素服从01均匀分布
        self.H = 0.5 * (self.H + self.H.T)  # 保证H是一个对称矩阵
        self.H = -self.H  # H的元素均匀分布在[-1，0]上
        self.h = -self.H.dot(np.ones((n, 1)))  # 给定h=-H^{T}u,u=(1,...1)^{T},为了保证函数单调性

    # def matrix(self, x):
    #     H = self.H.dot(x)
    #     return cvxopt.matrix(H)
    #
    # def vector(self, x):
    #     y = np.zeros((x.shape[0], 1))
    #     x = y.T.dot(x.dot(y))
    #     h = self.h.dot(x)
    #     return cvxopt.matrix(h)

    def stochastic_gradient_oracle(self, x):
        # return a stochastic gradient g=\\\\nable f(x)+\\\\deta*U[-1,1] for E(g)=\\\\nable f(x)=Hx+h\\n\",\n",
        noise = Submodular_QP.noise * np.random.randn(self.H.shape[0], 1)  # 读取矩阵H的行数（shape[0]）,随机生成n*1的（高斯噪声）向量
        return self.H.dot(x) + self.h + noise

    def gradient_oracle(self, x):
        # return a stochastic gradient g=\\\\nable f(x)+\\\\deta*U[-1,1] for E(g)=\\\\nable f(x)=Hx+h\\n\",\n",
        return self.H.dot(x) + self.h

    def stochastic_boosting_gradient_oracle(self, x):
        # we first generate a z~Z where Pr(Z\\\\le z)=(exp(z-1)-exp(-1))/(1-exp(-1))\\n\",\n",
        z = math.log((1 - math.exp(-1)) * np.random.rand() + math.exp(-1)) + 1
        return (1 - math.exp(-1)) * self.stochastic_gradient_oracle(z * x)

    def exact_value_oracle(self, x: object) -> object:
        # return f(x) # 求解常数项c的值
        return float(0.5 * x.T.dot(self.H.dot(x)) + self.h.T.dot(x))

    @staticmethod
    def set_constraints(m, n):
        Submodular_QP.constraints = np.random.rand(m, n)  # 生成m*n的约束矩阵A，每个元素都均匀分布在[0,1]上

    @staticmethod
    def set_noise(delta):
        Submodular_QP.noise = delta  # 给噪声常数

    # def set_delay(self, num):
    #     self.delay = np.random.randint(1, num + 1)


# # first method: traditional gradient ascent
# def projected_gradient_ascent(sub: Submodular_QP, T, batch):
#     np.random.seed(k)
#     # m = Submodular_QP.constraints.shape[0]
#     # n = Submodular_QP.constraints.shape[1]
#     m = sub.constraints.shape[0]  # 约束集合行数
#     n = sub.constraints.shape[1]  # 约束集列数
#     outcome = [0] * T
#     point = np.zeros((n, 1))  # 生成n行1列的矩阵
#     # setting the projection QP as 1/2x^{T}Px+q^{T}x s.t Gx \\\\le g, Cx=c\\n\",\n",
#     G = cvxopt.matrix(
#         np.concatenate((Submodular_QP.constraints, np.eye(n), -np.eye(n)), axis=0))  # 拼接三个矩阵的，按行拼接 np.eye(n)生成n维单位向量
#     g = cvxopt.matrix(
#         np.concatenate((np.ones((m, 1)), np.ones((n, 1)), np.zeros((n, 1))), axis=0))  # 对应三个约束 Ax\leb,x\le1,-x\le0
#     P = cvxopt.matrix(2 * np.eye(n))  # 构造目标函数系数，\|x-point\|^{2}~ x^{2}-2point*x
#     for i in range(T):
#         gradient = sub.gradient_oracle(point)  # 小批量多次采样求平均
#         point += gradient / math.sqrt(i + 1)  # +=: point=point+gradient...
#         q = cvxopt.matrix(-2 * point)  # 构造目标函数终一次项系数 -2point
#         sol = cvxopt.solvers.qp(P, q, G, g)  # 求解二次规划问题：投影步
#         point = np.array(sol['x'])
#         point = point.reshape(n, 1)
#         outcome[i] = sub.exact_value_oracle(point)
#
#     return outcome


# # second method: boosting gradient ascent
# def boosting_gradient_ascent(H, T=500, batch=1):
#     np.random.seed(T)
#     # 种子随机数，每次加上这个代码都可以产生与此次一样的随机数
#     m = Submodular_QP.constraints.shape[0]
#     n = Submodular_QP.constraints.shape[1]
#     outcome = [0] * T
#     point = np.zeros((n, 1))
#     # setting the projection QP as 1/2x^{T}Px+q^{T}x s.t Gx \\\\le g, Cx=c\
#     G = cvxopt.matrix(np.concatenate((Submodular_QP.constraints, np.eye(n), -np.eye(n)), axis=0))
#     g = cvxopt.matrix(np.concatenate((np.ones((m + n, 1)), np.zeros((n, 1))), axis=0))
#     P = cvxopt.matrix(2 * np.eye(n))
#     for i in range(T):
#         gradient = sum(H.stochastic_boosting_gradient_oracle(point) for j in range(batch)) / batch
#         point += gradient / math.sqrt(i + 1)
#         q = cvxopt.matrix(-2 * point)
#         sol = cvxopt.solvers.qp(P, q, G, g)
#         point = np.array(sol['x'])
#         point = point.reshape(n, 1)
#         outcome[i] = H.exact_value_oracle(point)
#     return outcome


# the third method:conditional gradient ascent
def continuous_greedy_method(H, T):
    np.random.seed(k)
    m = Submodular_QP.constraints.shape[0]
    n = Submodular_QP.constraints.shape[1]
    outcome = [0] * T
    point = np.zeros((n, 1))
    G = cvxopt.matrix(np.concatenate((Submodular_QP.constraints, np.eye(n), -np.eye(n)), axis=0))
    g = cvxopt.matrix(np.concatenate((np.ones((m, 1)), np.ones((n, 1)), np.zeros((n, 1))), axis=0))
    for i in range(T):
        gradient = H.stochastic_gradient_oracle(point)
        sol = cvxopt.solvers.lp(cvxopt.matrix(-gradient), G, g)
        point = point + np.array(sol['x']) / T
        outcome[i] = H.exact_value_oracle(point)
    return outcome


# the fouth method: SCG
def stochastic_greedy_method(H, T=500):
    np.random.seed(k)
    m = Submodular_QP.constraints.shape[0]
    n = Submodular_QP.constraints.shape[1]
    outcome = [0] * T
    point = np.zeros((n, 1))
    G = cvxopt.matrix(np.concatenate((Submodular_QP.constraints, np.eye(n), -np.eye(n)), axis=0))
    g = cvxopt.matrix(np.concatenate((np.ones((m, 1)), np.ones((n, 1)), np.zeros((n, 1))), axis=0))
    gradient = H.stochastic_gradient_oracle(point)
    for i in range(T):
        rho = 4 / ((i + 8) ** (2 / 3))
        gradient = (1 - rho) * gradient + rho * H.stochastic_gradient_oracle(point)
        sol = cvxopt.solvers.lp(cvxopt.matrix(-gradient), G, g)
        point = point + np.array(sol['x']) / T
        outcome[i] = H.exact_value_oracle(point)
    return outcome


# def stochastic_greedy_method1(H, T, batch):
#     np.random.seed(k)
#     m = Submodular_QP.constraints.shape[0]
#     n = Submodular_QP.constraints.shape[1]
#     outcome = [0] * T
#     point = np.zeros((n, 1))
#     G = cvxopt.matrix(np.concatenate((Submodular_QP.constraints, np.eye(n), -np.eye(n)), axis=0))
#     g = cvxopt.matrix(np.concatenate((np.ones((m, 1)), np.ones((n, 1)), np.zeros((n, 1))), axis=0))
#     gradient = sum(H.stochastic_gradient_oracle(point) for j in range(batch)) / batch
#     for i in range(T):
#         rho = 4 / ((i + 8) ** (2 / 3))
#         gradient = (1 - rho) * gradient + rho * sum(H.stochastic_gradient_oracle(point) for j in range(batch)) / batch
#         sol = cvxopt.solvers.lp(cvxopt.matrix(-gradient), G, g)
#         point = point + np.array(sol['x']) / T
#         outcome[i] = H.exact_value_oracle(point)
#     return outcome


# # the final method:SCG++
# def stochastic_greedy_pp(H, T=500):
#     np.random.seed(T)
#     m = Submodular_QP.constraints.shape[0]
#     n = Submodular_QP.constraints.shape[1]
#     outcome = [0] * T
#     point = np.zeros((n, 1))
#     G = cvxopt.matrix(np.concatenate((Submodular_QP.constraints, np.eye(n), -np.eye(n)), axis=0))
#     g = cvxopt.matrix(np.concatenate((np.ones((m, 1)), np.ones((n, 1)), np.zeros((n, 1))), axis=0))
#     g_t = sum(H.stochastic_gradient_oracle(point) for j in range(100 ** 2)) / (100 ** 2)  # 随机梯度
#     for i in range(T):
#         gradient = cvxopt.matrix(-g_t)
#         sol = cvxopt.solvers.lp(gradient, G, g)
#         point1 = np.array(sol['x']) / T
#         point = point + point1
#         outcome[i] = H.exact_value_oracle(point)
#         g_t = g_t + H.H.dot(point1)
#         # print(H.H.dot(point1))
#     return outcome


#
#
# our method: SPIDER_SCG
def spider_stochastic_greedy(H, T, batch):
    np.random.seed(k)
    m = Submodular_QP.constraints.shape[0]
    n = Submodular_QP.constraints.shape[1]
    outcome = [0] * T
    point = np.zeros((n, 1))
    G = cvxopt.matrix(np.concatenate((Submodular_QP.constraints, np.eye(n), -np.eye(n)), axis=0))
    g = cvxopt.matrix(np.concatenate((np.ones((m, 1)), np.ones((n, 1)), np.zeros((n, 1))), axis=0))
    g_t = sum(H.stochastic_gradient_oracle(point) for j in range(batch ** 2)) / (batch ** 2)
    for i in range(T):
        sol = cvxopt.solvers.lp(cvxopt.matrix(-g_t), G, g)
        point1 = point
        point = point + np.array(sol['x']) / T
        outcome[i] = H.exact_value_oracle(point)
        g_t = g_t + sum(
            H.stochastic_gradient_oracle(point) - H.stochastic_gradient_oracle(point1) for j in range(batch)) / batch
        # g_t = g_t + H.H.dot(point - point1)
    return outcome


#

# our method: Hybrid_SPIDER_SGD
def hybrid_spider_sgd(H, T, batch):
    np.random.seed(k)
    m = Submodular_QP.constraints.shape[0]
    n = Submodular_QP.constraints.shape[1]
    outcome = [0] * T
    point = np.zeros((n, 1))
    G = cvxopt.matrix(np.concatenate((Submodular_QP.constraints, np.eye(n), -np.eye(n)), axis=0))
    g = cvxopt.matrix(np.concatenate((np.ones((m, 1)), np.ones((n, 1)), np.zeros((n, 1))), axis=0))
    g_t = sum(H.stochastic_gradient_oracle(point) for j in range(batch)) / (batch)
    for i in range(T):
        sol = cvxopt.solvers.lp(cvxopt.matrix(-g_t), G, g)
        point1 = point
        point = point + np.array(sol['x']) / T
        outcome[i] = H.exact_value_oracle(point)
        rho = 8 / (i + 8)
        # rho =0
        g_t = sum(H.stochastic_gradient_oracle(point) for j in range(batch)) / batch + (1 - rho) * (g_t - sum(
            H.stochastic_gradient_oracle(point1) for j in range(batch)) / batch)
    return outcome


# def exact_solver(sub, x):
#     m = sub.constraints.shape[0]  # 约束集合行数
#     n = sub.constraints.shape[1]
#     G = cvxopt.matrix(
#         np.concatenate((Submodular_QP.constraints, np.eye(n), -np.eye(n)),
#                        axis=0))  # 拼接三个矩阵的，按行拼接 np.eye(n)生成n维单位向量
#     g = cvxopt.matrix(
#         np.concatenate((np.ones((m, 1)), np.ones((n, 1)), np.zeros((n, 1))), axis=0))  # 对应三个约束 Ax\leb,x\le1,-x\le0
#     sol = cvxopt.solvers.qp(sub.matrix(x), sub.vector(x), G, g)  # 求解二次规划问题：投影步
#     s = np.array(sol['x'])
#     s = s.reshape(n, 1)
#     outcome = sub.exact_value_oracle(s)
#
#     return outcome


if __name__ == '__main__':
    delta = 10
    constrain_m = 12
    dimension = 25
    # delay_num = 5
    epoch = 80
    K = 20
    outcome2_K = 0
    outcome3_K = 0
    outcome7_K = 0
    outcome8_K = 0
    outcome9_K = 0
    # x = np.eye(dimension)

    for k in range(K):
        sub = Submodular_QP()
        sub.set_noise(delta=delta)
        sub.set_constraints(m=constrain_m, n=dimension)
        # sub.set_delay(delay_num)
        sub.generate(dimension)

        # outcome1 = projected_gradient_ascent(sub, T=epoch, batch=epoch)
        outcome2 = np.array(continuous_greedy_method(sub, T=epoch))
        outcome3 = np.array(stochastic_greedy_method(sub, T=epoch))
        # outcome4 = np.array(stochastic_greedy_method1(sub, T=epoch, batch=epoch))
        # outcome4 = sum(np.array(stochastic_greedy_pp(sub, T=epoch)) for k in range(K)) / K
        # outcome5 = sum(np.array(boosting_gradient_ascent(sub, T=epoch, batch=1)) for k in range(K)) / K
        # outcome6 = sum(np.array(boosting_gradient_ascent(sub, T=epoch, batch=100)) for k in range(K)) / K
        outcome7 = np.array(spider_stochastic_greedy(sub, T=epoch, batch=epoch))
        outcome8 = np.array(hybrid_spider_sgd(sub, T=epoch, batch=10))
        outcome9 = np.array(hybrid_spider_sgd(sub, T=epoch, batch=40))
        # print(exact_solver(sub, x), outcome1, outcome2[epoch - 1], outcome3[epoch - 1], outcome7[epoch - 1],
        #       outcome8[epoch - 1])

        my_font = font_manager.FontProperties(fname=
                                              "C:/Windows/Fonts/msyh.ttc")
        plt.xlabel(u'Iteration Index', fontproperties=my_font)
        plt.ylabel(u'Objective value', fontproperties=my_font)
        # plt.plot(range(len(outcome1)), outcome1, label='PGA')
        plt.plot(range(len(outcome2)), outcome2, label='CG')
        plt.plot(range(len(outcome3)), outcome3, label='SCG')
        # plt.plot(range(len(outcome4)), outcome4, label='SGA1')
        # plt.plot(range(len(outcome4)), outcome4, label='SG++')
        # plt.plot(range(len(outcome5)), outcome5, label='BGA(1)')
        # plt.plot(range(len(outcome6)), outcome6, label='BGA(100)')
        plt.plot(range(len(outcome7)), outcome7, label='spider_SCG')
        plt.plot(range(len(outcome8)), outcome8, label='hybrid_spider_SCG(10)')
        plt.plot(range(len(outcome9)), outcome9, label='hybrid_spider_SCG(40)')

        plt.legend()
        plt.savefig("D:/2023文件/实验图像/20230702/monotone/temp{}.png".format(k))  # 输入地址，并利用format函数修改图片名称
        plt.clf()  # 需要重新更新画布，否则会出现同一张画布上绘制多张图片

        outcome2_K = outcome2_K + outcome2
        outcome3_K = outcome3_K + outcome3
        outcome7_K = outcome7_K + outcome7
        outcome8_K = outcome8_K + outcome8
        outcome9_K = outcome9_K + outcome9

    my_font = font_manager.FontProperties(fname=
                                          "C:/Windows/Fonts/msyh.ttc")
    plt.xlabel(u'Iteration Index', fontproperties=my_font)
    plt.ylabel(u'Objective value', fontproperties=my_font)
    # plt.plot(range(len(outcome1)), outcome1, label='PGA')
    plt.plot(range(len(outcome2_K)), outcome2_K / K, label='CG')
    plt.plot(range(len(outcome3_K)), outcome3_K / K, label='SCG')
    # plt.plot(range(len(outcome4)), outcome4, label='SGA1')
    # plt.plot(range(len(outcome4)), outcome4, label='SG++')
    # plt.plot(range(len(outcome5)), outcome5, label='BGA(1)')
    # plt.plot(range(len(outcome6)), outcome6, label='BGA(100)')
    plt.plot(range(len(outcome7_K)), outcome7_K / K, label='spider_SCG')
    plt.plot(range(len(outcome8_K)), outcome8_K / K, label='hybrid_spider_SCG(10)')
    plt.plot(range(len(outcome9_K)), outcome9_K / K, label='hybrid_spider_SCG(40)')

    plt.legend()
    plt.savefig("D:/2023文件/实验图像/20230702/monotone/temp{average}.png")  # 输入地址，并利用format函数修改图片名称

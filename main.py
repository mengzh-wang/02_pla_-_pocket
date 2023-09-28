import numpy as np
from matplotlib import pyplot as plt
import time
from pla import pla
from pocket import pocket

"""----------------------数据集初始化----------------------"""
# 数据分布与规模
u1 = [5, 0]
s1 = [[1, 0], [0, 1]]
u2 = [0, 5]
s2 = [[1, 0], [0, 1]]
n = 200
train_rate = 0.8
n_train = int(n * train_rate)
n_test = n - n_train
# 数据填充
x1 = np.empty([n, 2])  # A
x2 = np.empty([n, 2])  # B
x_train = np.empty([n_train * 2, 2])  # 320
x_test = np.empty([n_test * 2, 2])  # 80

for i in range(n):  # 200
    x1[i] = np.random.multivariate_normal(u1, s1)
    x2[i] = np.random.multivariate_normal(u2, s2)

for i in range(n_train):  # 160
    x_train[i] = x1[i]  # A
    x_train[n_train + i] = x2[i]  # B
for i in range(n_test):  # 40
    x_test[i] = x1[i]  # A
    x_test[n_test + i] = x2[i]  # B

y_train = np.empty([n_train * 2, 1])
for i in range(n_train):
    y_train[i] = 1
    y_train[n_train + i] = -1
y_test = np.empty([n_test * 2, 1])
for i in range(n_test):
    y_test[i] = 1
    y_test[40 + i] = -1


time_pla_start = time.time()
max_iter_pla = 1000
w_pla = pla(x_train, y_train, max_iter_pla)
time_pla_end = time.time()
time_pla_spend = time_pla_end - time_pla_start

time_pocket_start = time.time()
max_iter_pocket = 1000
max_iter_no_change = 300
w_pocket = pocket(x_train, y_train, max_iter_pocket, max_iter_no_change)
time_pocket_end = time.time()
time_pocket_spend = time_pocket_end - time_pocket_start

x_min = min(min(x1[:, 0]), min(x2[:, 0]))
x_max = max(max(x1[:, 0]), max(x2[:, 0]))
y_min = min(min(x1[:, 1]), min(x2[:, 1]))
y_max = max(max(x1[:, 1]), max(x2[:, 1]))
x_co = np.linspace(x_min - 1, x_max + 1)


print("--------------PLA算法--------------")
print("w=", w_pla)

wrongCases_train_pla = 0
wrongCases_test_pla = 0

for i in range(n_train):
    if np.dot(w_pla, x_train[i]) * y_train[i] <= 0:
        wrongCases_train_pla += 1
wrongRate_train_pla = wrongCases_train_pla / n_train

for i in range(n_test):
    if np.dot(w_pla, x_test[i]) * y_test[i] <= 0:
        wrongCases_test_pla += 1
wrongRate_test_pla = wrongCases_test_pla / n_test

print("训练集正确率=", 1 - wrongRate_train_pla)
print("测试集正确率=", 1 - wrongRate_test_pla)
print("算法运行时间=", time_pla_spend, "s")

plt.figure("PLA算法")
str1="PLA, x1~N(%s,%s), x2~N(%s,%s)" % (u1,s1,u2,s2)
plt.title(str1)
z_pla = -(w_pla[0] / w_pla[1]) * x_co
plt.scatter(x1[:, 0], x1[:, 1], c='r')
plt.scatter(x2[:, 0], x2[:, 1], c='b')
plt.plot(x_co, z_pla, c='g')
plt.xlim(x_min - 1, x_max + 1)
plt.ylim(y_min - 1, y_max + 1)

print("--------------Pocket算法--------------")
print("w=", w_pocket)

wrongCases_train_pocket = 0
wrongCases_test_pocket = 0

for i in range(n_train):
    if np.dot(w_pocket, x_train[i]) * y_train[i] <= 0:
        wrongCases_train_pocket += 1
wrongRate_train_pocket = wrongCases_train_pocket / n_train

for i in range(n_test):
    if np.dot(w_pocket, x_test[i]) * y_test[i] <= 0:
        wrongCases_test_pocket += 1
wrongRate_test_pocket = wrongCases_test_pocket / n_test

print("训练集正确率=", 1 - wrongRate_train_pocket)
print("测试集正确率=", 1 - wrongRate_test_pocket)
print("算法运行时间=", time_pocket_spend, "s")

plt.figure("Pocket算法")
str2="Pocket, x1~N(%s,%s), x2~N(%s,%s)" % (u1,s1,u2,s2)
plt.title(str2)
z_pocket = -(w_pocket[0] / w_pocket[1]) * x_co
plt.scatter(x1[:, 0], x1[:, 1], c='r')
plt.scatter(x2[:, 0], x2[:, 1], c='b')
plt.plot(x_co, z_pocket, c='g')
plt.xlim(x_min - 1, x_max + 1)
plt.ylim(y_min - 1, y_max + 1)


plt.show()

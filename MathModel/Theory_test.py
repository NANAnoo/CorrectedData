import matplotlib.pyplot as plt
import numpy as np
import torch

w_size = 3

data_c = np.load('data_c.npy')

n = data_c.shape[0]
c = data_c.shape[1]

# 以真实浓度数据作为X
# x = data_c / (data_c + 50) * 50

# 以随机值作为X
x = np.random.uniform(0.1,2,n*c).reshape(n, c)

# 随机初始化每条直线的斜率
k = np.random.uniform(0,1,n).repeat(c).reshape(n,c)

# 初始化系数W
w = np.random.uniform(-1,1,w_size)


def noice(x, l = 0.1):
    return np.random.uniform(-l,l,x.size).reshape(x.shape)

# 作为真实误差，形式可以改变，满足error(0) = 0 即可
def error(x):
    return -1.0/(x + 0.1)*np.sin(np.log(x + 0.1))*x

# 作为修正函数，以多项式表示 Wi* x**(i+1)
def correct_func(x,w):
    ans = 0
    j = 2
    for i in w:
        ans += i*x**j
        j+=1
    return ans

# 添加误差后的曲线
def fade_f(k,x,n = 0.0):
    return k*x + error(x) + noice(x,n)

# 真实的直线
def real_f(k,x):
    return k*x

# 修正后的曲线
def fixed_f(k,x,w):
    return fade_f(k,x) - correct_func(x,w)

# 任意一点的斜率表示式： K = (Y - correct_func(X))/X
def func(y, x, w):
    ans = y/x
    j = 1
    for i in w:
        ans -= i*x**j
        j += 1
    return ans

# 每个W的偏导数
def dfunc_w(x,w):
    ans = []
    j = 1
    for i in w:
        ans.append(-x**j)
        j+=1
    return np.array(ans)

# 两点间的斜率差值的导数，d【(K1 - K2)**2】
def dloss(y1,x1,y2,x2,w):
    return 2 * (func(y1, x1,w) - func(y2, x2,w)) * (dfunc_w(x1,w) - dfunc_w(x2,w))

# 总误差
def total_loss(w):
    ans = 0
    for i in range(n):
        for j in range(c):
            for k in range(c-1-j):
                ans += (func(y[i][j],x[i][j],w) - func(y[i][k+j+1],x[i][k+j+1],w))**2
    return ans

y = fade_f(k,x,0.01)

epoch = 30000

LR = 0.001

print('total_loss',total_loss(np.zeros(w_size)))
print(func(y, x, w))

fig = plt.figure()
x0 = np.linspace(0,1.5,50)
k0 = 1
plt.ion()

# 训练过程，每组任意两点斜率误差之和

for i in range(epoch):
    dw = np.zeros_like(w)
    for t in range(n):
        for j in range(c):
            for k in range(c - j - 1):
                dw += dloss(y[t][j], x[t][j], y[t][k + j + 1], x[t][k + j + 1],w)
    w = w - dw*LR
    if i%500 == 0:
        print('total_loss',total_loss(w),w,dw)
        plt.cla()
        plt.plot(x0,real_f(k0,x0),c='green')
        plt.scatter(x0,fade_f(k0,x0),c='red')
        plt.plot(x0,fixed_f(k0,x0,w),c='blue')
        plt.pause(0.01)
plt.ioff()
plt.show()
print(func(y, x, w))
plt.savefig('1')
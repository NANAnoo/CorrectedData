import matplotlib.pyplot as plt
import numpy as np


# 此思路为废案，仅供参考

epoch = 10000

l = 1

LR = 0.001

w_size = 3

data_c = np.load('data_c.npy')
data_p = np.load('data_p.npy')

n = data_c.shape[0]
c = data_c.shape[1]

x = data_c / (data_c + 50) * 50

y = data_p[l][1:].reshape(n,c)

y0 = data_p[l][0]

t = n
d = 0
x = x[d:t]
y=  y[d:t]
n = t - d

def func(w,y,x,y0):
    ans = 0
    j = 1
    for i in w :
        ans += i*(y**j - y0**j)
        j += 1
    return ans / x

def d_func(w,y,x,y0):
    ans = []
    j = 1
    for i in w:
        ans.append((y**j - y0**j)/x)
        j +=0
    return np.array(ans)

def total_loss(w,y,x,y0):
    ans = 0
    for i in range(n):
        for j in range(c):
            for k in range(c-1-j):
                ans += (func(w,y[i][j],x[i][j],y0) - func(w,y[i][k+j+1],x[i][k+j+1],y0))**2
    return ans

def dloss(y1,x1,y2,x2,w,y0):
    return 2 * (func(w,y1, x1,y0) - func(w,y2, x2,y0)) * (d_func(w,y1, x1,y0) - d_func(w,y2, x2,y0))

w = np.random.uniform(-1,1,w_size)

print(total_loss(w,y,x,y0))
print(func(w,y,x,y0))


for i in range(epoch):
    dw = np.zeros_like(w)
    for t in range(n):
        for j in range(c):
            for k in range(c - j - 1):
                dw += dloss(y[t][j], x[t][j], y[t][k + j + 1], x[t][k + j + 1],w,y0)
    w = w - dw*LR
    if i%1000 == 0:
        print(total_loss(w,y,x,y0),w,dw)

print(func(w,y,x,y0))
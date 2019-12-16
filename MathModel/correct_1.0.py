# import matplotlib.pyplot as plt
import numpy as np

# return F
def math_model(ingredient, model = 'km'):
    if model == 'km':
        return (np.ones_like(ingredient) - ingredient)**2 / (ingredient * 2)
    elif model == 'recip':
        return 1.0 / (ingredient + 1.0)
    else:
        print('Sorry no model of that name')
        exit(1)


# return ingredient
def i_math_model(f, model='km'):
    if model == 'km':
        return f - ((f + 1) ** 2 - 1) ** 0.5 + 1
    elif model == 'recip':
        return 1.0 / f - 1.0
    else:
        print('Sorry no model of that name')
        exit(1)


def correct_func(x,w):
    ans = 0
    j = 2
    for i in w:
        ans += i*x**j
        j+=1
    return ans

def func(y, x, w):
    ans = y/x
    j = 1
    for i in w:
        ans -= i*x**j
        j += 1
    return ans

def dfunc_w(x,w):
    ans = []
    j = 1
    for i in w:
        ans.append(-x**j)
        j+=1
    return np.array(ans)

# 考虑常数项
'''
def func(y, x, w):
    ans = y/x - w[0]/x
    j = 1
    for i in w[1:]:
        ans -= i*x**j
        j += 1
    return ans

def dfunc_w(x,w):
    ans = [-1.0/x]
    j = 1
    for i in w[1:]:
        ans.append(-x**j)
        j+=1
    return np.array(ans)
'''


def dloss(y1,x1,y2,x2,w):
    return 2 * (func(y1, x1,w) - func(y2, x2,w)) * (dfunc_w(x1,w) - dfunc_w(x2,w))

def total_loss(w,n,c,y,x):
    ans = 0
    for i in range(n):
        for j in range(c):
            for k in range(c-1-j):
                ans += (func(y[i][j],x[i][j],w) - func(y[i][k+j+1],x[i][k+j+1],w))**2
    return ans

# data_c 浓度数据，21种色浆 * 3次取点
# data_p 分光反射率数据 size: (1 + 21 * 3) * 31 , 1为基底
data_c = np.load('data_c.npy')
data_p = np.load('data_p.npy')

def train(l,epoch,LR,w_size):
    print('lamda = ', 400 + l * 10)
    n = data_c.shape[0]
    c = data_c.shape[1]

    x = data_c

    # 以 K/S作为Y_fade
    y = data_p[l][1:].reshape(n, c)

    y = math_model(y) - math_model(data_p[l][0])

    w = np.random.uniform(-1, 1, w_size)

    print('total_loss', total_loss(np.zeros(w_size),n,c,y,x))

    for i in range(epoch):
        dw = np.zeros_like(w)
        for t in range(n):
            for j in range(c):
                for k in range(c - j - 1):
                    dw += dloss(y[t][j], x[t][j], y[t][k + j + 1], x[t][k + j + 1], w)
        w = w - dw * LR

        if i % 1000 == 0:
            print('total_loss', total_loss(w,n,c,y,x), w, dw)

    return w



data_w = []

for i in range(31):
    data_w.append(train(i,10000,0.001,2))

print(data_w)

np.save('data_w_3',data_w)

'''
# 选择哪个波长
l = 7
print('lamda = ', 400 + l*10)

epoch = 10000

LR = 0.001

w_size = 3


n = data_c.shape[0]
c = data_c.shape[1]

x = data_c / (data_c + 50) * 50


# 以 K/S作为Y_fade
y = data_p[l][1:].reshape(n,c)

y = math_model(y) - math_model(data_p[l][0])

# d,t 控制选择色浆的分组
t = n
d = 0
x = x[d:t]
y=  y[d:t]
n = t - d


w = np.random.uniform(-1,1,w_size)

print('total_loss',total_loss(np.zeros(w_size)))
print(func(y,x,np.zeros_like(w)))
fig = plt.figure()

plt.ion()
xx = np.ravel(x)
yy = np.ravel(y)
x0 = np.linspace(0,1.5,50)
plt.plot(x0, correct_func(x0, w), c='green')
plt.scatter(xx, yy, c='blue')

for i in range(epoch):
    dw = np.zeros_like(w)
    for t in range(n):
        for j in range(c):
            for k in range(c - j - 1):
                dw += dloss(y[t][j], x[t][j], y[t][k + j + 1], x[t][k + j + 1],w)
    w = w - dw*LR

    if i%1000 == 0:
        print('total_loss',total_loss(w),w,dw)
        plt.cla()
        plt.plot(x0, correct_func(x0, w), c='green')
        plt.scatter(xx, yy, c='blue')
        plt.pause(0.01)

plt.ioff()
plt.show()
print(func(y, x, w))

'''
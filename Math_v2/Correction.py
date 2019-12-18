import Math_v2.Functions as mf
import numpy as np

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

def dloss(y1,x1,y2,x2,w):
    return 2 * (func(y1, x1,w) - func(y2, x2,w)) * (dfunc_w(x1,w) - dfunc_w(x2,w))

def total_loss(w,n,c,y,x):
    ans = 0
    for i in range(n):
        for j in range(c):
            for k in range(c-1-j):
                ans += (func(y[i][j],x[i][j],w) - func(y[i][k+j+1],x[i][k+j+1],w))**2
    return ans

def train(data_c,data_p,l,epoch,LR,w_size):
    print('lamda = ', 400 + l * 10)
    n = data_c.shape[0]
    c = data_c.shape[1]

    x = data_c

    # 以 K/S作为Y_fade
    y = data_p[l][1:].reshape(n, c)

    y = mf.math_model(y) - mf.math_model(data_p[l][0])

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

def main():

    epoch = 10000
    LR = 0.001
    w_size = 2
    save_name = 'data_w_size12'
    fil = 1
    # data_c 浓度数据，21种色浆 * 3次取点
    # data_p 分光反射率数据 size: (1 + 21 * 3) * 31 , 1为基底
    data_c = np.load('data_c.npy')
    data_p = np.load('data_p.npy')

    # 6 , 12 , 18
    filiters = np.array(
        [[0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]]
    )

    t_c, t_p = mf.data_filiter(filiters[fil],data_c,data_p)

    data_w = []

    for i in range(31):
        data_w.append(train(t_c, t_p, i, epoch, LR, w_size))

    print(data_w)

    np.save(save_name, data_w)

main()
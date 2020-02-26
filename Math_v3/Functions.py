import numpy as np

# CIE标准照明体D65光源，10°视场
optical_relevant = np.array([[0.136, 0.667, 1.644, 2.348, 3.463, 3.733, 3.065, 1.934, 0.803, 0.151, 0.036, 0.348, 1.062,
                              2.192, 3.385, 4.744, 6.069, 7.285, 8.361, 8.537, 8.707, 7.946, 6.463, 4.641, 3.109, 1.848,
                              1.053, 0.575, 0.275, 0.120, 0.059],
                             [0.014, 0.069, 0.172, 0.289, 0.560, 0.901, 1.300, 1.831, 2.530, 3.176, 4.337, 5.629, 6.870,
                              8.112, 8.644, 8.881, 8.583, 7.922, 7.163, 5.934, 5.100, 4.071, 3.004, 2.031, 1.295, 0.741,
                              0.416, 0.225, 0.107, 0.046, 0.023],
                             [0.613, 3.066, 7.820, 11.589, 17.755, 20.088, 17.697, 13.025, 7.703, 3.889, 2.056, 1.040,
                              0.548, 0.282, 0.123, 0.036, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
                              0.000, 0.000, 0.000, 0.000, 0.000, 0.000]])
perfect_white = np.array([[94.83], [100.00], [107.38]])

def reflectance2rgb(reflectance):
    tri = np.dot(optical_relevant, reflectance.reshape(31, 1))
    M = np.array(
        [[3.240479, -1.537150, -0.498535],
         [-0.969256, 1.875992, 0.041556],
         [0.055648, -0.204043, 1.057311]])
    return M.dot(tri)

def reflectance2lab(reflectance):
    tri = np.dot(optical_relevant, reflectance.reshape(31, 1))
    return xyz2lab(tri)

def reflectance2xyz(reflectance):
    return np.dot(optical_relevant, reflectance.reshape(31, 1))

# lab 色差
def color_diff(reflectance1, reflectance2):
    tri1 = np.dot(optical_relevant, reflectance1.reshape(31, 1))
    tri2 = np.dot(optical_relevant, reflectance2.reshape(31, 1))

    lab1 = xyz2lab(tri1)
    lab2 = xyz2lab(tri2)
    delta_lab = lab1 - lab2

    diff = (delta_lab[0] ** 2 + delta_lab[1] ** 2 + delta_lab[2] ** 2) ** (1 / 2)
    return diff

def xyz2lab(xyz):
    r = 0.008856
    lab = np.zeros(3 * 1)

    if xyz[0] / perfect_white[0] > r and xyz[1] / perfect_white[1] > r and xyz[2] / perfect_white[2] > r:
        lab[0] = (xyz[1] / perfect_white[1]) ** (1 / 3) * 116 - 16
        lab[1] = ((xyz[0] / perfect_white[0]) ** (1 / 3) - (xyz[1] / perfect_white[1]) ** (1 / 3)) * 500
        lab[2] = ((xyz[1] / perfect_white[1]) ** (1 / 3) - (xyz[2] / perfect_white[2]) ** (1 / 3)) * 200
    else:
        lab[0] = (xyz[1] / perfect_white[1]) * 903.3
        lab[1] = (xyz[0] / perfect_white[0] - xyz[1] / perfect_white[1]) * 3893.5
        lab[2] = (xyz[1] / perfect_white[1] - xyz[2] / perfect_white[2]) * 1557.4

    return lab

# return F
def math_model(ingredient, model = 'km'):
    if model == 'km':
        return (np.ones_like(ingredient) - ingredient)**2 / (ingredient * 2)
    elif model == 'four_flux':
        return (8 * ingredient + (1.0 - 6 * ingredient) *
               ((4 * ingredient ** 2 - 4 * ingredient + 25.0) ** 0.5)
               + 12 * ingredient ** 2 + 5.0)/ (48 * ingredient)
    else:
        print('Sorry no model of that name')
        exit(1)


# return ingredient
def i_math_model(f, model='km'):
    f = np.squeeze(f)
    for i in range(len(f)):
        if f[i]<0:
            f[i]=0
    if model == 'km':
        return f - ((f + 1) ** 2 - 1) ** 0.5 + 1
    elif model == 'four_flux':
        return 0.5 * (1 / ((4 * (f ** 2) + 4 * f) ** 0.5 + 2 * f + 1.0)) + 0.5 * (
                (((f + 1.0) * ((4 * (f ** 2) + 4 * f) ** 0.5)) + 2 * (f ** 2) - 2 * 1.0) / (
                2 * (f + 1.0) * (3 * f - 1.0) * (
                ((4 * (f ** 2) + 4 * f) ** 0.5) + 2 * f + 1.0)))
    else:
        print('Sorry no model of that name')
        exit(1)



# 根据filiter切分数据
def data_filiter(filiter,data_C,data_P):
    c = []
    p = [data_P.T[0]]
    for i in range(filiter.size):
        if filiter[i] == 1:
            c.append(data_C[i])
            p.append(data_P.T[3 * i + 1])
            p.append(data_P.T[3 * i + 2])
            p.append(data_P.T[3 * i + 3])
    return np.array(c),np.array(p).T

def data_filiter_(filiter,data):
    ans = []
    for i in range(filiter.size):
        if filiter[i] == 1:
            ans.append(data[i])
    return np.array(ans)

def pad_by_filiter(filiter,data):
    ans = np.zeros_like(filiter).astype(float)
    j = 0
    for i in range(filiter.size):
        if filiter[i]==1:
            ans[i] = data[j]
            j += 1

    return ans

def get_filiter(sample):
    ans = []
    for i in sample:
        if i == 0:
            ans.append(0)
        else :
            ans.append(1)
    return np.array(ans)
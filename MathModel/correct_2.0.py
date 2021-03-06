import matplotlib.pyplot as plt
import numpy as np

RealCompose = np.array([
        [0.127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0748, 0, 0, 0, 0, 0, 0.56, 0, 0, 0],
        [0, 0.8014, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1491, 0, 0, 0, 0.2241, 0],
        [0.0451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7428, 0, 0, 0, 0, 0, 0, 0, 0.3364, 0],
        [0.3306, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0615, 0, 0.1219, 0, 0, 0]
    ])
RealIngredient = np.array(
    [[0.2869235,0.3562616,0.3648288,0.3690848,0.3720965,0.3743734,0.3748034,0.3726986,0.3670701,0.3595550,
      0.3495962,0.3348914,0.3152257,0.2898063,0.2588453,0.2237525,0.1902168,0.1687493,0.1594849,0.1573166,
      0.1543398,0.1508693,0.1542613,0.1652469,0.1817535,0.1943915,0.2001948,0.1974784,0.1753394,0.1711839,
      0.1716521],
     [0.2673378,0.3132285,0.3183329,0.3234908,0.3318701,0.3409707,0.3604081,0.4168356,0.5351773,0.6202191,
      0.6618687,0.6919741,0.7136238,0.7292901,0.7314631,0.7131701,0.6773048,0.6302681,0.5738088,0.5133060,
      0.4535525,0.4108878,0.3908512,0.3808001,0.3752591,0.3727644,0.3801365,0.3976869,0.4237110,0.4332685,
      0.4433292],
     [0.2805455,0.3224138,0.3143094,0.3027386,0.2937162,0.2832071,0.2719996,0.2643588,0.2582215,0.2541490,
      0.2498622,0.2446374,0.2448266,0.2456081,0.2388589,0.2233387,0.2128640,0.2172354,0.2586341,0.4077814,
      0.5150258,0.5389543,0.5433356,0.5433164,0.5431498,0.5423625,0.5413173,0.5409844,0.5348357,0.5314131,
      0.5256729],
     [0.2335778,0.2669207,0.2692738,0.2711587,0.2714446,0.2716927,0.2717375,0.2719162,0.2708953,0.2690464,
      0.2680669,0.2653399,0.2614551,0.2566476,0.2497642,0.2403944,0.2294249,0.2198784,0.2130321,0.2085395,
      0.2036385,0.1986061,0.1988262,0.2010616,0.2045918,0.2065673,0.2080161,0.2085742,0.2017922,0.2004079,
      0.1997598]
    ]
)

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
    elif model == 'recip':
        return 1.0 / (ingredient + 1.0)
    else:
        print('Sorry no model of that name')
        exit(1)


# return ingredient
def i_math_model(f, model='km'):
    #print(f)
    if model == 'km':
        return f - ((f + 1) ** 2 - 1) ** 0.5 + 1
    elif model == 'recip':
        return 1.0 / f - 1.0
    else:
        print('Sorry no model of that name')
        exit(1)

data_c = np.load('data_c.npy')
data_p = np.load('data_p.npy').T

base_f = math_model(data_p[0])
data_p = data_p[1:].T

def get_W(w_size):
    W = []
    for i in range(data_c.shape[0]):
        line = []
        for j in range(data_p.shape[0]):
            line.append(np.polyfit(data_c[i],(math_model(data_p[j][i*3:i*3+3]) - base_f[j])/data_c[i],w_size))
        W.append(line)
    return W

def main():
    com = 1
    W = np.load('data_w_02.npy')
    #W = get_W(10)
    #np.save('data_w_02',np.array(W))
    c =  RealCompose[com]
    I = np.zeros(31)
    for i in range(21):
        temp = np.zeros(31)
        if c[i] != 0:
            for j in range(31):
                temp[j] += np.poly1d(W[i][j])(c[i]) * c[i]
        I += temp
    I = i_math_model(I + base_f)

    print(color_diff(RealIngredient[com],I))

main()
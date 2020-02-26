import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
import Math_v3.Functions as mf
import numpy as np


RealCompose = np.array(
    [
        [0.127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0748, 0, 0, 0, 0, 0, 0.56, 0, 0, 0],
        [0, 0.8014, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1491, 0, 0, 0, 0.2241, 0],
        [0.0451, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7428, 0, 0, 0, 0, 0, 0, 0, 0.3364, 0],
        [0.3306, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0615, 0, 0.1219, 0, 0, 0]
    ]
)

RealIngredient = np.array(
    [
     [0.2869235,0.3562616,0.3648288,0.3690848,0.3720965,0.3743734,0.3748034,0.3726986,0.3670701,0.3595550,
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

# 反射率差权重
diff_weight = np.array([0.06953774, 0.34547297, 0.87092963, 1.26676032, 1.88873878,
       2.05036283, 1.6786528 , 1.07535807, 0.66317696, 0.83183059,
       1.21411871, 1.53150604, 1.71718416, 1.79391978, 1.66120845,
       1.43628235, 1.15224085, 0.98674527, 1.08886807, 1.25134826,
       1.43380307, 1.42401593, 1.2250983 , 0.90907449, 0.62465464,
       0.37832069, 0.21716633, 0.11913914, 0.05713688, 0.02509716,
       0.01225074])


def Sanderson(Rm, k1, k2):
    return  (Rm - k1) / (1.0 - k1 - k2 + k2*Rm)

def i_Sanderson(R, k1, k2):
    return  ((1.0 - k1 - k2)*R + k1) / ( 1.0 - k2 * R)


n = 1.5
# Fresnel
k1 = (n-1.0)**2 / (n+1.0)**2
k2 = 0.5

data_c = np.load('data_c.npy')
data_p = np.load('data_p.npy').T

# F： 各色浆单位浓度下的K/S， m*n， m:色浆数， n:波长数
# c： 配方的浓度向量
# Model： 配色模型的反函数
def Mix(F,base_f,c,Model):
    return mf.i_math_model(c.dot(F)+ base_f,Model)


# 获取原料色浆的F矩阵, F：各色浆在每个波长单位浓度下的K/S， m*n， m:色浆数， n:波长数
def get_F(p,c,base,Model):
    ff = mf.math_model(p,Model)
    F = []
    sample_size = c.shape[1]

    for i in range(c.shape[0]):
        k = 0.0
        for j in range(sample_size):
            k += ff[sample_size*i + j] / c[i][j]
        F.append(k/sample_size)
    return np.array(F)

# load data
RealCompose_2 = np.load('RealCompose_2.npy')
RealIngredient_2 = np.load('RealIngredint_2.npy')

Composes = np.append(RealCompose_2,RealCompose,axis=0)
Ingredients = np.append(RealIngredient_2,RealIngredient,axis=0)


R = Sanderson(data_p,k1,k2)

def get_diffs(model):
    base = mf.math_model(R[0],model)
    F = get_F(R[1:],data_c,base,model)

    diffs = []
    for i in range(Composes.shape[0]):
        PredictIngredint =Mix(F, base, Composes[i], model)
        dif = mf.color_diff(i_Sanderson(PredictIngredint,k1,k2), Ingredients[i])
        if dif < 50:
            dif = (i_Sanderson(PredictIngredint,k1,k2)-Ingredients[i])*diff_weight
            diffs.append(dif)
        else:
            print(i)
    return np.array(diffs)

def plot(label,diffs,i):
    X = np.arange(len(diffs))
    plt.scatter(X, diffs, label=label)
    plt.plot(X, np.repeat(sum(diffs) / len(diffs), len(diffs)))
    E = sum(diffs) / len(diffs)
    D = sum((diffs - E)**2) / len(diffs)
    text = 'E: %.5f'%E+'  D: %.5f'%D
    plt.text(i, E, text)

def plot_sub(diffs,ax):
    X = np.arange(diffs.shape[0])[:,np.newaxis]
    ax.set_xticks(X)
    X = np.repeat(X,diffs.shape[1],axis=1)
    Y = np.arange(diffs.shape[1])
    Y = np.tile(Y, diffs.shape[0])
    ax.scatter3D(X,Y,diffs)

fig = plt.figure('km-V-four_flux_with_sanderson')

ax = Axes3D(fig)
model = 'km'
diffs = get_diffs(model)
plot_sub(diffs,ax)

# model = 'km'
# label = model + '_with_sanderson'
# diffs = get_diffs(model)
# plot(label,diffs,10)
#
# model = 'four_flux'
# label = model + '_with_sanderson'
# diffs = get_diffs(model)
# plot(label,diffs,1)
#
# plt.xlabel('加料方案',fontproperties=font_set)
# plt.ylabel('色差',fontproperties=font_set)
#
# X = np.arange(len(diffs))
# plt.xticks(X)
#
# plt.legend()

plt.show()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import Math_v3.Functions as mf
import numpy as np
from tqdm import tqdm

model = 'km'

data_c = np.load('data_c.npy')
data_p = np.load('data_p.npy').T

def get_loss(f,c):
    k = f/c
    return (k[0]-k[1])**2 + (k[1]-k[2])**2 + (k[0]-k[2])**2

def get_total_loss(K1,K2):
    R =(data_p - K1) / (1 - K1 - K2 + K2*data_p)
    F = mf.math_model(R,model)
    ans = 0
    for i in range(31):
        for j in range(21):
            ans += get_loss(F[i][3*j+1:3*j+4] - F[i][0],data_c[j])
    return ans

# L = []
# index = []
# k1 = np.linspace(0,0.8,100)
# k2 = np.linspace(0,0.8,100)
#
# for K1 in tqdm(k1):
#     for K2 in k2:
#         L.append(get_total_loss(K1,K2))
#         index.append([K1,K2])
# min = 100000
# ans = []
# for i in range(len(L)):
#     if L[i]<min:
#         min = L[i]
#         ans = index[i]
# print(ans, min)
# N = np.linspace(1.2,1.6,100)
# L = []
# for n in N:
#     k1 = (n-1.0)**2 / (n+1.0)**2
#     L.append([get_total_loss(k1,0.5),n])
#
# min = 100000
# ans = []
# for i in range(len(L)):
#     if L[i][0]<min:
#         min = L[i][0]
#         ans = L[i]
# # [0.47020162099114465, 1.2665656565656567]
# print(ans)

N = np.linspace(1.5,1.4,31)
# #

# print(get_total_loss(k1,0.5),n,k1)
# print()
n = 1.5
K1 = (n-1.0)**2 / (n+1.0)**2
K2 = 0.5

# for i in range(31):
#     n = N[i]
#     K1 = (n-1.0)**2 / (n+1.0)**2
#     data_p[i] = (data_p[i] - K1) / (1 - K1 - K2 + K2*data_p[i])
# data_p = data_p.T
data_p =(data_p - K1) / (1 - K1 - K2 + K2*data_p)

Y = np.arange(400,710,10)
Y = np.tile(Y,3)

def plot_ks(id,ax):
    ax.set_title('id=%d'%id)
    ks = mf.math_model(data_p[id*3+1:id*3+4],"four_flux") - mf.math_model(data_p[0],"four_flux")
    c = data_c[id][:,np.newaxis]
    ks = ks/c
    c = np.repeat(c, 31,axis=1).reshape(93)
    #ks = ks.reshape(93)
    ax.scatter3D(c,Y,ks)
    ax.set_xlabel('C')
    ax.set_ylabel('wavelength')
    ax.set_zlabel('k/s per unit')


fig = plt.figure()
ax = Axes3D(fig)
plot_ks(19,ax)
# ax = fig.subplots(7,3)
# for j in range(7):
#     i = j+14
#     xyz = np.array([mf.reflectance2lab(data_p[1 + 3*i]),mf.reflectance2lab(data_p[2 + 3*i]),mf.reflectance2lab(data_p[3 + 3*i])]).T
#     ax[j][0].scatter(data_c[i],xyz[0],color='coral')
#
#     ax[j][1].scatter(data_c[i],xyz[1])
#
#     ax[j][2].scatter(data_c[i],xyz[2],color='red')
# plt.savefig('XYZ_C3.png')
plt.show()

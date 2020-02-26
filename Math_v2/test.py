import matplotlib.pyplot as plt
import Math_v2.Functions as mf
import numpy as np

data_c = np.load('data_c.npy')
data_p = np.load('data_p.npy').T

print(np.load('RealCompose_2.npy'))

C = [mf.reflectance2rgb(data_p[0]),mf.reflectance2rgb(data_p[0]),mf.reflectance2rgb(data_p[0])]
for i in data_p[1:]:
    C.append(mf.reflectance2rgb(i) / 100.0)

C = np.array(C).reshape(22,3,3)
print(C)
plt.figure()
plt.imshow(C)
plt.figsize= (150, 50*64)
# ignore ticks
plt.xticks([])
plt.yticks([])
# plt.legend()
plt.savefig('梯度.pdf')
plt.show()

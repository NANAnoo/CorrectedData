import numpy as np
name = np.array(
    ['07H','08','08S','09','09B','09S','10B','12','13','14','15','16','17A','18A','19A','20A-2','23A','2740','2803','2804','2807']
)

RealCompose = np.array(
    [
        {'07H' :0.127,'2740':0.56,'16':0.0748},
        {'20A-2':0.1491,'2804':0.2241,'08':0.8014},
        {'2804':0.3364,'16':0.7428,'07H':0.0451},
        {'07H':0.3306,'2740':0.1219,'20A-2':0.0615}
    ]
)

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

data_c = np.load('data_c.npy')
data_p = np.load('data_p.npy').T

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

def correct_func(x,w):
    ans = 0
    j = 2
    for i in w:
        if type(x) == type(w):
            xx = np.array(x**j).reshape(x.size,1)
            ans += xx.dot(i.reshape(1,31))
        else:
            ans += i*x**j
        j+=1
    return ans
base_f = math_model(data_p[0])

def Mix(compose):
    F = np.zeros_like(base_f)
    for i in compose.keys():
        index = np.argwhere(name == i)[0][0]
        df = (math_model(data_p[index*3 + 2]) - base_f)/ data_c[index][1]
        F += df * compose[i]
    return i_math_model(F + base_f)

def corrected_Mix_1(compose,w):
    F = np.zeros_like(base_f)
    total_c = 0
    for i in compose.keys():
        index = np.argwhere(name == i)[0][0]
        df = (math_model(data_p[index * 3 + 1]) - base_f - correct_func(data_c[index][0],w)) / data_c[index][0]
        df += (math_model(data_p[index * 3 + 2]) - base_f - correct_func(data_c[index][1], w)) / data_c[index][1]
        df += (math_model(data_p[index * 3 + 3]) - base_f - correct_func(data_c[index][2], w)) / data_c[index][2]
        F += df * compose[i] / 3
        total_c += compose[i]
    # 这里的scale原本用来防止K/A变成负数，调整后发现可以提升精度
    scale = 0.7
    return i_math_model(F + correct_func(total_c* scale ,w) + base_f)

def get_dfs_KM(w):
    dfs = []
    for i in range(data_c.shape[0]):
        df = (math_model(data_p[i * 3 + 1]) - base_f - correct_func(data_c[i][0], w)) / data_c[i][0]
        df += (math_model(data_p[i * 3 + 2]) - base_f - correct_func(data_c[i][1], w)) / data_c[i][1]
        df += (math_model(data_p[i * 3 + 3]) - base_f - correct_func(data_c[i][2], w)) / data_c[i][2]
        dfs.append(df / 3)
    return np.array(dfs)

def corrected_Mix(compose,w,dfs):
    total_c = np.sum(compose,1)
    F = compose.dot(dfs)
    return i_math_model(F + correct_func(total_c*0.7,w) + base_f)

def corrected_Mix_Recip(compose,w):
    base_f = math_model(data_p[0],'recip')
    F = np.zeros_like(base_f)
    total_c = 0
    for i in compose.keys():
        index = np.argwhere(name == i)[0][0]
        df = (math_model(data_p[index * 3 + 2],'recip') - base_f - correct_func(data_c[index][1],w)) / data_c[index][1]
        F += df * compose[i]
        total_c += compose[i]
    # 这里的scale原本用来防止K/A变成负数，调整后发现可以提升精度
    scale = 0.5
    return i_math_model(F + correct_func(total_c * scale,w) + base_f,'recip')

def main():
    # com = 0
    # compose = np.array([0.127,0,0,0,0,0,0,0,0,0,0,0.0748,0,0,0,0,0,0.56,0,0,0])
    w = np.load('data_w.npy').T
    dfs = get_dfs_KM(w)

    # print(np.sum((RealIngredient[com] - corrected_Mix(compose,w,dfs)) ** 2))
    # MSE of Ingredient
    # print(np.sum((RealIngredient[com] - corrected_Mix_1(RealCompose[com],w))**2))
    # print(np.sum((RealIngredient[com] - Mix(RealCompose[com]))**2))


    dataset_size = 2*30
    sum = np.random.rand(dataset_size) +0.2
    a = sum * np.random.rand(dataset_size)
    sum  -= a
    b = sum * np.random.rand(dataset_size)
    concentrations = np.concatenate((a.reshape(dataset_size,1),
                                     b.reshape(dataset_size,1),
                                     (sum - b).reshape(dataset_size,1),
                                     np.zeros((dataset_size,18))),1)
    for i in range(dataset_size):
        np.random.shuffle(concentrations[i])
    reflectance = corrected_Mix(concentrations,w,dfs)
    np.savez("dataset_Corrected_01", concentrations = concentrations, reflectance = reflectance)

    data = np.load('dataset_Corrected_01.npz')
    # print(data['concentrations'])
    # print(data['reflectance'])

main()
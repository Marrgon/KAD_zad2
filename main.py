import numpy as npy
import matplotlib.pyplot as plt


# !!!!!!! Niemalze wszystkie oznaczenia zgodne są z instrukcja do zadania 2  !!!!!!!!!

data1 = npy.genfromtxt('data1.csv', delimiter=',')
# print(data1)
data2 = npy.genfromtxt('data2.csv', delimiter=',')
data3 = npy.genfromtxt('data3.csv', delimiter=',')
data4 = npy.genfromtxt('data4.csv', delimiter=',')
data1 = npy.transpose(data1)
data2 = npy.transpose(data2)
data3 = npy.transpose(data3)
data4 = npy.transpose(data4)


# print(data1)
# print(data2)
# print(data3)
# print(data4)

def variance(x):
    var = npy.average(x ** 2) - npy.average(x) ** 2
    return var


def covariance(x, y):
    cov = npy.average((x - npy.average(x)) * (y - npy.average(y)))
    return cov


def Error(table_y, Z):
    Err = table_y - Z
    return Err


def MSE(table_y, Z):
    return npy.sum((table_y - Z) ** 2) / npy.size(table_y)


def max_dev(table_y, Z):
    return npy.max(abs(Error(table_y, Z)))


def FUV(table_y, Z):
    return variance(Error(table_y, Z)) / variance(Z)


def R2(table_y, Z):
    return 1 - FUV(table_y, Z)


# function to calculate the formula from instruction A=(X.T*X)**-1*X.T*Y

def A(x, y, noconstance=False):
    x = x.T
    y = y.T
    xtx = npy.matmul(x.T, x)

    if noconstance:
        inverse = npy.linalg.inv(npy.atleast_2d(xtx))
    else:
        inverse = npy.linalg.inv(xtx)

    xty = npy.matmul(x.T, y)

    if noconstance:
        return npy.matmul(npy.atleast_2d(inverse), npy.atleast_2d(xty))
    else:
        return npy.matmul(inverse, xty)


def data_12(x, y, name):
    a1 = A(x, y, True)
    a1 = a1.item()

    MSE1 = MSE(y, a1 * x)

    max_dev1 = max_dev(y, a1 * x)
    R21 = R2(y, a1 * x)
    a2 = (covariance(x, y) / variance(x))
    b2 = npy.average(y) - a2 * npy.average(x)
    MSE2 = MSE(y, a2 * x + b2)
    max_dev2 = max_dev(y, a2 * x + b2)
    R22 = R2(y, a2 * x + b2)

    x2 = npy.array([npy.square(x), npy.sin(x), npy.ones(x.size)]).reshape(3, x.size)
    a = A(x2, y)
    a3 = a[0]
    b3 = a[1]
    c3 = a[2]
    MSE3 = MSE(y, a3 * npy.square(x) + b2 * npy.sin(x) + c3)
    max_dev3 = max_dev(y, a3 * npy.square(x) + b3 * npy.sin(x) + c3)
    R23 = R2(y, a3 * npy.square(x) + b3 * npy.sin(x) + c3)

    plt.figure(figsize=(9, 5))
    plt.plot(x, y, 'r.', label=name)
    plt.plot(x, a1 * x, '-', label='f(x)=ax', lw=1.5, color='green')
    plt.plot(x, a2 * x + b2, '-', label='f(x)=ax+b', lw=1.5, color='orange')
    plt.plot(x, a3 * npy.square(x) + b3 * npy.sin(x) + c3, '-',
             label='f(x)=ax**2+bsin(x)+c', lw=1.5, color='purple')
    plt.legend()
    plt.suptitle(name, fontsize=22, fontdict={'position': (0.5, 0.96)})

    #plt.show()
    plt.savefig(name + '.svg')

    plt.clf()



    x121 = plt.subplot(1, 3, 1)
    x121.hist(Error(y, a1 * x), bins=20, color='green')
    x121.set_title('f(x)=ax', fontdict={'size': 22, 'position': (0.5, 1.03)})

    x122 = plt.subplot(1, 3, 2)
    x122.hist(Error(y, a2 * x + b2), bins=20, color='orange')
    x122.set_title('f(x)=ax+b', fontdict={'size': 22, 'position': (0.5, 1.03)})

    ax3 = plt.subplot(1, 3, 3)
    ax3.hist(Error(y, a3 * npy.square(x) + a3 * npy.sin(x) + c3), bins=20, color='purple')
    ax3.set_title('f(x)=ax**2+bsin(x)+c', fontdict={'size': 22, 'position': (0.5, 1.03)})

    x121.get_shared_y_axes().join(x121, x122, ax3)
    #plt.show()
    plt.savefig(name + '_hist.svg')
    plt.clf()
    return {
        '1': {
            'a': a1,
            'MSE': MSE1,
            'maxDev': max_dev1,
            'R2': R21,

        },
        '2': {
            'a': a2,
            'b': b2,
            'MSE': MSE2,
            'maxDev': max_dev2,
            'R2': R22,

        },
        '3': {
            'a': a3,
            'b': b3,
            'c': c3,
            'MSE': MSE3,
            'maxDev': max_dev3,
            'R2': R23,

        }
    }


def data_34(x1, x2, y, name):
    x341 = npy.array([x1])
    x341 = npy.insert(x341, 1, x2, axis=0)
    x341 = npy.insert(x341, 2, npy.ones(x1.size), axis=0)

    a = A(x341, y)[0]
    b = A(x341, y)[1]
    c = A(x341, y)[2]

    zz = npy.array([])
    xx, yy = npy.meshgrid(npy.arange(0, 9, 0.25), npy.arange(0, 9, 0.25))
    for i, xk in enumerate(xx):

        for j, yk in enumerate(yy):
            zz = npy.append(zz, a * xk[i] + b * yk[j] + c)
    zz = zz.reshape(npy.size(xx, 0), npy.size(yy, 0))

    Z34 = npy.array([])
    for xs, ys in zip(x1, x2):
        Z34 = npy.append(Z34, a * xs + b * ys + c)

    MSE1 = MSE(y, Z34)
    max_err = max_dev(y, Z34)
    R21 = R2(y, Z34)
    x_x = npy.array([])
    x_x = npy.append(x_x, [x1])
    x_x = npy.append(x_x, [x2]).reshape(2, x1.size)

    x_x = npy.prod(x_x, axis=0)

    finalx = npy.array([])
    finalx = npy.append(finalx, x1 ** 2)
    finalx = npy.append(finalx, x_x)
    finalx = npy.append(finalx, x2 ** 2)
    finalx = npy.append(finalx, x1)
    finalx = npy.append(finalx, x2)
    finalx = npy.append(finalx, npy.ones(x1.size)).reshape(6, x1.size)

    a2 = A(finalx, y)[0]
    b2 = A(finalx, y)[1]
    c2 = A(finalx, y)[2]
    d2 = A(finalx, y)[3]
    e2 = A(finalx, y)[4]
    f2 = A(finalx, y)[5]
    Z342 = npy.array([])
    for xk, yk in zip(x1, x2):
        Z342 = npy.append(Z342, a2 * xk ** 2 + b2 * xk * yk +
                          c2 * yk ** 2 + d2 * xk + e2 * yk + f2)
    MSE2 = MSE(y, Z342)
    max_err2 = max_dev(y, Z342)
    R22 = R2(y, Z342)
    zz2 = npy.array([])

    for i, xs in enumerate(xx):

        for j, ys in enumerate(yy):
            zz2 = npy.append(zz2, a2 * xs[i] ** 2 + b2 * xs[i] * ys[j] +
                             c2 * ys[j] ** 2 + d2 * xs[i] + e2 * ys[j] + f2)

    zz2 = zz2.reshape(npy.size(xx, 0), npy.size(yy, 0))

    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    bx = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_wireframe(xx, zz, yy, color='blue', rstride=2, cstride=2,
                      label='f(X1, X2) = '
                            'a * X1 + b * X2 + c')
    ax.scatter(x2, y, x1, color='red', marker='o', label=name)
    ax.set_xlabel('X1', fontsize=22)
    ax.set_ylabel('X2', fontsize=22)
    ax.set_zlabel('f(X1, X2)', fontsize=22)
    ax.legend(fontsize=8)
    ax.view_init(40, -75)

    bx.plot_wireframe(xx, zz2, yy, color='green', rstride=2, cstride=2,
                      label='f(X1, X2) = a * X1**2 + '
                            'b * X1*X2 + c * X2**2 + '
                            'd * X1 + e * X2 + f')
    bx.scatter(x2, y, x1, color='red', marker='o', label=name)
    bx.set_xlabel('X1', fontsize=22)
    bx.set_ylabel('X2', fontsize=22)
    bx.set_zlabel('f(X1, X2)', fontsize=22)
    bx.legend(fontsize=8)
    bx.view_init(30, -70)
    #fig.show()
    fig.savefig(name + '.svg')
    fig.clf()

    plt.figure(figsize=(9, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ax1.hist(Error(y, Z34), bins=20, color='blue')
    ax1.set_title('f(X1, X2) = a * X1 + b * X2 + c',
                  fontdict={'size': 10, 'position': (0.5, 1.03)})

    ax2.hist(Error(y, Z342), bins=20, color='green')
    ax2.set_title('f(X1, X2) = a * X1**2 + '
                  'b * X1*X2 + c * X2**2 + \n'
                  'd * X1 + e * X2 + f',
                  fontdict={'size': 10, 'position': (0.5, 1.03)})

    ax1.get_shared_y_axes().join(ax1, ax2)
    #plt.show()
    plt.savefig(name + '_hist.svg')
    fig.clf()

    return {
        '1': {
            'a': a,
            'b': b,
            'c': c,
            'MSE': MSE1,
            'maxDev': max_err,
            'R2': R21,

        },
        '2': {
            'a': a2,
            'b': b2,
            'c': c2,
            'd': d2,
            'e': e2,
            'f': f2,
            'MSE': MSE2,
            'maxDev': max_err2,
            'R2': R22,

        }
    }


result_1 = data_12(data1[0], data1[1], 'data_1')
result_2 = data_12(data2[0], data2[1], 'data_2')
result_3 = data_34(data3[0], data3[1], data3[2], 'data_3')
result_4 = data_34(data4[0], data4[1], data4[2], 'data_4')


def showResult(result):
    print("Model 1: " + "f(X) = " + "%.2f" % result['1']['a'] + 'X')
    print("MSE: " + "%.3f" % result['1']['MSE'])
    print("Max Dev:" + "%.2f" % result['1']['maxDev'])
    print("R2: " + "%.2f" % result['1']['R2'] + '\n')

    print("Model 2: " + "f(X) = " + "%.2f" % result['2']['a'] +
          'X + ' + "%.2f" % result['2']['b'])
    print("MSE: " + "%.3f" % result['2']['MSE'])
    print("Max Dev:" + "%.2f" % result['2']['maxDev'])
    print("R2: " + "%.2f" % result['2']['R2'] + '\n')

    print("Model 3: " + "f(X) = " + "%.2f" % result['3']['a'] +
          'X**2 + ' + "%.2f" % result['3']['b'] +
          ' * sin(X) + ' + "%.2f" % result['3']['c'])
    print("MSE: " + "%.3f" % result['3']['MSE'])
    print("Max Dev:" + "%.2f" % result['3']['maxDev'])
    print("R2: " + "%.2f" % result['3']['R2'] + '\n')


def showResult2(result):
    print("Model 1: " + "f(X1,X2) = " + "%.2f" % result['1']['a'] +
          'X1 + ' + "%.2f" % result['1']['b'] +
          'X2 + ' + "%.2f" % result['1']['c'])
    print("MSE: " + "%.3f" % result['1']['MSE'])
    print("Max Dev: " + "%.2f" % result['1']['maxDev'])
    print("R2: " + "%.2f" % result['1']['R2'] + '\n')

    print("Model 2: " + "f(X1,X2) = " + "%.2f" % result['2']['a'] +
          'X1**2 + ' + "%.2f" % result['2']['b'] +
          'X1 * X2 + ' + "%.2f" % result['2']['c'] +
          'X2**2 + ' + "%.2f" % result['2']['d'] +
          'X1 + ' + "%.2f" % result['2']['e'] +
          'X2 + ' + "%.2f" % result['2']['f'])
    print("MSE: " + "%.3f" % result['2']['MSE'])
    print("Max Dev: " + "%.2f" % result['2']['maxDev'])
    print("R2: " + "%.2f" % result['2']['R2'])


print("-------- Data Set 1 ----------")
showResult(result_1)
print("-------- Data Set 2 ----------")
showResult(result_2)
print("-------- Data Set 3 ----------")
showResult2(result_3)
print("-------- Data Set 4 ----------")
showResult2(result_4)

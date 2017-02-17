import numpy as np
import matplotlib.pyplot as plt


# Assigning X and y from the dataset
data = np.loadtxt('datapath', delimiter=',')
rows=(data.size)/2
X = np.array((data[:, 0])).reshape(rows, 1)
y = np.array(data[:, 1]).reshape(rows, 1)
m = np.size(X)
X = np.insert(X, 0, values=1, axis=1)
t = np.ones(shape=[2, 1])

def linReg():
    h = np.dot(X,t)
    J = 1/(2*m) * sum((h - y)**2)
    print('Cost:',J)
    print("Error:",h-y)
    for i in range(1,2000):
        h = np.dot(X,t)
        t[0] = t[0] - 0.01*(1/m)*(np.dot((X[:,0]),(h-y)))
        t[1] = t[1] - 0.01*(1/m)*(np.dot((X[:,1]),(h-y)))
        J = 1/(2*m) * sum((h - y)**2)
        print(i)
        print('Cost:', J)
    plt.scatter(X[:,1],y,color= 'blue')
    plt.plot(X[:,1],h)
    return t

def predict(newval):
    W = linReg()
    predValue = np.dot(newval,W[1]) + W[0]
    print("Predicted Value:-",predValue)
    plt.plot(newval, predValue)
    plt.scatter(newval, predValue, color='red')
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.show()

print("Enter the number to be predicted:-")
nv = input()
nv = float(nv)
predict(nv)


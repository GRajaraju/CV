import numpy as np

X = np.array([[1,0,1],
             [0,1,0],
             [1,1,1],
             [0,0,1],
                  ])

y = np.array([[0,1],
              [1,0],
              [0,0],
              [1,1],
                 ])

w1 = np.random.randn(3,40)
w2 = np.random.randn(40,2)

def sigmoid(z,deriv):
    output = 1/(1+np.exp(-z))
    if deriv == False:
        return output
    if deriv == True:
        return output * (1 - output)

for iteration in range(2000):
    print("Iteration:" +str(iteration))
    for row_index in range(len(X)):
        input = X[row_index]
        input = np.reshape(input,newshape=[1,3])
        goal_prediction = y[row_index]
        z2 = np.dot(input,w1)
        a2 = sigmoid(z2,False)
        z3 = np.dot(a2,w2)
        a3 = sigmoid(z3,False)
        print("Predicted:", a3)
        delta3 = np.multiply(-(goal_prediction-a3),sigmoid(z3,True))
        dJdW2 = np.dot(a2.T,delta3)
        delta2 = np.dot(delta3,w2.T) * sigmoid(z2,True)
        dJdW1 = np.dot(input.T,delta2)
        w1 = w1 - (0.1*dJdW1)
        w2 = w2 - (0.1*dJdW2)


x1 = np.array([[0,0,0]])
print("new prediction...")
newval = sigmoid(np.dot((sigmoid(np.dot(x1,w1),False)),w2),False)
print("Predicted:",newval,"for",x1)

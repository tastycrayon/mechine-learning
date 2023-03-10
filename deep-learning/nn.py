import numpy as np
x = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

x = x/np.amax(x, axis=0)
y = y/100

# synapse, x1 = input * weight
# synapse only multiplies
# neurons add all multiplied values
# neurons, Σx = x1 + x2 + x3
# neurons activation, a = 1/(1+e^-z)

print(x)
print(y)


def sigmoid(z):
    # Apply sigmoid activation function
    return 1/(1+np.exp(-z))


print(sigmoid(1))


class Neural_Netowrk(object):
    def __init(self):
        # Define HyperParameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

    def forwardPropagate():
        return ''

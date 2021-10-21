import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display


## n-hidden layer NETWORK ##
Nin = 2      # input layer neurons
Nout = 1    # output layer neurons
Nh = 10     # hidden layer neurons
n = 3       # number of hidden layers

## ////////////// MAIN ///////////// ## 

def main():
    #apply_net(y_in)
    #getSingleLayerResults()
    #makeRandomImg()

    
    inputLayer = Layer(Nin, Nh, y_in)
    lastHiddenLayer = applyHiddenLayers(inputLayer, 3)    ##Applies n hidden layers
    finalLayer = Layer(len(lastHiddenLayer.y_out), Nout, lastHiddenLayer.y_out) ##Calculates output of final layer with out_size
    print(finalLayer.y_out)

## Randomize weight, bias and test inputs ##

## ////////////// WEIGHTS ////////////// ##
# Input layer weights Ni x Nh array (fully connected to first hidden layer)
w_in = np.random.uniform(low = -10, high = 10, size = (Nin, Nh))  ## Input -> first hidden layer weights
w_out = np.random.uniform(low = -10, high = 10, size=(Nh, Nout)) ## LastHidden -> output layer weights

## Hidden weights:


# BIAS: N1 vector #
b = np.random.uniform(low = -1, high = 1, size = Nout)

## ////////////////////// INPUT ///////////////////////////// ##
#  N0 vector
y_in = ([0.2, 0.4])
## Image MxM pixels
M = 50 # num pixels
#y_in = np.zeros([M, M]) ##Input
y_out = np.zeros([M, M]) ##Output

def makeRandomImg():
    
    for j1 in range(M):
        for j2 in range(M):
            value0=float(j1)/M-0.5
            value1=float(j2)/M-0.5
            y_out[j1, j2] = apply_net([value0, value1])
    
    plt.imshow(y_out,origin='lower',extent=(-0.5,0.5,-0.5,0.5))
    plt.colorbar()
    plt.title("NN output as a function of input values")
    plt.xlabel("y_2")
    plt.ylabel("y_1")
    plt.show()


## ////////////////////// NETWORK //////////////////////////// ##
##Defines activation function and derivative to be used
def f(z):
    ##Sigmoid
    return (1/(1+np.exp(-z)))

def df(z):
    ##First derivative df/dz
    return (f(z)/(1+np.exp(-z)))


def applyLayer(layer):
    print(layer.y_in)
    print(layer.w)

    z = np.dot(layer.y_in, layer.w) + layer.b

    print("layer applied")
    return activation_fn(z)

##Apply n hidden layers to the starting layer  
def applyHiddenLayers(startLayer, n):
    
    currentLayer = startLayer ##Layer object contains output of that layer as layer.y_out 
    for i in range(n):
        nextInput = currentLayer.y_out
        currentLayer = Layer(Nh, Nh, nextInput)
        
    lastLayer = currentLayer ##Final layer computed 
    return lastLayer

def getSingleLayerResults():
    print("network input y_in:", y_in)
    print("weights w:", w)
    print("bias vector b:", b)
    print("linear superposition z:", z)
    print("network output y_out:", y_out)

## //////////////// LAYER OBJECT ///////////////// ##
class Layer :
    def __init__(self, Nin, Nout, y_in):
        self.Nin = len(y_in) #Number of input neurons in layer (must equal size of y_in)
        self.Nout = Nout #Number of output neurons
        self.w = np.random.uniform(low = -10, high = 10, size = (Nin, Nout)) #Randomize layer weights 
        self.b = np.random.uniform(low = -1, high = 1, size = Nout)
        self.y_in = y_in
        self.y_out = applyLayer(self)
    ##Overload for constructing with w, b
    @classmethod
    def withWandB(self, Nin, Nout, y_in, w, b):
        self.Nin = len(y_in) #Number of input neurons in layer (must equal size of y_in)
        self.Nout = Nout #Number of output neurons
        self.w = w 
        self.b = b
        self.y_in = y_in
        self.y_out = applyLayer(y_in, w, b)
  







if(__name__ == "__main__"):
    main()

    
    
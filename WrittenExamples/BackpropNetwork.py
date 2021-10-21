from sys import prefix
from matplotlib.image import composite_images
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

## ///////////// GLOBAL VARIABLES ///////////// ##

##global y_N                         ## Array of outputs for each layer
##global df_N                        ## Array of df/dz at each layer

global NumLayers                     ## Number of layers to apply (excluding input, includes final output layer)
global Layers                        ## Array containing all layers
global LayerSizes                    ## Sizes of each layer, first is input layer
global Weights  
global Bias 
global batchsize
global y_target                      ## Target values the algorithm is trying to predict
global inputLayer
global eta                           ## Learning rate
global batches                       ## Num of batches
global costs                         ## Cost functions

##Layer object, constructed from an input array of neurons and a specified layer index (for setting sizes etc)
class Layer :
    ## Layer constructor
    def __init__(self, y_in, N) :
        self.y_in = y_in                ## Input neurons for this layer
        self.N = N                      ## Layer index (input is 0)
        self.size = LayerSizes[N]       ## Size of the layer
        self.NumIn = len(y_in)          ## Number of input neurons
        self.NumOut = LayerSizes[N]     ## Number of output neurons
        self.y_out = 0                  ## Output neurons               (set on forward pass)
        self.dActivation_dz = 0         ## df/dz values                 (set on forward pass)
        if(N < NumLayers):
            self.weight = Weights[N]       ## Weight matrix for this layer (set by backprop)
            self.bias = Bias[N]        ## Bias for this layer          (set by backprop)
        else:
            self.weight = 0
            self.bias = 0

        self.delta = 0                  ## Delta value                  (set on backprop pass)
        self.dCost_dWeight = 0          ## dCost/dWeight                (set on backprop pass)
        self.dCost_dBias = 0            ## dCost/dBias                  (set on backprop pass)
        Layers.append(self)             ##Add layer to list of layers
        



## /////////////// VALUES ////////////// ##
NumLayers = 3
Layers = list()        ## NumLayers + 1 total layers (including input)
LayerSizes = [2, 20, 30, 1]    ## Layer sizes
#Weights = [NumLayers + 1]
#Bias = [NumLayers + 1]
batchsize = 100
batches = 100
eta = 0.001


# initialize random weights and biases for all layers (except input of course)
Weights=[np.random.uniform(low=-1,high=+1,size=[ LayerSizes[j],LayerSizes[j+1] ]) for j in range(NumLayers)]
Bias=[np.random.uniform(low=-1,high=+1,size=LayerSizes[j+1]) for j in range(NumLayers)]


## Input values for first layer
y_in = np.random.uniform(low = -1, high = +1, size = [batchsize, LayerSizes[0]])
## Target values the network is trying to predict
y_target = np.random.uniform(low = -1, high = +1, size = [batchsize, LayerSizes[-1]]) 
## Defines the input layer with index N = 0
inputLayer = Layer(y_in, 0)
Layers.append(inputLayer)


##Defines activation function and derivative to be used
def activation(z) :
    ##Sigmoid
    return (1/(1+np.exp(-z)))

def activation_derivative(z):
    ##First derivative activation fn
    return (activation(z)/(1+np.exp(-z)))





def main() :
    print("running")
    costs = np.zeros(batches)
    for k in range(batches):
        costs[k] = train_net(inputLayer, y_target, eta)
    
    plt.plot(costs)
    plt.show()


            
## Single forward step of network 
## Sets out value on current layer and creates next layer using output of current layer                 
def forward_step(layer) :
    z = np.dot(layer.y_in, layer.weight)
    layer.y_out = activation(z)
    layer.df = activation_derivative(z)
    nextLayer = Layer(layer.y_out, layer.N + 1) ## Create next layer with index N + 1
    return nextLayer
##Perform a single backward step on the passed layer
def backward_step(delta, weight , dActivation_dz) :
    return ( np.dot(delta, np.transpose(weight))*dActivation_dz)

def backprop(finalLayer) :
    print(Layers)
    ## Calculate the delta vector for this layer
    finalLayer.delta = (finalLayer.y_out - y_target) * finalLayer.dActivation_dz
    #if(N<)
    prevLayer = Layers[finalLayer.N - 1]
    finalLayer.dCost_dWeight = np.dot(np.transpose(prevLayer.y_out), finalLayer.delta)/batchsize
    finalLayer.dCost_dBias = finalLayer.delta.sum(0)/batchsize

    for j in range(NumLayers - 1):
        currentLayer = Layers[-1-j]
        previousLayer = Layers[-2-j]
        ##Computes the backward step for the current layer, using the current delta and layer weights
        delta = backward_step(currentLayer.delta, currentLayer.weight, previousLayer.dActivation_dz)
        previousLayer.dCost_dWeight = np.dot(np.transpose(Layers[-2-j].y_out), delta)
        previousLayer.dCost_dBias = delta.sum(0)/batchsize

def gradient_step(eta): # update weights & biases (after backprop!)
    #global dw_layer, db_layer, Weights, Biases
    
    for j in range(NumLayers):
        Weights[j]-=eta*Layers[j].dCost_dWeight
        Bias[j]-=eta*Layers[j].dCost_dBias



#global y_out_result


##Return the cost function
def train_net(inputLayer, y_target, eta):
    finalLayer = apply_net(inputLayer)
    backprop(finalLayer)
    gradient_step(eta)
    cost=((y_target-finalLayer.y_out)**2).sum()/batchsize
    return(cost)


##Apply the net to all layers
def apply_net(inputLayer) :
    currentLayer = inputLayer        ## Start with input layer
    
    ##Apply the net at each layer
    for i in range(NumLayers):
        print(currentLayer.N)
        ## i = 0 is first layer after the input layer 
        nextLayer = forward_step(currentLayer) 
        currentLayer = nextLayer
    ##Return the final layer
    return currentLayer

if(__name__ == "__main__"):
    main()



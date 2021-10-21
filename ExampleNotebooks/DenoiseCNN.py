# keras: Sequential is the neural-network class, Dense is
# the standard network layer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers # to choose more advanced optimizers like 'adam'

from tqdm import tqdm # progress bar

import numpy as np

import matplotlib.pyplot as plt # for plotting
import matplotlib
matplotlib.rcParams['figure.dpi']=300 # highres display

# for updating display 
# (very simple animation)
from IPython.display import clear_output
from time import sleep

from perlin_noise import PerlinNoise

## Used to generate random circles

def my_generator2D(batchsize,x,y):
    ##Random radius
    R=np.random.uniform(low=0.2,high=1,size=batchsize)
    ##Random origin
    x0=np.random.uniform(size=batchsize,low=-0.8,high=0.8)
    y0=np.random.uniform(size=batchsize,low=-0.8,high=0.8)

    
    ##IsCircle=(np.random.uniform(size=batchsize)<0.5)*1.0 # Circle? (==1) or Square?
    Circles=1.0*((x[None,:]-x0[:,None])**2 + (y[None,:]-y0[:,None])**2 < R[:,None]**2)
    #Squares=1.0*(np.abs(x[None,:]-x0[:,None])<R[:,None])*(np.abs(y[None,:]-y0[:,None])<R[:,None])
    input = Circles  #(1-IsCircle[:,None])*Squares
    noisyInput = addNoise(inputLayer)
    
    resultLayer = np.zeros([batchsize,1])
    resultLayer[:,0] = IsCircle
    #resultLayer[:,1]=1-IsCircle
    return( inputLayer, resultLayer )

def addNoise(image):
    xpix = len(x)
    ypix = len(y)
    noiseGen = PerlinNoise(octaves = 3, seed = 1)
    noise = [[noiseGen([i/xpix, j/ypix]) for j in range(xpix)] for i in range(ypix)]
    return image + noise



Net = Sequential()




batchsize=20
steps=1000

vals=np.linspace(-1,1,N)
X,Y=np.meshgrid(vals,vals)
x,y=X.flatten(),Y.flatten() # make 1D arrays, as needed for dense layers!

costs=np.zeros(steps)
accuracy=np.zeros(steps)
skipsteps=10

for j in range(steps):
    y_in,y_target=my_generator2D(batchsize,x,y)
    costs[j],accuracy[j]=Net.train_on_batch(y_in, y_target)
    if j%skipsteps==0:
        clear_output(wait=True)
        plt.plot(costs,color="darkblue",label="cost")
        plt.plot(accuracy,color="orange",label="accuracy")
        plt.legend()
        plt.show()
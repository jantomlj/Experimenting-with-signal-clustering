import matplotlib.pyplot as plt
import numpy as np

def draw_reconstructions(ins, outs, shape_in):
    plt.figure(figsize=(8, 12*4))
    for i in range(10):

        plt.subplot(10, 4, 4*i + 1)
        plt.imshow(ins[i].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.subplot(10, 4, 4*i + 2)
        plt.imshow(outs[i][0:784].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
    plt.tight_layout()
    plt.show()
    
def plot_reconstructions(ins, outs):
    plt.figure(figsize=(8, 20))
    for i in range(5):

        plt.subplot(5, 2, (i+1)*2 - 1)
        plt.plot(np.linspace(0, 1, num=len(ins[i])), ins[i])
        #plt.title("Test input")
        #plt.subplot(5, 2, (i+1)*2)
        plt.plot(np.linspace(0, 1, num=len(outs[i])), outs[i])
        #plt.title("Reconstruction")
    plt.tight_layout()
    plt.show()

def scatterTsne(results, y, cs='bgrcmykw'):
    colors = np.array([x for x in cs])
    plt.figure(figsize=(10, 8))
    plt.scatter(results[:, 0], results[:, 1], color=colors[y])
    plt.show()

def draw_graph(ins, shape_in):
    plt.figure(figsize=(8, 12*4))
    for i in range(10):
        plt.subplot(10, 4, 4*i + 1)
        plt.imshow(ins[i].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
    plt.tight_layout()
    plt.show()
    
def plotCoverage(coverage):
    x = np.linspace(0, 1, num=len(coverage))
    plt.plot(x, coverage)
    plt.show()
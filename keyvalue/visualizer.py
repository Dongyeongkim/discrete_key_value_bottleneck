import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Referenced from https://stats.stackexchange.com/questions/71335/decision-boundary-plot-for-a-perceptron/71339#71339

def visualizer(inp, lab, model, model_type, leng):
    colorbook = ['mediumseagreen','yellowgreen','palegoldenrod','coral','grey','yellow','violet','grey']
    
    typ = int(model_type[leng:])
    X = inp.cpu().numpy()
    Y = lab.cpu().numpy()
    
    colorcode = []
    for i in Y:
        colorcode.append(colorbook[i])
    cm = LinearSegmentedColormap.from_list('colorcode', colorcode)

    h = .02  # step size in the mesh
    x_min, x_max = 0, 8
    y_min, y_max = 0, 8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    positions = torch.Tensor(np.vstack(list(zip(xx.ravel(), yy.ravel())))).cuda()
    

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = model(positions)
    Z = Z.detach()
    Z = Z.cpu().numpy()
    Z = np.argmax(Z, axis=1)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z/10, cmap=cm)
    ax.axis('off')
    
    
    # Plot also the training points
    ax.scatter(X[:, 0], X[:, 1], c=colorcode, cmap=plt.cm.Paired, edgecolors='black')
    ax.set_title(model_type)
    fig.savefig(model_type+".png")
    plt.close()


def visualizer_without_scat(inp, lab, model, model_type, leng):
    colorbook = ['mediumseagreen','yellowgreen','palegoldenrod','coral','grey','yellow','violet','grey']
    
    typ = int(model_type[leng:])
    X = inp.cpu().numpy()
    Y = lab.cpu().numpy()
    
    colorcode = []
    for i in Y:
        colorcode.append(colorbook[i])
    cm = LinearSegmentedColormap.from_list('colorcode', colorcode)

    h = .02  # step size in the mesh
    x_min, x_max = 0, 8
    y_min, y_max = 0, 8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    positions = torch.Tensor(np.vstack(list(zip(xx.ravel(), yy.ravel())))).cuda()
    

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = model(positions)
    Z = Z.detach()
    Z = Z.cpu().numpy()
    Z = np.argmax(Z, axis=1)

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z/10, cmap=cm)
    ax.axis('off')
    
    ax.set_title(model_type)
    fig.savefig(model_type+".png")
    plt.close()

def visualizer_VQ(inp, lab, model, model_type, leng):
    colorbook = ['mediumseagreen','yellowgreen','palegoldenrod','coral','grey','yellow','violet','grey']
    
    typ = int(model_type[leng:])
    X = inp.cpu().numpy()
    Y = lab.cpu().numpy()
    
    colorcode = []
    for i in Y:
        colorcode.append(colorbook[i])
    cm = LinearSegmentedColormap.from_list('colorcode', colorcode)

    h = .02  # step size in the mesh
    x_min, x_max = 0, 8
    y_min, y_max = 0, 8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    positions = torch.Tensor(np.vstack(list(zip(xx.ravel(), yy.ravel())))).cuda()
    

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = model(positions)
    Z = Z[1].cpu().detach()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm)
    ax.axis('off')
    
    
    # Plot also the training points
    ax.scatter(X[:, 0], X[:, 1], c=colorcode, cmap=plt.cm.Paired, edgecolors='black')
    ax.set_title(model_type)
    fig.savefig(model_type+".png")
    plt.close()

def visualizer_without_scat_VQ(inp, lab, model, model_type, leng):
    colorbook = ['mediumseagreen','yellowgreen','palegoldenrod','coral','grey','yellow','violet','grey']
    
    typ = int(model_type[leng:])
    X = inp.cpu().numpy()
    Y = lab.cpu().numpy()
    
    colorcode = []
    for i in Y:
        colorcode.append(colorbook[i])
    cm = LinearSegmentedColormap.from_list('colorcode', colorcode)

    h = .02  # step size in the mesh
    x_min, x_max = 0, 8
    y_min, y_max = 0, 8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    positions = torch.Tensor(np.vstack(list(zip(xx.ravel(), yy.ravel())))).cuda()
    

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = model(positions)
    Z = Z[1].cpu().detach()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=cm)
    ax.axis('off')
    
    ax.set_title(model_type)
    fig.savefig(model_type+".png")
    plt.close()


def visualizer_KV(inp, lab, model, model_type, leng):
    colorbook = ['mediumseagreen','yellowgreen','palegoldenrod','coral','grey','yellow','violet','grey']
    
    typ = int(model_type[leng:])
    X = inp.cpu().numpy()
    Y = lab.cpu().numpy()
    
    colorcode = []
    for i in Y:
        colorcode.append(colorbook[i])
    cm = LinearSegmentedColormap.from_list('colorcode', colorcode)

    h = .02  # step size in the mesh
    x_min, x_max = 0, 8
    y_min, y_max = 0, 8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    positions = torch.Tensor(np.vstack(list(zip(xx.ravel(), yy.ravel())))).cuda()
    

    # Plot the decision boundary. For that, we will assign a color to each

    fig, ax = plt.subplots()
    Z = model(positions)
    Z = Z.cpu().detach()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=cm)
    ax.axis('off')
    
    # Plot also the training points
    ax.scatter(X[:, 0], X[:, 1], c=colorcode, cmap=cm, edgecolors='black')
    ax.set_title(model_type)
    fig.savefig(model_type+".png")
    plt.close()




def visualizer_without_scat_KV(inp, lab, model, model_type, leng):
    colorbook = ['mediumseagreen','yellowgreen','palegoldenrod','coral','grey','yellow','violet','grey']
    
    typ = int(model_type[leng:])
    X = inp.cpu().numpy()
    Y = lab.cpu().numpy()
    
    colorcode = []
    for i in Y:
        colorcode.append(colorbook[i])
    cm = LinearSegmentedColormap.from_list('colorcode', colorcode)

    
    h = .02  # step size in the mesh
    x_min, x_max = 0, 8
    y_min, y_max = 0, 8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    positions = torch.Tensor(np.vstack(list(zip(xx.ravel(), yy.ravel())))).cuda()
    

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    

    # Put the result into a color plot
    Z = model(positions)
    Z = Z.cpu().detach()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    ax.axis('off')
    
    ax.set_title(model_type)
    fig.savefig(model_type+".png")
    plt.close()
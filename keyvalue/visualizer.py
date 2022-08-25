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
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).cuda())
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
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).cuda())
    Z = Z.detach()
    Z = Z.cpu().numpy()
    Z = np.argmax(Z, axis=1)
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z/10, cmap=cm)
    ax.axis('off')
    
    ax.set_title(model_type)
    fig.savefig(model_type+".png")

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
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).cuda())
    Z = Z[1]
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
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).cuda())
    Z = Z[1]
    Z = Z.detach()
    Z = Z.cpu().numpy()
    Z = np.argmax(Z, axis=1)
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z/10, cmap=cm)
    ax.axis('off')
    
    ax.set_title(model_type)
    fig.savefig(model_type+".png")


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
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).cuda())
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
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    fig, ax = plt.subplots()
    Z = model(torch.Tensor(np.c_[xx.ravel(), yy.ravel()]).cuda())
    Z = Z.detach()
    Z = Z.cpu().numpy()
    Z = np.argmax(Z, axis=1)
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z/10, cmap=cm)
    ax.axis('off')
    
    ax.set_title(model_type)
    fig.savefig(model_type+".png")
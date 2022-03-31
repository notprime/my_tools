""" ------------------- """
"""                     """
""" Just some tools  :D """
"""                     """
""" ------------------- """

# Packages
import torch
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from IPython import display



""" Matplotlib Snippet """
# for sharper images
def use_svg_display():
    # use the svg format to display a plot in the notebook
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    # set the figure size for matplotlib
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    # set the axes for matplotlib
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None,
         legend=None, xlim=None, ylim=None,
         xscale='linear', yscale='linear',
         fmts=('-', '--m', '-.g', ':r'),
         figsize=(3.5, 2.5), axes=None):
    """ plot data points """
    # if legend is None:
    #  legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # return true if X (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))  # prendo singolo elemento di lista

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    
    
""" Timer class """
class Timer:
  """ record multiple running times """
  def __init__(self):
    self.times = []
    self.start()

  def start(self):
    """ start the timer """
    self.tik = time.time()

  def stop(self):
    """ stop the timer and record the time in a list """
    self.times.append(time.time() - self.tik)
    return self.times[-1]

  def avg(self):
    """ return the average time """
    return sum(self.times) / len(self.times)
  
  def cumsum(self):
    """ return the accumulated time """
    return np.array(self.times).cumsum().tolist()

""" Fashion MNIST Labels """
def get_fashion_mnist_labels(labels):
"""Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

""" Show some Images """
def show_images(imgs, num_rows, num_cols, titles = None, scale = 1.5):
  figsize = (num_cols * scale, num_rows * scale)
  _, axes = plt.subplots(num_rows, num_cols, figsize = figsize)
  axes = axes.flatten()
  for i, (ax, img) in enumerate(zip(axes, imgs)):
    if torch.is_tensor(img):
      # Tensor Image
      ax.imshow(img.numpy())
    else:
      # PIL Image
      ax.imshow(img)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    if titles:
      ax.set_title(titles[i])
  return axes

"""Fashion MNIST Dataset and DataLoader """
def load_data_fashion_mnist(batch_size, resize = None):
  # download the Fashion-MNIST dataset and load it into memory
  trans = [transforms.ToTensor()]
  if resize:
    trans.insert(0, transforms.Resize(resize))
  trans = transforms.Compose(trans)
  mnist_train = torchvision.datasets.FashionMNIST(
      root = '../data', train = True, transform = trans,
      download = True)
  mnist_test = torchvision.datasets.FashionMNIST(
      root = '../data', train = False, transform = trans,
      download = True)
  return (data.DataLoader(mnist_train, batch_size = batch_size,
                          shuffle = True, num_workers = WORKERS),
          data.DataLoader(mnist_test, batch_size = batch_size,
                          shuffle = False, num_workers = WORKERS))

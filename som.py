# Self Organizing Maps


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')

# iloc gets indexes of observation, all the lines we use : and all the columns except 
# last so :-1 and values
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
\
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

# fit sc object to X so sc gets all info (min and max) and all info for normalization
# apply normalization to X, fit method returns normalized version of X
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)

# randomly initialize the weight vectors to small numbers close to 0
som.random_weights_init(X)

# train som on X, matrix of features and patterns recognized
som.train_random(data = X, num_iteration = 100)

# Visualising the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

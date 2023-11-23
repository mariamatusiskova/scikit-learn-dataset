from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(0)
feature_set_x, labels_y = datasets.make_moons(100, noise=0.10)
X_train, X_test, y_train, y_tespusht = train_test_split(feature_set_x, labels_y, test_size=0.33, random_state=42)
# 123
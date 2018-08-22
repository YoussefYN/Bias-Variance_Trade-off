import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def get_y_third_order_function(x):
    return 10 + x + (x - 4)**2 + (x - 3)**3 + np.random.normal()


data_x = np.random.uniform(0, 5, 200)
data_y = np.vectorize(get_y_third_order_function)(data_x)
plt.plot(data_x, data_y, "r.")
plt.show()

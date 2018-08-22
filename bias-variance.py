import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


def get_y_third_order_function(x):
    return 10 + x + (x - 4)**2 + (x - 3)**3 + np.random.normal() * 2


data_x = np.random.uniform(0, 5, 300)
data_y = np.vectorize(get_y_third_order_function)(data_x)
data_x = [[x] for x in data_x]
plt.plot(data_x, data_y, "r.")
plt.show()


def model(degree, X, y):
    poly = PolynomialFeatures(degree=degree)
    high_degree_X = poly.fit_transform(X)

    regression = linear_model.LinearRegression()
    regression.fit(high_degree_X, y)
    predicted_y = regression.predict(high_degree_X)

    print("Report for model of Degree: %d" %degree)
    print("Mean squared error: %.3f" % mean_squared_error(y, predicted_y))
    print('Variance score: %.3f' % r2_score(y, predicted_y))
    print("Bias: %.3f\n\n" % regression.intercept_)
    plt.plot(X, predicted_y, "b.")
    plt.plot(X, y, "r.")
    plt.show()


model(1, data_x, data_y)
model(2, data_x, data_y)
model(3, data_x, data_y)
model(12, data_x, data_y)

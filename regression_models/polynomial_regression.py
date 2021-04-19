from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def polynomial_regression(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    poly_f = PolynomialFeatures(degree=4)
    x_train = poly_f.fit_transform(x_train)
    x_test = poly_f.fit_transform(x_test)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    print('multiple linear regression r_squared:')
    print(r2_score(y_test, y_pred))
    print('multiple linear regression RMSE:')
    print(mean_squared_error(y_test, y_pred, squared=False))

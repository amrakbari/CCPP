from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def multiple_linear_regression(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    print('multiple linear regression r_squared:')
    print(r2_score(y_test, y_pred))
    print('multiple linear regression RMSE:')
    print(mean_squared_error(y_test, y_pred, squared=False))





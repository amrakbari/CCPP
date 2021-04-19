from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

def dtr(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    print('DTR r_squared:')
    print(r2_score(y_test, y_pred))
    print('DTR RMSE:')
    print(mean_squared_error(y_test, y_pred, squared=False))

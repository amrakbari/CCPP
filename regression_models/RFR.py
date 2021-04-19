from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def rfr(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    print('RFR r_squared:')
    print(r2_score(y_test, y_pred))
    print('RFR RMSE:')
    print(mean_squared_error(y_test, y_pred, squared=False))
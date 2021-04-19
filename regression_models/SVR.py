from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

def svr(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_train_SC = StandardScaler()
    x_test_SC = StandardScaler()
    y_test_SC = StandardScaler()
    y_train_SC = StandardScaler()
    x_test = x_test_SC.fit_transform(x_test)
    x_train = x_train_SC.fit_transform(x_train)
    y_test = y_test_SC.fit_transform(y_test.reshape(len(y_test), 1))
    y_train = y_train_SC.fit_transform(y_train.reshape(len(y_train), 1))
    regressor = SVR(kernel='rbf')
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    print('SVR r_squared:')
    print(r2_score(y_test, y_pred))
    print('SVR RMSE:')
    print(mean_squared_error(y_test, y_pred, squared=False))

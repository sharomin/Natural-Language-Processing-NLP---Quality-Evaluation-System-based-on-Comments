from data_process import load_data
import score
import learning_rate as lr
import linear_regression as classifier

X_train, y_train, X_val, y_val, X_test, y_test = load_data('154801595314')


mse = classifier.LinearRegressionMSE()
mse.fit(X_train, y_train)
y_pred = mse.pred(X_val)
print("Error (MSE)", score.mse(y_val, y_pred))

m = lr.Momentum()
gd = classifier.LinearRegressionGD(m)
gd.fit(X_train, y_train)
y_pred = gd.pred(X_val)
print("Error (GD)", score.mse(y_val, y_pred))


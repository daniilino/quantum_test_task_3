import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

# create dataset instance
train_df = pd.read_csv('./internship_train.csv')
test_df = pd.read_csv('./internship_hidden_test.csv')
# --------------------

# split into train and validation slices
X_train, X_val, Y_train, Y_val = train_test_split(train_df.drop(["8", "target"], axis=1), train_df["target"], test_size=0.2)

X_test = test_df.drop(["8"], axis=1).copy()

print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, X_test.shape)
# --------------------------

# dataset standard scaling:
sc_X = preprocessing.StandardScaler().fit(X_train)
sc_y = preprocessing.StandardScaler().fit(pd.DataFrame(Y_train))

X = sc_X.transform(X_train)
y = sc_y.transform(pd.DataFrame(Y_train)).reshape(-1)
# -------------------

# building the model
poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(X)

poly_reg_model = LinearRegression()
# -------------------

# fit
poly_reg_model.fit(poly_features, y)

# validation inference
Y_pred = poly_reg_model.predict(poly.transform(sc_X.transform(X_val)))

# metrics
acc_logr = round(poly_reg_model.score(poly_features, y) * 100, 2)
print("train accuracy: ", acc_logr)
mse = mean_squared_error(sc_y.transform(pd.DataFrame(Y_val)).reshape(-1), Y_pred)
print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0))
# ------------------

# test inference
Y_test = poly_reg_model.predict(poly.transform(sc_X.transform(X_test)))
Y_test = pd.DataFrame(sc_y.inverse_transform(pd.DataFrame(Y_test)).reshape(-1))

# write to a file
Y_test.to_csv("./submissions.csv")
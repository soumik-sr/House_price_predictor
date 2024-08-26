import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
data = pd.read_csv(r"data\Housing.csv")
binary_map = {"yes":1,"no":0}
data['mainroad'] = data['mainroad'].map(binary_map)
data['guestroom'] = data['guestroom'].map(binary_map)
data['basement'] = data['basement'].map(binary_map)
data['hotwaterheating'] = data['hotwaterheating'].map(binary_map)
data['airconditioning'] = data['airconditioning'].map(binary_map)
data['prefarea'] = data['prefarea'].map(binary_map)
data['furnishingstatus'] = data['furnishingstatus'].map({"furnished": 1,"semi-furnished":0,"unfurnished":0})
# print(data.head())
X_features = ["area","bedrooms","bathrooms","stories","mainroad","guestroom","basement","hotwaterheating","airconditioning","parking","prefarea","furnishingstatus"]
updata = np.array(data) 
X_train = updata[:,1:]
y_train = updata[:,0]
fig, ax = plt.subplots(1,len(X_features),figsize=(25, 5), sharey=True)
for i in range(len(X_features)):
    # ax[i].scatter(updata[:,i],updata[:,0])
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price")
plt.tight_layout()
plt.show()
# # plt.plot()

# print(X_train[0])

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
# # print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
# # print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

sgdr = SGDRegressor(max_iter=10000)
sgdr.fit(X_norm,y_train)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")

# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
# print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

# print(f"Prediction on training set:\n{y_pred[:4]}" )
# print(f"Target values \n{y_train[:4]}")
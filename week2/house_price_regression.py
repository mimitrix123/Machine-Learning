"""Week 2 Mini Project: House Price Prediction with Linear Regression."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

for path in ("data/house_prices.csv","data/train.csv"):
    if os.path.exists(path):
        raw=pd.read_csv(path)
        if "SalePrice" in raw:
            cols=[c for c in ["GrLivArea","OverallQual","GarageCars","TotalBsmtSF"] if c in raw]
            if cols: df=raw[cols+["SalePrice"]].dropna(); break
else:
    rng=np.random.default_rng(42); n=300
    df=pd.DataFrame({"GrLivArea":rng.integers(700,3500,n),"OverallQual":rng.integers(3,10,n),"GarageCars":rng.integers(0,4,n),"TotalBsmtSF":rng.integers(400,2500,n)})
    df["SalePrice"]=(df.GrLivArea*110+df.OverallQual*18000+df.GarageCars*9000+df.TotalBsmtSF*45+rng.normal(0,15000,n)).clip(30000)
X=df.drop(columns="SalePrice"); y=df.SalePrice
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42); model=LinearRegression().fit(Xtr,ytr); pred=model.predict(Xte)
print("R²:",round(r2_score(yte,pred),4)); print("RMSE:",round(mean_squared_error(yte,pred,squared=False),2))
plt.figure(figsize=(7,5)); plt.scatter(yte,pred,alpha=.7); lim=[min(yte.min(),pred.min()),max(yte.max(),pred.max())]; plt.plot(lim,lim,linestyle="--"); plt.xlabel("Actual Price"); plt.ylabel("Predicted Price"); plt.title("Actual vs Predicted House Prices"); plt.tight_layout(); plt.show()

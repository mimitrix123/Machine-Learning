"""Week 4 Capstone: Heart Disease Prediction.
Put Kaggle/UCI data in data/heart.csv. Target: target or HeartDisease.
"""
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

path="data/heart.csv"
if not os.path.exists(path): raise FileNotFoundError("Add heart.csv to data/heart.csv")
df=pd.read_csv(path); target="target" if "target" in df else "HeartDisease"; y=df[target]
if y.dtype=="object": y=y.map({"Yes":1,"No":0,"Y":1,"N":0,"Positive":1,"Negative":0}).fillna(pd.to_numeric(y,errors="coerce")).astype(int)
X=df.drop(columns=target); cat=X.select_dtypes(include=["object","category","bool"]).columns.tolist(); num=[c for c in X if c not in cat]
pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
model=Pipeline([("pre",pre),("clf",LogisticRegression(max_iter=2000))]); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y); model.fit(Xtr,ytr)
p=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]; print("Accuracy:",round(accuracy_score(yte,p),4)); print("Precision:",round(precision_score(yte,p,zero_division=0),4)); print("Recall:",round(recall_score(yte,p,zero_division=0),4)); print("Confusion matrix:\n",confusion_matrix(yte,p))
fpr,tpr,_=roc_curve(yte,prob); print("ROC-AUC:",round(auc(fpr,tpr),4)); plt.figure(); plt.plot(fpr,tpr,label=f"AUC={auc(fpr,tpr):.3f}"); plt.plot([0,1],[0,1],linestyle="--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("Heart Disease ROC Curve"); plt.legend(); plt.show()
os.makedirs("models",exist_ok=True); joblib.dump(model,"models/heart_disease_logreg.joblib")

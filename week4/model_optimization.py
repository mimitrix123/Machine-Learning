"""
Week 4 - Model Evaluation & Optimization.
Cross-validation, GridSearchCV for SVM, joblib save/load, prediction.
"""
import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iris=load_iris(); X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=.2,random_state=42,stratify=iris.target)
pipe=Pipeline([("scaler",StandardScaler()),("svc",SVC())])
cv_scores=cross_val_score(pipe,X_train,y_train,cv=5,scoring="accuracy")
print("5-fold CV scores:",np.round(cv_scores,4)); print("Mean CV accuracy:",round(cv_scores.mean(),4))
grid=GridSearchCV(pipe,{"svc__C":[.1,1,10,100],"svc__gamma":["scale",.01,.1,1],"svc__kernel":["rbf","linear"]},cv=5,scoring="accuracy",n_jobs=-1)
grid.fit(X_train,y_train)
print("Best parameters:",grid.best_params_); print("Best CV accuracy:",round(grid.best_score_,4)); print("Test accuracy:",round(grid.score(X_test,y_test),4))
os.makedirs("models",exist_ok=True); joblib.dump(grid.best_estimator_,"models/iris_svm.joblib")
loaded=joblib.load("models/iris_svm.joblib"); new_data=np.array([[5.1,3.5,1.4,.2]])
print("Prediction for [5.1, 3.5, 1.4, 0.2]:",loaded.predict(new_data))

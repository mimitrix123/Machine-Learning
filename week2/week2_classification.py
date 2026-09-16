"""Week 2: Logistic Regression, Decision Tree and KNN on Titanic."""
import os
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

def load_titanic():
    return pd.read_csv("data/titanic.csv") if os.path.exists("data/titanic.csv") else sns.load_dataset("titanic")

def main():
    df=load_titanic(); target="Survived" if "Survived" in df else "survived"
    y=df[target].astype(int); cols=[c for c in ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"] if c in df]
    X=df[cols]; cat=X.select_dtypes(include=["object","category","bool"]).columns; num=[c for c in X if c not in cat]
    pre=ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
                           ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    models={"Logistic Regression":LogisticRegression(max_iter=1000),"Decision Tree":DecisionTreeClassifier(max_depth=4,random_state=42),"KNN":KNeighborsClassifier(n_neighbors=5)}
    for name,m in models.items():
        pipe=Pipeline([("preprocessor",pre),("model",m)]); pipe.fit(Xtr,ytr); p=pipe.predict(Xte)
        print(name,"accuracy:",round(accuracy_score(yte,p),4))
        if name=="Logistic Regression": print("Confusion matrix:\n",confusion_matrix(yte,p)); print(classification_report(yte,p,zero_division=0))
    print("\nDecision-tree max_depth comparison:")
    for depth in [2,3,4,5,7,None]:
        pipe=Pipeline([("preprocessor",pre),("model",DecisionTreeClassifier(max_depth=depth,random_state=42))]); pipe.fit(Xtr,ytr)
        print(depth,round(pipe.score(Xte,yte),4))

if __name__=="__main__": main()


print("starting imports")
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
print("model imports done")
import model as ml
import pipline as pl
print("done reading python files")
import pandas as pd 
list_of_models= [
 LogisticRegression(),
 HistGradientBoostingClassifier(),
 KNeighborsClassifier(),
 GaussianNB(),
 BernoulliNB(),
 MLPClassifier(),
 LinearDiscriminantAnalysis(),
 #QuadraticDiscriminantAnalysis(),
 SGDClassifier(),
 PassiveAggressiveClassifier(),
 RidgeClassifier(),
 SVC(),
 LinearSVC(),
 NuSVC(),
 DecisionTreeClassifier(),
 RandomForestClassifier(),
 ExtraTreesClassifier(),
 GradientBoostingClassifier(),
 AdaBoostClassifier()

]
traindata_data_desc = {
        "HOG": "data/Shuffled_HOG.csv",
        "LPB": "data/Shuffled_LBP.csv",
    }
for name,traindata in traindata_data_desc.items():
    print("loading data")
    df = pd.read_csv(traindata, nrows=10000)
    print(df.head())
    X = df.drop(columns=["animal","file"])
    print("starting train_test_spilt")
    X_train, X_test, y_train, y_test = pl.train_test_split_for_model(df=df)
    print("starting create preprocessor")
    preprocessor = pl.create_preprocessor(X)
    print("done pre")



    res =[]
    

    for model in list_of_models:
        print()
        print("starting " + str(model))
        Pipeline = ml.model_creater_and_run(preprocessor,model)
        print("model created done")
        score = pl.cross_validation(X_train=X_train,y_train=y_train,model =Pipeline)
        print("score is done")
        res.append({
            "score": score,
            f"model {name}":str(model),
            
        })
        print("finnished " + str(model))
    df = pd.DataFrame(res)
    df = df.sort_values(by="score",ascending=False)
    df.to_csv("data/model_data.csv",mode="a")
    print("done")

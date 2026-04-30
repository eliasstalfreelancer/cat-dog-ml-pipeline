import pipline as pl
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

list_of_models= [
 LogisticRegression(),
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
 HistGradientBoostingClassifier(),
 AdaBoostClassifier()

]
traindata_data_desc = {
        "HOG_LPB": "data/ShuffledHOG_LBP.csv",
    }
for name,traindata in traindata_data_desc.items():
    pl.model_create_save_compare(list_of_models,"models/",name=name,data_path=traindata)
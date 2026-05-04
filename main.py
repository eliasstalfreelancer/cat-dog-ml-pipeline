print("starting imports")
import pandas as pd
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
import data_preprocessing as  dp
import os 
print("done with imports")
def convert_to_seconds(t):
    d, h, m, s = map(int, t.split(":"))
    return d*86400 + h*3600 + m*60 + s
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

print("starting data preprocessing ------------------------------------------------------------ ")

#list_of_feature_methods = {"HOG" : dp.extract_HOG,"LBP":dp.extract_lbp }
if  os.path.exists("data/Dog_HOG_LBP.csv") and  os.path.exists("data/Cat_HOG_LBP.csv"):
    exct_list = False
else:
    if os.path.exists("data/Cat_HOG_LBP.csv"):
        exct_list = [["data/Dog_HOG.csv","data/Dog_LBP.csv","data/Dog_HOG_LBP.csv","HOG","LBP"]]
    elif os.path.exists("data/Dog_HOG_LBP.csv"):
        exct_list = [["data/Cat_HOG.csv","data/Cat_LBP.csv","data/Cat_HOG_LBP.csv","HOG","LBP"]]

print(exct_list)
if exct_list != False:  
    for item in exct_list:
        dp.merge_csv(item[0],item[1],item[2],item[3],item[4])
    print("starting pipeline ----------------------------------------------------------------")
    dp.interleave_csv("data/Cat_HOG_LBP.csv","data/Dog_HOG_LBP.csv","data/shuffledHOG_LBP.csv",chunksize=50000)
print("start model creation and saving ")
traindata_data_desc = {
        "HOG": "data/shuffled_HOG.csv",
        "LBP": "data/shuffled_LBP.csv",
        "HOG_LBP": "data/shuffledHOG_LBP.csv",
    }
for name,traindata in traindata_data_desc.items():
    pl.model_create_save_compare(list_of_models,"models/",name=name,data_path=traindata)

print("starting model data proccesing")
df = pd.read_csv("data/model_data.csv")
df["time_seconds"] = df["time DD:HH:MM:SS"].apply(convert_to_seconds)
df.loc[df["time_seconds"] == 0, "time_seconds"] = 1
df["efficiency"] = df["score"] / df["time_seconds"]
print(df.sort_values(by="efficiency", ascending=False).groupby("feature_method").head(1))
df.to_csv("data/model_data.csv",index= False)
candidates = dp.get_top_eff_and_score(df)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import joblib
import os
import model as ml
import warnings
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import tools


def train_test_split_for_model(
        df: pd.DataFrame,
        target_column="animal",
        test_size=0.2,
        random_state=42
    ):
    df = df.copy()
    
    X = df.drop(columns=[target_column])
    y = df[target_column]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def create_preprocessor(X):

    numeric_features = X.columns.tolist()


    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

  

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        
    ])

    return preprocessor

def create_pipeline(preprocessor,model):
    pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
    ])
    return pipeline


def cross_validation(X_train, y_train, model
 ):
    results = {}

    scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="f1")
        
    results =scores.mean()
    
    return results


def model_create_save_compare(list_of_model_name,save_path, data_path= "data/shuffleddata.csv",nrows = 9000,target_colum ="animal"):
    var = tools.time_lapsed()
    print("loading data")
    df = pd.read_csv(data_path, nrows=nrows)
    var = tools.time_lapsed(var)
    print(f"Done loading data: timetook {var[-1]}:{var[-2]}:{var[-3]}:{var[-4]}")
    X = df.drop(columns=[target_colum])
    print("starting train_test_spilt")
    X_train, X_test, y_train, y_test = train_test_split_for_model(df=df)
    print("starting create preprocessor")
    preprocessor = create_preprocessor(X)
    print("done pre")
    if "model_data.csv" in os.listdir("data/"):
        res = pd.read_csv("data/model_data.csv").to_dict("records")
    else:
        res =[]
    saved_models_list = []
    for path in os.listdir(save_path):
         saved_models_list.append(path)
    for model in list_of_model_name:
        if not str(model)+".pkl" in saved_models_list:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                print("starting " + str(model))
                Pipeline = ml.model_creater_and_run(preprocessor,model)
                print("model created done")
                var = tools.time_lapsed()
                score = cross_validation(X_train=X_train,y_train=y_train,model =Pipeline)
                
                var = tools.time_lapsed(var)
                print(f"score is done : timetook {var[-1]}:{var[-2]}:{var[-3]}:{var[-4]}")
                
                res.append({
                    "score": score,
                    "model": str(model),
                    "time (Day:Hours:Minutes:Secunds)": f"{var[-1]}:{var[-2]}:{var[-3]}:{var[-4]}",
                    "Warning" : w
                })
                print("finnished " + str(model))
                df = pd.DataFrame(res)
                df = df.sort_values(by="score",ascending=False)
                df.to_csv("data/model_data.csv",index=False)
                joblib.dump(Pipeline,save_path+str(model)+".pkl")
        else:
            print("skiping " + str(model))
        
    print("done")


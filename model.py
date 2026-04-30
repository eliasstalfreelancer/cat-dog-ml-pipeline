print("start model import")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pipline as pl

print("imoprts loaded")
def baseline_model(preprocessor):
    return pl.create_pipeline(preprocessor,LogisticRegression(max_iter=250000))

print("base loaded")
def model_creater_and_run(preprocessor,model,random_state = 42):
    try:
       output = pl.create_pipeline(preprocessor,model(random_state=random_state))
    except:
        try:
            output = pl.create_pipeline(preprocessor,model(max_iter=250000))
        except:
            output = pl.create_pipeline(preprocessor,model)
    return output
print("model creater loaded")
def tune_random_forest(rf_model, X_train, y_train):

    param_grid = {
        "model__n_estimators": [300,500],
        "model__max_depth": [20,30],
        "model__min_samples_split": [5,8],
        "model__min_samples_leaf": [1, 2]
    }
    
    total_sum = 1
    for i in param_grid:
        total_sum = total_sum * len(param_grid[i])
    total_sum = total_sum*5

    grid_search = GridSearchCV(
        rf_model,
        param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose= 0
    )


print("model done import")
import time


from sklearn import svm, datasets
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV,HalvingGridSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform

def generate_dataset():
    data = datasets.load_iris()
    return data

def grid_search_pipeline():
    data = generate_dataset()
    svc = svm.SVC()
    params = {"kernel":["linear","rbf","sigmoid"],"C":[1,5,10]}

    classifier = GridSearchCV(svc,params)
    start_time = time.time()
    classifier.fit(data.data,data.target)
    print(f"Time taken for fitting {time.time() - start_time} seconds")

    print(classifier.best_params_)

def random_search_pipeline():
    data = generate_dataset()
    logistic = LogisticRegression(solver="saga",max_iter=500,tol=1e-3,random_state=21)
    params = {"C":uniform(loc=0, scale=6),"penalty":["l1","l2","elasticnet"],"l1_ratio":uniform(loc=0,scale=1)}


    classifier = RandomizedSearchCV(logistic,params,random_state=32)
    start_time = time.time()
    classifier.fit(data.data, data.target)
    print(f"Time taken for fitting {time.time() - start_time} seconds")

    print(classifier.best_params_)

def half_grid_search_pipeline():
    data = generate_dataset()
    svc = svm.SVC()
    params = {"kernel":["linear","rbf","sigmoid"],"C":list(range(1,20))}

    classifier = HalvingGridSearchCV(svc,params,scoring="accuracy",factor=3)
    start_time = time.time()
    classifier.fit(data.data,data.target)
    print(f"Time taken for fitting {time.time() - start_time} seconds")

    print("Best Params", classifier.best_params_)
    print("Best CV Score", classifier.best_score_)




if __name__=="__main__":
    # grid_search_pipeline()
    # random_search_pipeline()
    half_grid_search_pipeline()
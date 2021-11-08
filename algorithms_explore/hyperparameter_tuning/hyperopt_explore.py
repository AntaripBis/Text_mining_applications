import time

from sklearn import svm, datasets
from scipy.stats import uniform
from hpsklearn import HyperoptEstimator, svc, any_preprocessing
from hyperopt import tpe


def generate_dataset():
    X,y = datasets.load_iris(return_X_y=True)
    return X,y

def bayesian_opt_pipeline():
    X,y = generate_dataset()

    estimator = HyperoptEstimator(classifier=svc("hyperopt_svc"),preprocessing=any_preprocessing("hyperopt_preprocess"),
                                  algo=tpe.suggest,max_evals=100,trial_timeout=120)
    start_time = time.time()
    estimator.fit(X,y)
    print(f"Time taken for fitting {time.time() - start_time} seconds")

    print("best model:")
    print(estimator.best_model())

if __name__=="__main__":
    bayesian_opt_pipeline()
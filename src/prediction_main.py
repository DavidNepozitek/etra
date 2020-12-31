import argparse
import datetime
import functools
import os
import pathlib
import re
import time

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sktime.classification.compose import (ColumnEnsembleClassifier,
                                           TimeSeriesForestClassifier)
from sktime.transformers.panel.rocket import Rocket
from sktime.transformers.panel.shapelets import ContractedShapeletTransform
from sktime.transformers.panel.summarize import RandomIntervalFeatureExtractor
from sktime.transformers.panel.truncation import TruncationTransformer
from sktime.transformers.panel.tsfresh import TSFreshFeatureExtractor
from sktime.utils.time_series import time_series_slope
from sktime.classification.shapelet_based import MrSEQLClassifier

import src.data_pipeline as data

np.random.seed(1)

def generate_dataset(column=None):
    X, y, groups  = data.generate_dataset("data", data.ALL_SUBJECTS, data.ALL_STIMULI)

    if column is not None:
        return X[column].to_frame(), y, groups
    
    return X, y, groups

MAX_LENGTH = 18886

datasources = {
        "LX": functools.partial(generate_dataset, column="LX"),
        "LY": functools.partial(generate_dataset, column="LY"),
        "full": generate_dataset
    }

steps = [
(
    "extract",
    RandomIntervalFeatureExtractor(
        n_intervals="sqrt", features=[np.mean, np.std, time_series_slope]
    ),
),
("clf", DecisionTreeClassifier()),
]
time_series_tree = Pipeline(steps)

models = {
    "features": make_pipeline(
        TruncationTransformer(lower=MAX_LENGTH),
        TSFreshFeatureExtractor(default_fc_parameters="efficient", show_warnings=False,n_jobs=-1),
        RandomForestClassifier(n_jobs=-1, random_state=1),
        verbose=True
    ),
    "interval": make_pipeline(
        TruncationTransformer(lower=15000),
        TimeSeriesForestClassifier(
            estimator=time_series_tree,
            n_estimators=100,
            criterion="entropy",
            bootstrap=True,
            oob_score=True,
            random_state=1,
            n_jobs=-1,
        ),
        verbose=True
    ),
    "shapelet": make_pipeline(
        TruncationTransformer(lower=1000),
        ContractedShapeletTransform(
            time_contract_in_mins=10,
            num_candidates_to_sample_per_case=10,
            verbose=2,
            random_state=1
        ),
        RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1),
        verbose=True
    ),
    "rocket": make_pipeline(
        TruncationTransformer(lower=MAX_LENGTH),
        Rocket(random_state=1),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True),
        verbose=True
    ),
    "mr-seql": make_pipeline(
        TruncationTransformer(lower=MAX_LENGTH),
        MrSEQLClassifier(symrep=['sax', 'sfa']),
        verbose=True
    ),
    "full_features": make_pipeline(
        TruncationTransformer(lower=MAX_LENGTH),
        ColumnEnsembleClassifier([
            (
                "features_0", 
                make_pipeline(
                    TSFreshFeatureExtractor(default_fc_parameters="efficient", show_warnings=False,n_jobs=-1),
                    RandomForestClassifier(n_jobs=-1, random_state=1),
                    verbose=True
                ),
                [0]
            ),
            (
                "features_1", 
                make_pipeline(
                    TSFreshFeatureExtractor(default_fc_parameters="efficient", show_warnings=False,n_jobs=-1),
                    RandomForestClassifier(n_jobs=-1, random_state=1),
                    verbose=True
                ),
                [1]
            ),
            (
                "features_2", 
                make_pipeline(
                    TSFreshFeatureExtractor(default_fc_parameters="efficient", show_warnings=False,n_jobs=-1),
                    RandomForestClassifier(n_jobs=-1, random_state=1),
                    verbose=True
                ),
                [2]
            ),
        ], verbose=True),
        verbose=True
    ),
    "full_interval": make_pipeline(
        TruncationTransformer(lower=MAX_LENGTH),
        ColumnEnsembleClassifier([
            (
                "TimeSeriesForestClassifier_0", 
                TimeSeriesForestClassifier(
                    estimator=time_series_tree,
                    n_estimators=100,
                    criterion="entropy",
                    bootstrap=True,
                    oob_score=True,
                    random_state=1,
                    n_jobs=-1,
                    verbose=True
                ),
                [0]
            ),
            (
                "TimeSeriesForestClassifier_1", 
                TimeSeriesForestClassifier(
                    estimator=time_series_tree,
                    n_estimators=100,
                    criterion="entropy",
                    bootstrap=True,
                    oob_score=True,
                    random_state=1,
                    n_jobs=-1,
                    verbose=True
                ),
                [1]
            ),
            (
                "TimeSeriesForestClassifier_2", 
                TimeSeriesForestClassifier(
                    estimator=time_series_tree,
                    n_estimators=100,
                    criterion="entropy",
                    bootstrap=True,
                    oob_score=True,
                    random_state=1,
                    n_jobs=-1,
                    verbose=True
                ),
                [2]
            ),
        ], verbose=True),
        verbose=True
    )
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Name of the model to use.")
    parser.add_argument("--data", type=str, help="Type of data to use.")
    parser.add_argument("--info", type=str, help="Additional information about the training.")
    parser.add_argument("--model-path", type=str, help="Path to a model to load. This will override teh --model parameter.")
    args = parser.parse_args()
    arg_dict = vars(args)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    arg_dict["log_dir"] = "./logs/" + current_time
    pathlib.Path(arg_dict["log_dir"]).mkdir(parents=True, exist_ok=True)

    print(arg_dict, flush=True)

    X, y, groups = datasources[arg_dict["data"]]()

    print(list(X.columns), flush=True)

    model = models[arg_dict["model"]]

    if "model_path" in arg_dict and arg_dict["model_path"] is not None:
        model = load(arg_dict["model_path"])
        print(f"Loaded model from {arg_dict['model_path']}")

    start_time = time.time()

    logo = LeaveOneGroupOut()
    scores = cross_val_score(model, X, y, groups=groups, cv=logo, n_jobs=-1, verbose=1)
    print(f"Mean of cross validation scores: {scores.mean()}", flush=True)
    print(arg_dict, flush=True)
    dump(model, f"{arg_dict['log_dir']}/model.joblib")
    
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Task completed in {str(datetime.timedelta(seconds=elapsed))}.")


if __name__ == '__main__':
    main()
    


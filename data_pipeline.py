import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_score
from sktime.classification.compose import (
    ColumnEnsembleClassifier,
    TimeSeriesForestClassifier,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sktime.transformers.panel.tsfresh import TSFreshFeatureExtractor
from sktime.transformers.panel.truncation import TruncationTransformer
import re
import os

FILE_REG = re.compile("([^_]+)_([^_]+)_([^_]+)_([^_]+).*")

STIMULI_ENCODING = {
    "Blank": 0,
    "Natural": 1,
    "Puzzle": 3,
    "Waldo": 4,
}

ALL_SUBJECTS = ["019", "009", "022", "058", "059", "060", "062", "SMC"]
ALL_STIMULI =  ["Waldo", "Blank", "Natural", "Puzzle"]

def match_trialfile_name(file_name):
    """Gets trial information from filename

    Args:
        filename (string): filename of the trial

    Returns:
        Tuple: 
            Item 0 - Subject id
            Item 2 - Task (Fixation|FreeViewing)
            Item 3 - Stimulus
    """
    match_res = FILE_REG.search(file_name)
    if match_res is None:
        print(file_name)
    return match_res.group(1), match_res.group(3), match_res.group(4)

def generate_dataset(dataset_path, subjects, stimuli):
    count = 0
    dataset_dict = {
        "Time": [],
        "LX": [],
        "LY": [],
        "LP": []
    }
    trial_stimuli = []
    trial_subjects = []
    for subject_dir in os.listdir(dataset_path):
        if subject_dir in subjects:
            for trial_filename in os.listdir(f"{dataset_path}/{subject_dir}"):
                subj, task, stimulus = match_trialfile_name(trial_filename)

                if task != "FreeViewing" or stimulus not in stimuli:
                    continue

                print(f"Processing Trial {count}: Subject: {subj}, Task: {task}, Stimulus: {stimulus} ({STIMULI_ENCODING[stimulus]})")
                
                trial_df = pd.read_csv(f"{dataset_path}/{subject_dir}/{trial_filename}")
                dataset_dict["Time"].append(trial_df["Time"])
                dataset_dict["LX"].append(trial_df["LXpix"].dropna())
                dataset_dict["LY"].append(trial_df["LYpix"])
                dataset_dict["LP"].append(trial_df["LP"])
                trial_stimuli.append(STIMULI_ENCODING[stimulus])
                trial_subjects.append(subj)
                count += 1


    print(f"Processed {count} trials")
    return pd.DataFrame(dataset_dict), pd.Series(trial_stimuli), pd.Series(trial_subjects)



# trial_df = pd.read_csv("data/019/019_024_FreeViewing_Waldo_wal003.csv")
# trial_df.head()

if __name__ == '__main__':
    X, y, groups  = generate_dataset("data", ALL_SUBJECTS, ["Puzzle", "Blank", "Waldo", "Natural"])

    clf = make_pipeline(
        TruncationTransformer(lower=250),
        TSFreshFeatureExtractor(default_fc_parameters="efficient", show_warnings=False),
        RandomForestClassifier(),
    )

    # clf = ColumnEnsembleClassifier(
    #     estimators=[
    #         ("TSF0", TimeSeriesForestClassifier(n_estimators=100), ["LX"])
    #     ]
    # )
    # clf.fit(X["LX"].to_frame(), y)
    # print(clf.score(X["LX"].to_frame(), y))

    logo = LeaveOneGroupOut()
    scores = cross_val_score(clf, X["LX"].to_frame(), y, groups=groups, cv=logo)
    print(scores.mean())


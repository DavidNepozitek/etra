# Visual Stimuli Classification Using Universal Time Series Classification Techniques


## Docs
- [Report](Report.md)

## Requirments
In order to run the experiments:

1. Download the [ETRA 2019 challenge dataset](http://smc.neuralcorrelate.com/ETRA2019/ETRA2019Challenge.zip) and extract its contents to the root of this project
2. Install the following prerequisities
    - python=3.7
    - matplotlib
    - numpy
    - pandas
    - rope
    - cython
    - scikit-learn
    - sktime
    - tsfresh
3. To run an experiment, run bin/train.sh or execute its content manually


## Project Organization

    ├── README.md       <- The top-level README for developers using this project.
    ├── Report.md       <- Report source file
    ├── Report.pdf      <- Report exported as PDF
    |
    ├── bin             <- Executable scripts
    |   └── train.sh    <- Training script
    |
    └── src                <- Source code for use in this project.
        ├── data_pipeline.py     <- Scripts to process data
        └── prediction_main.py   <- Module with all the models the training script
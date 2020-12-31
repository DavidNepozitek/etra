#!/usr/bin/env bash

# Rocket
python -m "src.prediction_main" --model "rocket" --data "full" --info "ROCKET"

# Uncomment the following for Time Series Forest
# python -m "src.prediction_main" --model "full_interval" --data "full" --info "Time Series Forest"

# Uncomment the following for tsfresh features
# python -m "src.prediction_main" --model "full_features" --data "full" --info "tsfresh features"


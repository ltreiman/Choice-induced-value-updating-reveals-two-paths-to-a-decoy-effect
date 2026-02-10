# Data

This folder contains behavioral data and model outputs from the experiments reported in 'Choice-induced value updating reveals two paths to a decoy effect' by Lauren S. Treiman and Wouter Kool.

All conents can also be found on our OSF page: https://osf.io/qb8tz/overview

## Contents

### Root Level
- `simulation_results.csv` - Output from computational simulations (can be regenerated using `/model_fitting_and_simulation/simulation.py`) [Note: this file is too big, but you can find it on the OSF page or you can run `model_fitting_and_simulation/simulation.py`]

### main_experiment/
- `choices.csv` - Behavioral data from main study
- `subinfo.csv` - Information from each participant including gender, age, and final outcome
- `data_for_model_fitting.csv` - Preprocessed data formatted for computational modeling (can be regenerated using `/analysis/behavioral_analysis.R`)
- `deltas.csv` - Estimated delta parameters for each participant (can be regenerated using `/model_fitting_and_simulation/model_fitting.py`)
- `delta_behavior_validation_all_splits.csv` - Cross-validation results (can be regenerated using `/model_fitting_and_simulation/cross_validation.py`)

### pilot/
- `choices.csv` - Behavioral data from pilot study
- `subinfo.csv` - Information from each participant including gender, age, and final outcome
- `data_for_model_fitting_pilot.csv` - Preprocessed data formatted for computational modeling (can be regenerated using `/analysis/behavioral_analysis.R`)
- `deltas.csv` - Estimated delta parameters for each participant (can be regenerated using `/model_fitting_and_simulation/model_fitting.py`)
- `pilot_delta_behavior_validation_all_splits.csv` - Cross-validation results (can be regenerated using `/model_fitting_and_simulation/cross_validation.py`)

## Notes

- All data files are in CSV format
- Missing values are coded as `NA`
- Participant IDs have been anonymized

---

# Analysis Scripts

## analysis/
- `behavioral_analysis.R` - Main analysis of behavioral findings for main experiment and pilot
- `helper_functions.R` - Helper functions used in behavioral analysis

## model_fitting_and_simulation/
- `cross_validation.py` - Cross-validation analysis to test whether delta predicts behavior on independent data
  - **Input**: `data_for_model_fitting.csv` and `data_for_model_fitting_pilot.csv`
  - **Output**: `delta_behavior_validation_all_splits.csv` and `pilot_delta_behavior_validation_all_splits.csv`
  
- `model_fitting.py` - Fit computational model to estimate delta parameters for each participant
  - **Input**: `data_for_model_fitting.csv` and `data_for_model_fitting_pilot.csv`
  - **Output**: `deltas.csv`
  
- `simulation.py` - Computational simulations testing model predictions
  - **Output**: `simulation_results.csv`
  
- `plot_results.ipynb` - Generate figures from cross-validation, model fitting, and simulation outputs
  - Can be run independently if you don't want to regenerate all data files

## pre-registration/
- `model_fitting.py` - Pre-registered model fitting code (identical to `/model_fitting_and_simulation/model_fitting.py` but plot results in plot_results.ipynb)

# Task Code

This experiment is implemented in **jsPsych**, using **JavaScript, HTML, and CSS**.

## File Overview

- `consent.html` — Consent form. The original consent content has been removed, but the structure remains.
- `database_config.php` — Configures the connection to the PHP server.
- `debrief.html` — Contains participant debriefing information.
- `functions.js` — Helper functions used throughout the codebase. Split into a separate file for improved organization and readability.
- `index.html` — Main entry point that launches and runs the experiment.
- `intro.js` — Contains task instructions shown to participants.
- `jspsych/` — Folder containing jsPsych plugins and CSS files.
- `write_data.php` — Handles writing and saving participant data to the PHP server.
---

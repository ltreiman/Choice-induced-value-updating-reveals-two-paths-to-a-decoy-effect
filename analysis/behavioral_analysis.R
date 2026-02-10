setwd("./OSF")

library(ggplot2)
library(dplyr)
library(forcats)
library(rstatix)
library(ggsignif)
library(tidyr)
library(lmerTest)


source("./analysis/helper_functions.R")

############## EXPERIMENT

# Download data
choices <- read.csv("./data/main_experiment/choices.csv") 
subinfo<- read.csv("./data/main_experiment/subinfo.csv")

# Determine which participants to remove
dropouts <- determine_participants_to_remove(choices)
# Clean dataframe
cleaned_data <- clean_data_frame(choices, dropouts)

# Save data frame for model fitting, note that this data frame is already saved
#write.csv(cleaned_data, "./data/main_experiment/data_for_model_fitting.csv")

# Deltas are saved, but can also be updated after running model_fitting.py
deltas <- read.csv("./data/main_experiment/deltas.csv") %>% select(id, delta_delta) %>% rename(delta = delta_delta)

#---------------------- Results ----------------------
# All functions can be found in helper_functions.R
sequential_decoy_effect(cleaned_data)
ordering_effect(cleaned_data)
first_option_correlation(cleaned_data, deltas)
ordering_effects_by_delta(cleaned_data, deltas)
rt_delta_analysis(cleaned_data, deltas)


############## PILOT
# Note that the same functions are used to produce both the pilot and main experiment results. 

choices <- read.csv("./data/pilot/choices.csv") 
subinfo<- read.csv("./data/pilot/subinfo.csv")

dropouts <- determine_participants_to_remove(choices)
cleaned_data <- clean_data_frame(choices, dropouts)
# write.csv(cleaned_data, "./data/pilot/data_for_model_fitting_pilot.csv")
deltas_pilot <- read.csv("./data/pilot/deltas.csv") %>% select(id, delta_delta) %>% rename(delta = delta_delta)
sequential_decoy_effect(cleaned_data)
ordering_effect(cleaned_data)
first_option_correlation(cleaned_data, deltas_pilot)
ordering_effects_by_delta(cleaned_data, deltas_pilot)
rt_delta_analysis(cleaned_data, deltas_pilot)



import numpy as np
import copy 
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm.contrib.concurrent import process_map
from itertools import product, chain



from model_fitting import trial_loglik, neg_loglik, null_neg_loglik

def calculate_first_pair_proportions(gen_model_results, delta):
    prop_T = gen_model_results.groupby('first_pair')['decoy_final_response_gamble'].apply(lambda x: (x == 'T').mean())
    result_df = pd.DataFrame([prop_T.values], columns=prop_T.index)
    result_df.insert(0, 'delta', delta)  # Insert delta as the first column
    return result_df

def generative_model(n_trials, beta, delta, seed):
    if seed is not None:
        np.random.seed(seed)

    # Generate expected values
    EV_target = np.random.uniform(14, 45, n_trials)
    EV_competitor = copy.deepcopy(EV_target)
    EV_decoy = EV_target - np.random.uniform(5, 10, n_trials)

    # Determine first pair
    options = ["(C,T)", "(C,D)", "(D,T)"]
    probs = [1/3, 1/3, 1/3]
    first_pair = np.random.choice(options, size=n_trials, p=probs)
    # Ensure that number of each first pair are relatively close to each other
    while not np.allclose(np.bincount(pd.Categorical(first_pair, categories=options).codes), n_trials/3, atol=3):
        first_pair = np.random.choice(options, size=n_trials, p=probs)

    df = pd.DataFrame({
        "EV_target": EV_target,
        "EV_competitor": EV_competitor,
        "EV_decoy": EV_decoy,
        "first_pair": first_pair
    })

    df[["EV_first1", "EV_first2", "EV_final","option_first1", "option_first2", "option_final",]] = df.apply(get_first_pair_values, axis=1)
    # Simulate first choice
    df["decoy_first_response_gamble"] = df.apply(simulate_first_choice, axis=1, args=(beta,))
    # Simulate final choice
    df["decoy_final_response_gamble"] = df.apply(simulate_final_choice, axis=1, args=(beta, delta))

    
    result = df.drop(columns=["EV_first1", "EV_first2", "EV_final", "option_first1", "option_first2", "option_final"])
    return result


def get_first_pair_values(row):
    # Helper function to get rows so I do not have to use for loop
    if row["first_pair"] == "(C,T)":
        return pd.Series({"EV_first1": row["EV_competitor"], "EV_first2": row["EV_target"], "EV_final": row["EV_decoy"],
                          "option_first1": "C", "option_first2": "T", "option_final": "D"})
    elif row["first_pair"] == "(C,D)":
        return pd.Series({"EV_first1": row["EV_competitor"], "EV_first2": row["EV_decoy"], "EV_final": row["EV_target"],
                            "option_first1": "C", "option_first2": "D", "option_final": "T"})
    else:  # decoy_vs_target
        return pd.Series({"EV_first1": row["EV_decoy"], "EV_first2": row["EV_target"], "EV_final": row["EV_competitor"],
                           "option_first1": "D", "option_first2": "T", "option_final": "C"})
    
def simulate_first_choice(row, beta):
    p_first_option = np.exp(beta * row["EV_first1"]) / (np.exp(beta * row["EV_first1"]) + np.exp(beta * row["EV_first2"]))
    choice_idx = np.random.binomial(1, p_first_option)  # 1 = first option, 0 = second option
    return row["option_first1"] if choice_idx == 1 else row["option_first2"]

def simulate_final_choice(row, beta, delta):
    # figure out which pair forms second round
    first_choice = row["decoy_first_response_gamble"]
    first_pair = row["first_pair"]

    # figure out the third option (not in first pair)
    all_items = ["C", "D", "T"]
    col_map = {"T": "EV_target", "C": "EV_competitor", "D": "EV_decoy"}
    pair_items = first_pair.strip("()").split(",")
    third_item = [x for x in all_items if x not in pair_items][0]

    # make second pair (first choice + third)
    pair = [first_choice, third_item]

    EV1 = row[col_map[pair[0]]]
    EV2 = row[col_map[pair[1]]]

    # add delta only to the one that was chosen before
    EV1 += delta  # because pair[0] is the previously chosen one

    # compute probability and pick
    p1 = np.exp(beta * EV1) / (np.exp(beta * EV1) + np.exp(beta * EV2))
    choice_idx = np.random.binomial(1, p1)
    return pair[0] if choice_idx == 1 else pair[1]

# Update number of rounds, trials, and beta parameters here
def main(delta_value, n_trials = 200, betas=[.1, 0.2], n_rounds = 1000):
    rows = []
    for round in range(n_rounds):
        beta = np.random.uniform(betas[0], betas[1])
        df_delta = generative_model(n_trials, beta, delta = delta_value, seed=None)
        prop_T_row = calculate_first_pair_proportions(df_delta, delta_value)
        rows.append(prop_T_row)
    return rows



if __name__ == "__main__":  
    n_deltas = 25 # Number of deltas
    # Create list of deltas want to measure
    arg_list = list(np.arange(0.5, n_deltas + .5, 0.5)) * 50 

    results_list = process_map(main, arg_list, ncols=80, chunksize=1)
    results_list = chain(*results_list)

    final_df = pd.concat(results_list, ignore_index=True)
    # This takes a few hours to run. csv of results is saved in simulation folder
    final_df.to_csv('data/simulation_data.csv', index=False)
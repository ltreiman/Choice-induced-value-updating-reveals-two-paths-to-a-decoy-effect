import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from tqdm.contrib.concurrent import process_map
from itertools import product, chain


from model_fitting import trial_loglik, neg_loglik, null_neg_loglik

def stratified_split(df_pid, random_state):
    """
    Split data:
    - Training: ALL (C,T) + 33/44 of (C,D) + 33/44 of (D,T)
    - Test: Remaining 11/44 of (C,D) + 11/44 of (D,T)
    """
    np.random.seed(random_state)
    
    train_dfs = []
    test_dfs = []
    
    # Get all (C,T) trials for training
    ct_trials = df_pid[df_pid['first_pair'] == '(C,T)'].copy()
    train_dfs.append(ct_trials)
    
    # Split (C,D) trials: 33 for train, 11 for test
    cd_trials = df_pid[df_pid['first_pair'] == '(C,D)'].copy()
    cd_shuffled = cd_trials.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n_cd_train = int(len(cd_shuffled) * 0.75)
    train_dfs.append(cd_shuffled.iloc[:n_cd_train])
    test_dfs.append(cd_shuffled.iloc[n_cd_train:])
    
    # Split (D,T) trials: 33 for train, 11 for test
    dt_trials = df_pid[df_pid['first_pair'] == '(D,T)'].copy()
    dt_shuffled = dt_trials.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n_dt_train = int(len(cd_shuffled) * 0.75)
    train_dfs.append(dt_shuffled.iloc[:n_dt_train])
    test_dfs.append(dt_shuffled.iloc[n_dt_train:])
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    
    return train_df, test_df


def fit_participant_cv(args):
    """Fit delta on training data, calculate behavioral metrics on test data"""
    pid, df_pid_data, n_splits, rounds, delta_range, beta_range = args
    
    df_pid = pd.DataFrame(df_pid_data)
    split_results = []
    
    # Run multiple splits
    for split_i in range(n_splits):
        # Stratified split: ALL CT + 75% CD + 75% DT in training
        train_df, test_df = stratified_split(df_pid, random_state=split_i)
        
        train_trials = train_df.to_dict('records')
        
        # === FIT DELTA MODEL ON TRAINING DATA ===
        best_delta_res = None
        best_delta_ll = -np.inf

        for attempt in range(rounds):
            start_delta = np.random.uniform(delta_range[0], delta_range[1])
            start_beta = np.random.uniform(beta_range[0], beta_range[1])

            res_delta = minimize(lambda d: neg_loglik(d, train_trials), 
                            x0=[start_delta, start_beta], 
                            bounds=[delta_range, beta_range])

            current_ll = -res_delta.fun
            if current_ll > best_delta_ll:
                best_delta_ll = current_ll
                best_delta_res = res_delta 

        # Store fitted parameters
        fitted_delta = best_delta_res.x[0]
        fitted_beta = best_delta_res.x[1]
        
        # === CALCULATE BEHAVIORAL PROPORTIONS ON TEST DATA ===
        if len(test_df) > 0:
            # Group test data by first_pair and calculate P(Target selected)
            grouped = test_df.groupby('first_pair')['decoy_final_response_gamble'].apply(
                lambda x: (x == 'T').mean()
            ).to_dict()
            
            # Calculate the difference: P(Target|DT) - P(Target|CD)
            p_target_DT = grouped.get('(D,T)', np.nan)
            p_target_CD = grouped.get('(C,D)', np.nan)
            behavior_diff = p_target_DT - p_target_CD if not np.isnan(p_target_DT) and not np.isnan(p_target_CD) else np.nan
        else:
            p_target_DT = np.nan
            p_target_CD = np.nan
            behavior_diff = np.nan
        
        # Store results for this split
        split_results.append({
            'id': pid,
            'split_i': split_i,
            'delta': fitted_delta,
            'beta': fitted_beta,
            'behavior_diff': behavior_diff,
            'p_target_DT': p_target_DT,
            'p_target_CD': p_target_CD
        })
    
    return split_results  # Return list of dicts, one per split


if __name__ == "__main__": 
    # Download data from main experiment
    data = pd.read_csv("data/main_experiment/data_for_model_fitting.csv")
    # Download data from pilot experiment
    #data = pd.read_csv("data/pilot/data_for_model_fitting_pilot.csv")
    
    # Columns I want to use
    cols = ["id", "trial_number", "trial_set", "risky_amount", "risky_probability", 
            "safe_amount", "safe_probability", "decoy_amount", "decoy_probability", 
            "decoy_first_response_gamble", "decoy_final_response_gamble", "decoy_added_gamble"]
    df = data[cols]

    # Need to create target amount and probability as well as safe amount and probability
    df = df.assign(
        target_amount = np.where(df["trial_set"] == "riskyDecoy", df["risky_amount"], df["safe_amount"]),
        target_probability = np.where(df["trial_set"] == "riskyDecoy", df["risky_probability"], df["safe_probability"]),
        competitor_amount = np.where(df["trial_set"] == "riskyDecoy", df["safe_amount"], df["risky_amount"]),
        competitor_probability = np.where(df["trial_set"] == "riskyDecoy", df["safe_probability"], df["risky_probability"])
    )

    # Calculate expected values
    df = df.assign(EV_target = df['target_probability'] * df['target_amount'])
    df = df.assign(EV_competitor = df['competitor_probability'] * df['competitor_amount'])
    df = df.assign(EV_decoy = df['decoy_probability'] * df['decoy_amount'])

    # Create first pair dataset
    mapping = {"C": "(D,T)", "T": "(C,D)", "D": "(C,T)"}
    df["first_pair"] = df["decoy_added_gamble"].map(mapping)

    # Remove columns that are now there just to confuse me
    cleaned_df = df.drop(columns=["risky_amount", "risky_probability", "safe_amount", 
                                  "safe_probability", "trial_set", "target_amount", 
                                  "target_probability", "decoy_amount", "decoy_probability", 
                                  "competitor_probability", "competitor_amount"])

    # Parameters
    participants = cleaned_df['id'].unique()
    n_splits = 1000 # Set to 1000 for experiment
    rounds = 20
    beta_range = (0, 1)
    delta_range = (-50, 50)
    
    # Prepare arguments for each participant
    arg_list = []
    for pid in participants:
        df_pid = cleaned_df[cleaned_df['id'] == pid]
        df_pid_data = df_pid.to_dict('records')
        arg_list.append((pid, df_pid_data, n_splits, rounds, delta_range, beta_range))
    
    # Run in parallel with progress bar
    results = process_map(fit_participant_cv, arg_list, max_workers=8, chunksize=1)
    
    # Flatten results (each participant returns a list of dicts)
    flattened_results = [item for sublist in results for item in sublist]
    results_df = pd.DataFrame(flattened_results)
    # Save data (note file takes a while to run so data saved already in data folder)
    ## Main experiment
    #results_df.to_csv('data/main_experiment/delta_behavior_validation_all_splits.csv', index=False)
    ## Pilot experiment
    #results_df.to_csv('data/pilot/pilot_delta_behavior_validation_all_splits.csv', index=False)

    # CALCULATE CORRELATION FOR EACH SPLIT 
    correlations = []
    
    for split_i in range(n_splits):
        split_data = results_df[results_df['split_i'] == split_i]
        
        # Remove NaN values
        valid_mask = ~(split_data['delta'].isna() | split_data['behavior_diff'].isna())
        if valid_mask.sum() > 1:  # Need at least 2 points for correlation
            corr = split_data[valid_mask][['delta', 'behavior_diff']].corr().iloc[0, 1]
            correlations.append(corr)
    
    correlations = np.array(correlations)
    
    # PLOT HISTOGRAM OF CORRELATIONS
    plt.figure(figsize=(10, 6))
    plt.hist(correlations, bins=20, edgecolor='black', alpha=0.7, color='#928b8b')

    mean_corr = np.mean(correlations)
    plt.axvline(x=mean_corr, color='#005b88', linestyle='--', linewidth=2)
    plt.text(mean_corr + 0.01, plt.ylim()[1] * 0.95, f'$\\bar{{r}}$ = {mean_corr:.2f}', 
            color='#005b88', fontsize=16, verticalalignment='top')
    plt.xlabel('Correlation between delta and P(Target|DT) - P(Target|CD)', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=16)  # Larger tick labels
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    #plt.savefig('correlation_histogram.png', dpi=300)
    plt.show()
    
    # === SUMMARY STATISTICS ===
    print("\n=== Correlation Statistics ===")
    print(f"Mean correlation: {np.mean(correlations):.3f}")
    print(f"Std correlation: {np.std(correlations):.3f}")
    print(f"Min correlation: {np.min(correlations):.3f}")
    print(f"Max correlation: {np.max(correlations):.3f}")
    print(f"Median correlation: {np.median(correlations):.3f}")
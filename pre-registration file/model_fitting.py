import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def trial_loglik(params, trial):
    epsilon = 1e-10

    delta = params[0]
    beta = params[1]
    # Expected values
    EV = {
        "T": trial["EV_target"],
        "C": trial["EV_competitor"],
        "D": trial["EV_decoy"]
    }
    
    # First pair
    first_pair = trial["first_pair"].strip("()").split(",")
    first_choice = trial["decoy_first_response_gamble"]
    
    # Stage 1 probability
    p_first = np.exp(beta * EV[first_choice]) / (np.exp(beta * EV[first_pair[0]]) + np.exp(beta *EV[first_pair[1]]))
    if first_choice != first_pair[0]:
        p_first = 1 - p_first
    
    # Stage 2 probability
    first_stage_winner_value = EV[first_choice] + delta
    third_option = [o for o in ["T","C","D"] if o not in first_pair][0]
    p_final = np.exp(beta * first_stage_winner_value) / (np.exp(beta * first_stage_winner_value) + np.exp(beta * EV[third_option]))
    if trial["decoy_final_response_gamble"] != first_choice:
        p_final = 1 - p_final
    
    return np.log(p_first + epsilon) + np.log(p_final + epsilon)

def neg_loglik(params, trials):
    return -sum(trial_loglik(params, t) for t in trials)

def null_neg_loglik(beta, trials):
    return -sum(trial_loglik(np.array([0,beta[0]]), t) for t in trials)

if __name__ == "__main__":  
    data = pd.read_csv("../Data/experiment2/data_for_model_fitting.csv")
    count_table = pd.crosstab(data['decoy_added_gamble'], data['decoy_final_response_gamble'])
    prop_table = count_table.div(count_table.sum(axis=1), axis=0)

    # Columns I want to use
    cols = ["id", "trial_number", "trial_set", "risky_amount", "risky_probability", "safe_amount", "safe_probability", "decoy_amount", "decoy_probability", 
        "decoy_first_response_gamble",  "decoy_final_response_gamble", "decoy_added_gamble"]
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
    cleaned_df = df.drop(columns=["risky_amount", "risky_probability", "safe_amount", "safe_probability", "trial_set",
                            "target_amount", "target_probability", "decoy_amount", "decoy_probability", "competitor_probability", "competitor_amount"])



    # Fit delta for a participant
    participants = cleaned_df['id'].unique()
    results = []
    rounds = 20
    beta_range = (0,1)
    delta_range = (-50,50)



    for pid in participants:
        df_pid = cleaned_df[cleaned_df['id'] == pid]
        trials = df_pid.to_dict('records')
        
        # Set best values
        best_delta_res = None
        best_delta_ll = -np.inf

        for attempt in range(rounds):
            start_delta = np.random.uniform(delta_range[0], delta_range[1])
            start_beta = np.random.uniform(beta_range[0], beta_range[1])


            # Minimize negative log-likelihood
            res_delta = minimize(lambda d: neg_loglik(d, trials), x0=[start_delta, start_beta], bounds=[delta_range, beta_range])

            # Update if need be
            current_ll = -res_delta.fun
            if current_ll > best_delta_ll:
                best_delta_ll = current_ll
                best_delta_res = res_delta 

        # Save outputs
        delta_delta = best_delta_res.x[0]
        delta_beta = best_delta_res.x[1]
        delta_ll = -best_delta_res.fun
        delta_aic = 2*2-(2*delta_ll) # Since there are 2 parameters

        # FOR NULL MODEL
        best_null_res = None
        best_null_ll = -np.inf

        for attempts in range(rounds):
            start_beta = np.random.uniform(beta_range[0], beta_range[1])
            # Run model
            res_null = minimize(lambda d: null_neg_loglik(d, trials), x0=[start_beta], bounds=[beta_range])
            # Update if need be
            current_ll = -res_null.fun
            if current_ll > best_null_ll:
                best_null_ll = current_ll
                best_null_res = res_null
        
        null_beta = best_null_res.x[0]
        null_ll = -best_null_res.fun
        null_aic = 2*1-(2*null_ll) # Since there are 2 parameters

        results.append({"id": pid, "delta_delta": delta_delta, "delta_beta": delta_beta, "delta_ll": delta_ll, "delta_AIC": delta_aic,
                        "null_beta": null_beta, "null_ll": null_ll, "null_AIC": null_aic})

    # Convert results to a DataFrame
    delta_df = pd.DataFrame(results)

    print("Which model performs better?")
    print("Null AIC:")
    print(np.mean(delta_df['null_AIC']))
    print("Delta AIC:")
    print(np.mean(delta_df['delta_AIC']))

    print("Are deltas mostly positive?")
    plt.hist(delta_df["delta_delta"])
    plt.xlabel("Delta")
    plt.title("Delta for full model")
    plt.show()

    # Run t-test test
    ttest_result = stats.ttest_1samp(delta_df["delta_delta"], 0, alternative='greater')
    print(f"P-value: {ttest_result.pvalue}") # Rejecting null hypothesis means they are stringly positive

    # Calculate the probability for each first pair that the selected the target
    grouped = cleaned_df.groupby(['id', 'first_pair'])
    prob_df = grouped['decoy_final_response_gamble'].apply(lambda x: (x == 'T').mean()).reset_index()
    prob_wide = prob_df.pivot(index='id', columns='first_pair', values='decoy_final_response_gamble')
    prob_pivot = prob_wide.reset_index()

    # Merge data frames
    result_df = pd.merge(delta_df, prob_pivot, on='id')


    # Sanity check: Do people with higher delta more likely to select target when (T,D) is first compared to (C,D)?
    result_df["(D,T) - (C,D)"] = result_df["(D,T)"] - result_df["(C,D)"]
    X = result_df["delta_delta"]
    y = result_df["(D,T) - (C,D)"]
    X_const = sm.add_constant(X)  # Add intercept
    model = sm.OLS(y, X_const).fit()
    print(f"Regression results based on difference in order:")
    print(model.summary())
    print("\n")

    # Plot result
    plt.scatter(result_df["delta_delta"], result_df["(D,T) - (C,D)"], color='blue', label='Data')
    X_vals = np.linspace(result_df["delta_delta"].min(), result_df["delta_delta"].max(), 100)
    y_vals = model.params.iloc[0] + model.params.iloc[1] * X_vals  # intercept + slope * X
    plt.plot(X_vals, y_vals, color='red', label='Regression line')

    plt.xlabel("Delta")
    plt.ylabel("P(Target|T,D) - P(Target|C,D)")
    plt.title("Linear regression of selecting target more when it is presented first compared to when competitor is presented first")
    plt.legend()
    plt.show()









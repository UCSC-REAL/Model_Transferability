# %%
import numpy as np
from EB_utils.pipeline_util import workhorse
from EB_utils.pipeline_util import dict_agent_history_record_retrieve as archive
import random
import pandas as pd
from utils import (
    compute_error,
    compute_optimal_threshold,
    compute_optimal_threshold_MinInduceRisk_LS,
    compute_upper_bound,
    compute_lower_bound,
    update_H_Y,
)
import warnings

warnings.filterwarnings("ignore")

random.seed(2023)
np.random.seed(2023)

n_agent = 3000
n_step = 6
M = 100
save_path_prefix = "creditscore_"

# %% hyperparameter
alpha_D = 0.01
alpha_Y = 0
scenario = "creditscore"
cardinality_protected = 4
decision_type = "accurate"
init_echelon_cdf_option = None
decision_feature_option = "all"
init_accept_p = 0.4

# %% data prep
dict_agent = workhorse(
    n_agent=n_agent,
    n_step=n_step,
    alpha_D=alpha_D,
    alpha_Y=alpha_Y,
    scenario=scenario,
    decision_type=decision_type,
    init_echelon_cdf_option=init_echelon_cdf_option,
    decision_feature_option=decision_feature_option,
    init_accept_p=init_accept_p,
)

population_H, population_Y_ori, population_Y_obs, population_D = archive(
    dict_agent=dict_agent,
    cardinality_protected=cardinality_protected,
    first_n_step=n_step,
    record_format="pd",
)

# %% repeated one-step bound computation
result_recorder_UB = pd.DataFrame(data=None, columns=["T", "Diff", "UB"])
result_recorder_LB = pd.DataFrame(data=None, columns=["T", "Max", "LB"])

for step_idx in range(n_step):
    print(step_idx)
    # get current echelon and Yori and form a df
    current_H = population_H[step_idx == population_H["T"]].drop(
        columns=["T", "ProtectedFeature"])
    current_Y = population_Y_ori[step_idx == population_Y_ori["T"]].drop(
        columns=["T", "ProtectedFeature"])
    df_current = pd.concat([current_H, current_Y], axis=1)  # [H, Y]

    # compute current statistical optimal
    (
        threshold_stat_optimal,
        error_stat_optimal_on_original_data,
    ) = compute_optimal_threshold(df_current, M=M)

    # compute induced optimal
    (
        threshold_induced_optimal,
        error_induced_optimal_on_induced_data,
    ) = compute_optimal_threshold_MinInduceRisk_LS(
        df_current, M=M, alpha_D=alpha_D)  # Err_hT_hT_LS
    # df_hT_LS
    df_induced_optimal_self_induce = update_H_Y(df_current,
                                                threshold_induced_optimal,
                                                alpha_D=alpha_D)

    # compute error of stat optimal on data induced by itself
    df_stat_optimal_self_induce = update_H_Y(df_current,
                                             threshold_stat_optimal,
                                             alpha_D=alpha_D)  # df_hS_LS
    error_stat_optimal_on_self_induced_data = compute_error(
        df_stat_optimal_self_induce, threshold_stat_optimal)  # Err_hS_hS_LS

    Diff = (error_stat_optimal_on_self_induced_data -
            error_induced_optimal_on_induced_data)
    UpperBound = compute_upper_bound(
        df_current,
        df_stat_optimal_self_induce,
        threshold_stat_optimal,
        df_induced_optimal_self_induce,
        threshold_induced_optimal,
    )

    Max_hS = max(error_stat_optimal_on_original_data,
                 error_stat_optimal_on_self_induced_data)

    LowerBound_hS, _ = compute_lower_bound(
        df_current,
        df_stat_optimal_self_induce,
        threshold_stat_optimal,
        df_induced_optimal_self_induce,
        threshold_induced_optimal,
    )

    # append result
    result_recorder_UB = result_recorder_UB.append(
        {
            "T": step_idx,
            "Diff": Diff,
            "UB": UpperBound
        }, ignore_index=True)
    result_recorder_LB = result_recorder_LB.append(
        {
            "T": step_idx,
            "Max": Max_hS,
            "LB": LowerBound_hS
        }, ignore_index=True)

# %% save them
with open(save_path_prefix + "UB.csv", "w") as f:
    result_recorder_UB.to_csv(f, index=False)

with open(save_path_prefix + "LB.csv", "w") as f:
    result_recorder_LB.to_csv(f, index=False)

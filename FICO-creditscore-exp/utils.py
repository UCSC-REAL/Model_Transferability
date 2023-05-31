import numpy as np
import pandas as pd


# compute average square error of a threshold on a dataframe df
def compute_error(df, threshold):
    """
    Input:
        - df: a dataset with a column for 'Q' already computed
        - a threshold

    output:
        - the mean squared error
    """
    error = 0
    # compute error
    for i in range(len(df)):
        # if df_result["x"][i]'s value is greater than the treshold then h(x) = +1; otherwise h(x) = 0
        error = (
            error
            + ((df.iloc[i]["Echelon"] > threshold) - df.iloc[i]["GroundTruthLabel"])
            ** 2
        )
    return error / len(df)


# compute the optimal threshold for the original distribution
def compute_optimal_threshold(df, M):
    """
    A function to learn the optimal threshold on the original dataset

    Input:
        - df: a dataset with a column for 'Q' already computed
        - M is the number of potential thresholds.

    Output:
        - the optimal threshold for df
        - lowest error
    """
    # compute the boundary of Q:
    Q_min, Q_max = np.min(df["Echelon"]), np.max(df["Echelon"])

    # loop through potential thresholds:
    error_list = []
    for i in range(M):
        threshold = Q_min + i * (Q_max - Q_min) / M
        error_list.append(compute_error(df, threshold))
    index_min = np.argmin(error_list)
    optimal_threshold = Q_min + index_min * (Q_max - Q_min) / M
    return optimal_threshold, np.min(error_list)


def compute_optimal_threshold_MinInduceRisk_LS(df, M, alpha_D=0.01):
    """
    find the optimal classifier achieve min Err(h)(h)
    where M is the number of potential classifiers to search for
    return the optimal threshold and the minimum error

    Input:
        - df: initial distribution with 'Q'
        - M: total number of threshold to loop through
        - alpha: ratio of Y = 1 instances for the initial distribution
        - mu0: mean for Y = 0 instances for the initial distribution
        - mu1: mean for Y = 1 instances for the initial distribution
        - sigma: variance for all instances for the initial distribution
        - bottom: lower bound for the truncated Gaussian variable
        - top: upper bound for the truncated Gaussian variable
    Output:
        - an optimal threshold
        - the corresponding error

    """
    N = len(df)
    # compute the boundary of Q:
    Q_min, Q_max = np.min(df["Echelon"]), np.max(df["Echelon"])
    # print(Q_min, Q_max)

    # loop through the potential thresholds
    error_list = []
    for j in range(M):
        threshold = Q_min + j * (Q_max - Q_min) / M
        # directly use echelon and update strategy (1 + alpha_D) to update echelon
        # df [H, Y]
        df_new = update_H_Y(df, threshold, alpha_D)

        # compute new data distribution's loss
        Err_h_h = compute_error(df_new, threshold)
        error_list.append(Err_h_h)

    index_min = np.argmin(error_list)
    min_error = np.min(error_list)

    # compute the optimal threshold on the dataset that it induced call it hT
    optimal_threshold = Q_min + index_min * (Q_max - Q_min) / M
    return optimal_threshold, min_error


def update_H_Y(df, threshold, alpha_D):
    array_H_Y = df.to_numpy()
    # update echelon according to alpha_D
    temp_H = array_H_Y[:, 0]
    decision = temp_H > threshold
    temp_H = temp_H * (1 + alpha_D * (2 * decision - 1))
    temp_H = np.maximum(1e-6, temp_H)
    temp_H = np.minimum(1 - 1e-6, temp_H)
    array_H_Y[:, 0] = temp_H

    n_sample = array_H_Y.shape[0]
    for i in range(n_sample):
        _echelon = temp_H[i]
        _y = np.random.binomial(n=1, p=_echelon)
        array_H_Y[i, 1] = _y

    df_new = pd.DataFrame(data=array_H_Y, columns=["Echelon", "GroundTruthLabel"])
    return df_new


# --> Upper bound on Diff
def compute_upper_bound(
    df_current,
    df_stat_optimal_self_induce,
    threshold_stat_optimal,
    df_induced_optimal_self_induce,
    threshold_induced_optimal,
):
    # compute w(hS), w(hT) and p
    w_hS_LS = (
        np.sum(df_stat_optimal_self_induce["Echelon"] > threshold_stat_optimal)
    ) / len(df_stat_optimal_self_induce)
    w_hT_LS = (
        np.sum(df_induced_optimal_self_induce["Echelon"] > threshold_induced_optimal)
    ) / len(df_induced_optimal_self_induce)
    p_LS = (np.count_nonzero(df_current["GroundTruthLabel"] == 1)) / len(df_current)

    # compute d_TV(D+(hS), D+(hT)) and d_TV(D-(hS), D-(hT))
    # Pr_{DS|Y = +1}(hS(X) = +1)
    P_DSY1_hS1 = np.sum(
        (df_current["Echelon"] > threshold_stat_optimal)
        & (df_current["GroundTruthLabel"] == 1)
    ) / np.sum(df_current["GroundTruthLabel"] == 1)
    # Pr_{DS|Y = +1}(hT(X) = +1)
    P_DSY1_hT1 = np.sum(
        (df_current["Echelon"] > threshold_induced_optimal)
        & (df_current["GroundTruthLabel"] == 1)
    ) / np.sum(df_current["GroundTruthLabel"] == 1)

    dtv_ST_plus = np.abs(P_DSY1_hS1 - P_DSY1_hT1)

    # print("dtv_ST_plus", dtv_ST_plus)

    # Pr_{DS|Y = 0}(hS(X) = +1)
    P_DSY0_hS1 = np.sum(
        (df_current["Echelon"] > threshold_stat_optimal)
        & (df_current["GroundTruthLabel"] == 0)
    ) / np.sum(df_current["GroundTruthLabel"] == 0)
    # Pr_{DS|Y = 0}(hT(X) = +1)
    P_DSY0_hT1 = np.sum(
        (df_current["Echelon"] > threshold_induced_optimal)
        & (df_current["GroundTruthLabel"] == 0)
    ) / np.sum(df_current["GroundTruthLabel"] == 0)

    dtv_ST_minus = np.abs(P_DSY0_hS1 - P_DSY0_hT1)

    # print("dtv_ST_minus", dtv_ST_minus)

    # compute the upper bound
    UB_LS = np.abs(w_hS_LS - w_hT_LS) + (1 + p_LS) * (dtv_ST_plus + dtv_ST_minus)
    # print("UB_LS", UB_LS)

    return UB_LS


def compute_lower_bound(
    df_current,
    df_stat_optimal_self_induce,
    threshold_stat_optimal,
    df_induced_optimal_self_induce,
    threshold_induced_optimal,
):
    # compute w(hS), w(hT) and p
    w_hS_LS = (
        np.sum(df_stat_optimal_self_induce["Echelon"] > threshold_stat_optimal)
    ) / len(df_stat_optimal_self_induce)
    w_hT_LS = (
        np.sum(df_induced_optimal_self_induce["Echelon"] > threshold_induced_optimal)
    ) / len(df_induced_optimal_self_induce)
    p_LS = (np.count_nonzero(df_current["GroundTruthLabel"] == 1)) / len(df_current)

    # lower bound for hS
    TPR_S_hS = np.sum(
        (df_current["Echelon"] > threshold_stat_optimal)
        & (df_current["GroundTruthLabel"] == 1)
    ) / np.sum(df_current["Echelon"] > threshold_stat_optimal)
    FPR_S_hS = np.sum(
        (df_current["Echelon"] > threshold_stat_optimal)
        & (df_current["GroundTruthLabel"] == 0)
    ) / np.sum(df_current["Echelon"] > threshold_stat_optimal)

    LB_hS = np.abs(p_LS - w_hS_LS) * (1 - np.abs(TPR_S_hS - FPR_S_hS)) / 2
    # print("LB_hS", LB_hS)

    # lower bound for hT
    TPR_S_hT = np.sum(
        (df_current["Echelon"] > threshold_induced_optimal)
        & (df_current["GroundTruthLabel"] == 1)
    ) / np.sum(df_current["Echelon"] > threshold_induced_optimal)
    FPR_S_hT = np.sum(
        (df_current["Echelon"] > threshold_induced_optimal)
        & (df_current["GroundTruthLabel"] == 0)
    ) / np.sum(df_current["Echelon"] > threshold_induced_optimal)

    LB_hT = np.abs(p_LS - w_hT_LS) * (1 - np.abs(TPR_S_hT - FPR_S_hT)) / 2
    # print("LB_hT", LB_hT)

    return LB_hS, LB_hT

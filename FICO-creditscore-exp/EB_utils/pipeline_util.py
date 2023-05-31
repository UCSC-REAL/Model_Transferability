# # --> pipeline related utils
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from EB_utils.Agent import Agent
from EB_utils.density_util import simulation_echelon_cdf
from EB_utils.density_util import creditscore_echelon_cdf
from EB_utils.density_util import predefine_feature_generate
import random

random.seed(42)
np.random.seed(42)


def workhorse(n_agent,
              n_step,
              alpha_D,
              alpha_Y,
              scenario,
              decision_type,
              init_echelon_cdf_option=None,
              decision_feature_option=None,
              init_accept_p=1):
    """Workhorse function to initialize and run multiple steps.

    Generate n_agent Agents, run n_step times (T = 1 ... n_step).
    For the purpose of balanced group samples, draw A from uniform.

    Initialization
        agent           initialize dynamics (A, H, X, Y_ori, Y_obs)
    Loop after initialization:
        | population    (mandatory for real data) get df_feature, df_Y_obs
        | population    (mandatory for real data) estimate H
        | population    fit decision-maker (if estimate H, i.e., 'accurate')
        | agent         register decision
        | agent         (debug) debug_check_echelon_estimate
        | population    (debug) get df_D
        | population    (debug) instantaneous evaluation
        | agent         register_dynamics
    Take history records and evaluate

    Args:
        `n_agent`: number of agent generated or recorded
        `n_step`: number of steps to run (T = 1 as start)
        `scenario`: 'simulation', 'creditscore', 'adult_race', 'adult_sex'
        `decision_type`: 'sim_perfect', 'accurate', 'CF'
            type of decision maker, see Agent.register_desicion()
        `init_echelon_cdf_option`: option for echelon cdf init
            For 'simulation', {'uniform', 'truncate_gaussian', 'trigonometric'}
            type of distribution used for simulation cdf initialization.
            For 'creditscore', None.
        `decision_feature_option`: 'all', 'CF' when fitting decision_model
            what feature to retrieve when fitting LR,
            see Agent.feature_label_retrieve()
        `init_accept_p`: see init_accept_p in (class) Agent.__init__()

    Return:
        `dict_agent`: dict of agents, each agent contains its history record.
    """
    dict_agent = dict()

    # simulation study
    if 'simulation' == scenario:
        # A binary
        population_A = np.random.choice(
            np.array([0, 1]), size=(n_agent, ))

        # define echelon cdf
        init_echelon_cdf = simulation_echelon_cdf(init_echelon_cdf_option)

        # --> initialize all agents
        for index in range(n_agent):
            # Agent require (not None == init_echelon_cdf)
            dict_agent[index] = Agent(index=index,
                                      scenario=scenario,
                                      protected_feature=population_A[index],
                                      feature_generate_func=predefine_feature_generate,
                                      init_echelon_cdf=init_echelon_cdf,
                                      init_accept_p=init_accept_p)

        # --> start loop
        for step in range(n_step):
            print(f'Step: {step}')
            # get df_feature and df_Y_obs
            df_feature, df_Y_obs = dict_agent_feature_label_retrieve_current(
                dict_agent, decision_feature_option)

            # (optional) actaully no need to estimate H in simulation
            # _all_feature, _ = dict_agent_feature_label_retrieve_current(
            #     dict_agent, 'all')
            # echelon_model = LogisticReg_fit_with_all(_all_feature, df_Y_obs)

            # fit decision-maker
            if 'CF' == decision_type:
                assert decision_type == decision_feature_option, \
                    'Invalid choice, here should be \'CF\'.'
                decision_model = LogisticReg_CF_level1(df_feature, df_Y_obs)
            elif decision_type in ('sim_perfect', 'accurate'):
                decision_model = None
                # 'sim_perfect' is used by default
                decision_type = 'sim_perfect'
            else:
                raise Exception('Invalid option for decision')

            # register decision (history record updated in Agent)
            for index in range(n_agent):
                dict_agent[index].register_decision(
                    decision_type, decision_model)

                # [debug] H_pred and H itself
                # dict_agent[index].debug_check_echelon_estimate(echelon_model)

            # get df_D
            df_D = dict_agent_decision_retrieve_current(dict_agent)

            # flatten to avoid issue
            df_Y_obs, df_D = df_Y_obs.flatten(), df_D.flatten()

            # [debug] instantaneous evaluate
            print(
                f'  Accuracy (ACC): {np.count_nonzero(df_D[-1 != df_Y_obs] == df_Y_obs[-1 != df_Y_obs]) / np.count_nonzero(-1 != df_Y_obs)}')

            # register dynamics
            for index in range(n_agent):
                dict_agent[index].register_dynamics(scenario=scenario,
                                                    feature_generate_func=predefine_feature_generate,
                                                    alpha_D=alpha_D,
                                                    alpha_Y=alpha_Y,
                                                    echelon_model=None)

    # creditscore data set
    elif 'creditscore' == scenario:
        # A cardinality = 4
        population_A = np.random.choice(
            np.array([0, 1, 2, 3, ]), size=(n_agent, ))

        # define echelon cdf
        init_echelon_cdf = creditscore_echelon_cdf()

        # --> initialize all agents
        for index in range(n_agent):
            # Agent require (not None == init_echelon_cdf)
            dict_agent[index] = Agent(index=index,
                                      scenario=scenario,
                                      protected_feature=population_A[index],
                                      feature_generate_func=predefine_feature_generate,
                                      init_echelon_cdf=init_echelon_cdf,
                                      init_accept_p=init_accept_p)

        # --> start loop
        for step in range(n_step):
            print(f'Step: {step}')
            # get df_feature and df_Y_obs
            df_feature, df_Y_obs = dict_agent_feature_label_retrieve_current(
                dict_agent, decision_feature_option)

            # need to estimate H
            _all_feature, _ = dict_agent_feature_label_retrieve_current(
                dict_agent, 'all')
            echelon_model = LogisticReg_fit_with_all(_all_feature, df_Y_obs)

            # fit decision-maker
            if 'CF' == decision_type:
                assert decision_type == decision_feature_option, \
                    'Invalid choice, here should be \'CF\'.'
                decision_model = LogisticReg_CF_level1(df_feature, df_Y_obs)
            elif 'accurate' == decision_type:
                decision_model = echelon_model  # fit with all
            else:
                raise Exception('Invalid option for decision')

            for index in range(n_agent):
                # register decision (history record updated in Agent)
                dict_agent[index].register_decision(
                    decision_type, decision_model)

            # get df_D
            df_D = dict_agent_decision_retrieve_current(dict_agent)

            # flatten to avoid issue
            df_Y_obs, df_D = df_Y_obs.flatten(), df_D.flatten()

            # [debug] instantaneous evaluate
            print(
                f'  Accuracy (ACC): {np.count_nonzero(df_D[-1 != df_Y_obs] == df_Y_obs[-1 != df_Y_obs]) / np.count_nonzero(-1 != df_Y_obs)}')

            # register dynamics
            for index in range(n_agent):
                dict_agent[index].register_dynamics(scenario=scenario,
                                                    feature_generate_func=predefine_feature_generate,
                                                    alpha_D=alpha_D,
                                                    alpha_Y=alpha_Y,
                                                    echelon_model=echelon_model)

    # [TODO] adult_race and adult_sex
    else:
        print('[TODO] Adult data set related in development')

    print('Done!')

    return dict_agent


def LogisticReg_fit_with_all(_all_feature, Y_obs):
    """Logistic Regressor fitted with all features.

    Call on group-level after initializing/registering dynamics for every agent.
    Can also be used for 'accurate' decision option.

    _all_feature = (A, X_not_protected_descendent, X_protected_descendent)

    Y_obs, binary, in {0, 1},
    for -1 entries, these agents did not receive approval previously

    [NOTE] Always train an LR to fit (A, X_all) -> Y_obs.
    Applicable to both simulation and real-world data.

    The purpose is to provide an estimate of echelon,
    see Agent.estimate_echelon().

    """
    LR = LogisticRegression()

    Y_obs = Y_obs.flatten()

    # only train on those whose Y_obs not -1, i.e., Y_obs \in {0, 1}
    feature_usable = _all_feature[-1 != Y_obs, :]

    # in case Y_obs \in {-1, 0}, or \in {-1, 1}
    # just randomize 10% of those 1 to 0 (otherwise cannot fit LR, only 1 class)
    _eps = 0.1
    if 0 == np.count_nonzero(0 == Y_obs):  # there is no default, Y_obs = 1
        print('NOTE: all observed Y_obs is 1, i.e., all previously approved credit are repaid!')
        loc_positive = np.array(np.where(1 == Y_obs)).flatten()
        number_to_flip = int(_eps * len(loc_positive)) + 1
        Y_obs[np.random.choice(loc_positive, size=number_to_flip)] = 0
    elif 0 == np.count_nonzero(1 == Y_obs):  # there is no repayment, Y_obs = 0
        print('NOTE: all observed Y_obs is 0, something is wrong!')
    else:
        pass

    LR.fit(feature_usable, Y_obs[-1 != Y_obs])

    return LR


def LogisticReg_CF_level1(X_not_protected_descendent, Y_obs):
    """Logistic Regressor fitted with X_not_protected_descendent.

    Call on group-level after initializing/registering dynamics for every agent.
    Can also be used for 'CF' decision option (level 1 implementation of CF).

    Only use X_not_protected_descendent, (X_not_protected_descendent) -> Y_obs.

    Y_obs, binary, in {0, 1},
    for -1 entries, these agents did not receive approval previously

    Applicable to both simulation and real-world data.

    [NOTE] Not for the purpose of echelon estimation.

    """
    LR = LogisticRegression()

    Y_obs = np.array(Y_obs).flatten()  # (-1, )

    # only train on those whose Y_obs not -1, i.e., Y_obs \in {0, 1}
    feature_usable = X_not_protected_descendent[-1 != Y_obs, :]

    # in case Y_obs \in {-1, 0}, or \in {-1, 1}
    # just randomize 10% of those 1 to 0 (otherwise cannot fit LR, only 1 class)
    _eps = 0.1
    if 0 == np.count_nonzero(0 == Y_obs):  # there is no default, Y_obs = 1
        print('NOTE: all observed Y_obs is 1, i.e., all previously approved credit are repaid!')
        loc_positive = np.array(np.where(1 == Y_obs)).flatten()
        number_to_flip = int(_eps * len(loc_positive)) + 1
        Y_obs[np.random.choice(loc_positive, size=number_to_flip)] = 0
    elif 0 == np.count_nonzero(1 == Y_obs):  # there is no repayment, Y_obs = 0
        print('NOTE: all observed Y_obs is 0, something is wrong!')
    else:
        pass

    LR.fit(feature_usable, Y_obs[-1 != Y_obs])

    return LR


def dict_agent_feature_label_retrieve_current(dict_agent,
                                              feature_retrieve_option='all'):
    """Retrieve current feature and label from dictionary of Agent.

    Just retrieve current feature and label, not the history record.
    See Agent.feature_label_retrieve()

    Returns:
        df_feature: (A, X_not_protected_descendent, X_protected_descendent)
            np.ndarray size = (n_agent, n_dim_all)
        df_Y_obs: Y_obs np.ndarray size = (n_agent, )
    """
    n_agent = len(dict_agent)
    l_feature = []
    l_Y_obs = []

    for index in range(n_agent):
        feature, label = dict_agent[index].feature_label_retrieve(
            feature_retrieve_option)
        l_feature.append(feature)
        l_Y_obs.append(label)

    return np.array(l_feature), np.array(l_Y_obs)


def dict_agent_decision_retrieve_current(dict_agent):
    """Retrieve current decision from dictionary of Agent.

    Just retrieve current decision, not the history record.

    Returns:
        df_D: D np.ndarray size = (n_agent, )
    """
    n_agent = len(dict_agent)
    l_D = []

    for index in range(n_agent):
        l_D.append(dict_agent[index].D)

    return np.array(l_D)


def dict_agent_history_record_retrieve(dict_agent,
                                       cardinality_protected,
                                       first_n_step,
                                       record_format='pd'):
    """Retrieve history records from dictionary of Agent.

    Archive format 'np' for np.ndarray, 'pd' for pd.DataFrame.
    (df is easier for sns.lineplot())

    Population information gathering.

    For 'np' return a dict, the key is index, the value is another dict.
    Organize df_ for different history on population level (condition on A).
    df_ (np.ndarray), size (n_agent, n_step)
    Use A as key and return a dict object.
    For example, dict_H_history = {0: df_H_history_disadv, 1: df_H_history_adv}
    E.g., dict[0] =
    { 'Echelon': np.ndarray (n_step, ),
      'GroundTruthLabel': np.ndarray (n_step, ),
      'ObservedLabel': np.ndarray (n_step, ),
      'Decision': np.ndarray (n_step, ) }

    For 'pd' return pd.DataFrame directly:
    pd_df_H: columns = ['T', 'Echelon', 'ProtectedFeature'],
    pd_df_Y_ori: columns = ['T', 'GroundTruthLabel', 'ProtectedFeature'],
    pd_df_Y_obs: columns = ['T', 'ObservedLabel', 'ProtectedFeature'],
    pd_df_D: columns = ['T', 'Decision', 'ProtectedFeature']

    """
    assert record_format in ('np', 'pd'), 'Invalid option.'

    def record_archive_init(cardinality_protected,
                            n_step,
                            record_format,
                            record_name):
        ''' for record initialization '''
        record_history = dict()
        if 'np' == record_format:
            for a in range(cardinality_protected):
                # ready for np.vstack
                record_history[a] = np.array([]).reshape(0, n_step)
        else:  # 'pd' == record_format:
            # create an empty DataFrame
            # object With column names only
            record_history = pd.DataFrame(
                columns=['T', str(record_name), 'ProtectedFeature'])

        return record_history

    def record_archive_update(record_history,
                              which_group,
                              agent_history,
                              record_format,
                              record_name):
        ''' update archieved record '''

        if 'np' == record_format:
            _history_archive = record_history[which_group]
            _history_archive = np.vstack((
                _history_archive,
                agent_history.reshape([1, -1])))

            record_history[which_group] = _history_archive

        else:  # 'pd' == record_format:
            # create column 'T'
            n_step = agent_history.size
            _array = np.hstack((
                np.arange(n_step).reshape(-1, 1),
                agent_history.reshape(-1, 1),
                which_group * np.ones(shape=(n_step, 1), dtype=np.int)))
            # pandas df
            _df_to_append = pd.DataFrame(
                _array, columns=['T', str(record_name), 'ProtectedFeature'])
            # update pandas df
            record_history = record_history.append(
                _df_to_append, ignore_index=True)

        return record_history

    # number of agent
    n_agent = len(dict_agent)

    # available length of history
    n_step_available = len(dict_agent[0].D_history)
    if first_n_step > n_step_available:
        raise Exception('Not enough history data recorded')
    n_step = min(n_step_available, first_n_step)

    # initialize history record dictionaries
    record_H_history = record_archive_init(
        cardinality_protected, n_step, record_format, 'Echelon')
    record_Y_ori_history = record_archive_init(
        cardinality_protected, n_step, record_format, 'GroundTruthLabel')
    record_Y_obs_history = record_archive_init(
        cardinality_protected, n_step, record_format, 'ObservedLabel')
    record_D_history = record_archive_init(
        cardinality_protected, n_step, record_format, 'Decision')

    # go over each agent
    for index in range(n_agent):
        # current agent
        _agent = dict_agent[index]
        _protected_feature = _agent.protected_feature
        _echelon, _Y_ori, _Y_obs, _decision = _agent.history_record_retrieve()

        # update history record dictionaries
        if 'np' == record_format:
            record_H_history = record_archive_update(
                record_H_history, _protected_feature, _echelon)
            record_Y_ori_history = record_archive_update(
                record_Y_ori_history, _protected_feature, _Y_ori)
            record_Y_obs_history = record_archive_update(
                record_Y_obs_history, _protected_feature, _Y_obs)
            record_D_history = record_archive_update(
                record_D_history, _protected_feature, _decision)
        else:  # 'pd' == record_format:
            record_H_history = record_archive_update(record_H_history,
                                                     _protected_feature,
                                                     _echelon,
                                                     'pd',
                                                     'Echelon')
            record_Y_ori_history = record_archive_update(record_Y_ori_history,
                                                         _protected_feature,
                                                         _Y_ori,
                                                         'pd',
                                                         'GroundTruthLabel')
            record_Y_obs_history = record_archive_update(record_Y_obs_history,
                                                         _protected_feature,
                                                         _Y_obs,
                                                         'pd',
                                                         'ObservedLabel')
            record_D_history = record_archive_update(record_D_history,
                                                     _protected_feature,
                                                     _decision,
                                                     'pd',
                                                     'Decision')

    # return population history record
    return record_H_history, record_Y_ori_history, record_Y_obs_history, record_D_history


def pd_acc_compute(record_Y_obs_history, record_D_history, n_step, cardinality_protected):
    """Compute Accuracy from pd format records.

    Among those whose Y_obs is not -1, for each step.

    record_Y_obs_history (pd.DataFrame):
        columns = ['T', 'ObservedLabel', 'ProtectedFeature']

    record_D_history (pd.DataFrame):
        columns = ['T', 'Decision', 'ProtectedFeature']

    See dict_agent_history_record_retrieve() for detail of record format.

    """
    # new data frame to save result
    df_accuracy = pd.DataFrame(
        columns=['T', 'Accuracy', 'ProtectedFeature'])

    # for each step: among Y_obs != -1, compute Accuracy (among observed)
    for idx_step in range(n_step):
        _Y_array = record_Y_obs_history[
            idx_step == record_Y_obs_history['T']
        ].to_numpy(dtype=np.int)[:, 1:]  # remove T column

        _D_array = record_D_history[
            idx_step == record_D_history['T']
        ].to_numpy(dtype=np.int)[:, 1:]  # remove T column

        # compute A dependent accuracy
        for value_a in range(cardinality_protected):
            _Y_temp = _Y_array[value_a == _Y_array[:, 1], :]
            _D_temp = _D_array[value_a == _D_array[:, 1], :]

            if 0 == np.count_nonzero(-1 != _Y_temp[:, 0]):
                _accuracy = 1  # manipulate
            else:
                _accuracy = np.count_nonzero(
                    _D_temp[-1 != _Y_temp[:, 0],
                            0] == _Y_temp[-1 != _Y_temp[:, 0], 0]
                ) / np.count_nonzero(-1 != _Y_temp[:, 0])

            # update pd df
            df_accuracy = df_accuracy.append({'T': idx_step,
                                              'Accuracy': _accuracy,
                                              'ProtectedFeature': value_a},
                                             ignore_index=True)

    return df_accuracy


def pd_approval_rate_compute(record_D_history, n_step, cardinality_protected):
    """Compute positive decision rate from pd format records.

    record_D_history (pd.DataFrame):
        columns = ['T', 'Decision', 'ProtectedFeature']

    See dict_agent_history_record_retrieve() for detail of record format.

    """
    # new data frame to save result
    df_approval_rate = pd.DataFrame(
        columns=['T', 'ConditionalApprovalRate', 'ProtectedFeature'])

    # for each step
    for idx_step in range(n_step):
        _D_array = record_D_history[
            idx_step == record_D_history['T']
        ].to_numpy(dtype=np.int)[:, 1:]  # remove T column

        # compute A dependent P(D = 1 | A = a)
        for value_a in range(cardinality_protected):
            _D_temp = _D_array[value_a == _D_array[:, 1], :]

            if 0 == np.count_nonzero(_D_array[:, 0]):
                _approval_rate = 0  # manipulate
            else:
                _approval_rate = np.count_nonzero(
                    1 == _D_temp[:, 0]) / _D_temp.shape[0]

            # update pd df
            df_approval_rate = df_approval_rate.append({'T': idx_step,
                                                        'ConditionalApprovalRate': _approval_rate,
                                                        'ProtectedFeature': value_a},
                                                       ignore_index=True)

    return df_approval_rate

# # --> data related utils
import os
import urllib.request
import numpy as np
import random

random.seed(42)
np.random.seed(42)


def data_fetch(scenario):
    """Fetch online data set.

    For 'creditscore', 'adult_race', and 'adult_sex'.

    [TODO] 'adult_*' data fetch and preprocess

    """
    if 'creditscore' == scenario:
        # --> 'creditscore' data
        # only fetch cdf data
        if not os.path.isfile('transrisk_cdf_by_race_ssa.csv'):
            urllib.request.urlretrieve(
                'https://raw.githubusercontent.com/fairmlbook/fairmlbook.github.io/master/code/creditscore/data/transrisk_cdf_by_race_ssa.csv',
                'transrisk_cdf_by_race_ssa.csv')
    elif 'adult_race' == scenario:
        # --> 'adult_race'
        pass
    elif 'adult_sex' == scenario:
        # --> 'adult_sex'
        pass
    else:
        raise Exception('Invalid option.')

# # --> figure related util
import numpy as np
import matplotlib.pyplot as plt
from EB_utils.density_util import inverse_cdf
from itertools import product as iter_product
import random

random.seed(42)
np.random.seed(42)


###########################################################
# For multi-step run
###########################################################


###########################################################
# [NOTE] for theory illustation only
###########################################################
def sample_from_mesh_pdf_2d(mesh_2d_pdf, n_sample):
    """Sample from the 2D pdf given via mesh weights.

    Pipeline (mesh should be dense enough)
    - Marginal of X: for each tick in x, sum over y
    - Sample from P_X: use inverse CDF to sample n_sample
    - Conditional of Y|X (compute as needed): fix x
    - Sample from P_Y|X=x

    Args:
        mesh_2d_pdf: np.ndarray, every element non-negative
            (could be non-normalized, or non-square)
        n_sample: number of sample

    """
    # default domain [0, 1]
    mesh_height = mesh_2d_pdf.shape[0]
    mesh_width = mesh_2d_pdf.shape[1]
    x_axis = np.linspace(0, 1, num=mesh_width)
    y_axis = np.linspace(0, 1, num=mesh_height)
    sample = np.zeros(shape=[n_sample, 2])

    # marginal PDF of X
    marginal_x = np.sum(mesh_2d_pdf, axis=0)

    # get inversed CDF
    cdf_x = inverse_cdf(x_axis, marginal_x)

    # sample from marginal of X
    x_sample = cdf_x(np.random.random_sample(size=(n_sample, )))
    sample[:, 0] = x_sample[...]

    # digitize to the bin, use it to get conditional Y|X
    x_index = np.digitize(x_sample, x_axis)  # where the sample falls

    for ith in range(x_index.shape[0]):
        # index of current x_sample
        _mesh_x_idx = x_index[ith]
        # find conditional PDF for Y|_x
        _conditional_pdf_y = mesh_2d_pdf[:, _mesh_x_idx - 1]  # column
        # inverse CDF
        conditional_cdf_y = inverse_cdf(y_axis, _conditional_pdf_y)
        # get sample from conditional
        sample[ith, 1] = conditional_cdf_y(np.random.random_sample())

    return sample


def mesh_pdf_2d_quantitative_assumption(gamma_up, gamma_low, mesh_bins=200):
    """Generate mesh 2D PDF that satisfies the quantitative assumption.

    """
    assert 2 == gamma_up + gamma_low, 'Sum of gamma should be 2.'

    mesh = np.ones(shape=(mesh_bins, mesh_bins))

    upper_triangle_index = np.triu_indices(
        mesh.shape[0], k=0)  # offset 0
    lower_triangle_index = np.tril_indices(
        mesh.shape[0], k=-1)  # offset -1

    # triangle assign value
    mesh[upper_triangle_index] = gamma_up
    mesh[lower_triangle_index] = gamma_low

    # (0, 0) coordinate origin at lower left
    mesh = np.rot90(mesh, k=2)  # need to rotate mesh

    return mesh


def mesh_pdf_2d_qualitative_assumption(mesh_bins=200):
    """Generate mesh 2D PDF that satisfies the qualitative assumption.

    Form of distribution, truncated multivariate Gaussian

    """
    _x = np.linspace(0, 1, mesh_bins)
    _y = np.linspace(0, 1, mesh_bins)
    coordinate = iter_product(_x, _y)
    mesh = np.array(list(map(
        pdf_multivariate_gaussian, coordinate)))
    mesh = mesh.reshape(mesh_bins, mesh_bins)

    # (0, 0) coordinate origin at lower left
    mesh = np.rot90(mesh, k=2)  # need to rotate mesh

    return mesh


def pdf_multivariate_gaussian(coordinate):
    """Compute PDF at the coordinate.

    coordinate = (f_0, f_1), tuple of size (2, ),
    use together with map()

    """
    # --> case vi
    sigma_matrix = np.array([[1.2, 0.4],
                             [0.15, 0.2]])
    mu_vector = np.array([[0.35],
                          [0.6]])

    # --> case xi
    # sigma_matrix = np.array([[0.2, 0.15],
    #                          [0.3, 0.9]])
    # mu_vector = np.array([[0.7],
    #                       [0.35]])

    # det_sigma
    det_sigma = np.linalg.det(sigma_matrix)

    # tuple to array
    _vector = np.array(coordinate).reshape(2, 1)

    coefficient_term = 1. / np.sqrt(
        np.square(2. * np.pi) * det_sigma)
    _vector = _vector - mu_vector
    exponential_term = np.exp(
        - 0.5 * np.matmul(np.matmul(_vector.T, np.linalg.inv(sigma_matrix)), _vector))

    return coefficient_term * exponential_term


def sim_echelon_pair_update(df_coordinate, alpha_D, alpha_Y, case=None):
    """For Case vi and xi, update (f(0, e), f(1, e)).

    case vi (D0 = Y0 = 0, D1 = Y1 = 1)
    f_0 -> f_0 ( 1 - alpha_D - alpha_Y ),
    f_1 -> min( 1, f_1 ( 1 + alpha_D + alpha_Y ) ).

    case xi (D0 = Y0 = 1, D1 = Y1 = 0)
    f_0 -> min( 1, f_0 ( 1 + alpha_D + alpha_Y ) ),
    f_1 -> f_1 ( 1 - alpha_D - alpha_Y ).

    df_coordinate (np.ndarray): size (n, 2), each row (f_0, f_1)



    """
    assert 0.5 > alpha_D, 'alpha_D should be less than 0.5'
    assert 0.5 > alpha_Y, 'alpha_Y should be less than 0.5'

    f_0, f_1 = df_coordinate[:, 0], df_coordinate[:, 1]

    if 'vi' == case:
        f_0 = f_0 * (1 - alpha_D - alpha_Y)
        f_1 = np.minimum(
            1 - 1e-6, f_1 * (1 + alpha_D + alpha_Y))
    elif 'xi' == case:
        f_0 = np.minimum(
            1 - 1e-6, f_0 * (1 + alpha_D + alpha_Y))
        f_1 = f_1 * (1 - alpha_D - alpha_Y)
    else:
        raise Exception('Invalid option for case')

    df_coordinate = np.hstack((
        f_0.reshape(-1, 1), f_1.reshape(-1, 1)))

    return df_coordinate


def sns_jointplot_with_correct_cbar(sns_jointplot):
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    # get current position
    pos_joint_ax = sns_jointplot.ax_joint.get_position()
    pos_marg_x_ax = sns_jointplot.ax_marg_x.get_position()
    # align joint_ax and marg_ax
    sns_jointplot.ax_joint.set_position(
        [pos_joint_ax.x0, pos_joint_ax.y0,
         pos_marg_x_ax.width, pos_joint_ax.height])
    # place the cbar
    sns_jointplot.fig.axes[-1].set_position(
        [.83, pos_joint_ax.y0, .07, pos_joint_ax.height])

    return sns_jointplot


def sns_jointplot_ticks_update(sns_jointplot, no_ticks):
    if True == no_ticks:
        sns_jointplot.ax_joint.set_xticks([0., 1.])
        sns_jointplot.ax_joint.set_yticks([0., 1.])
    else:
        pass
    return sns_jointplot

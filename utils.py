import numpy as np


def random_sampling_anchor(pbounds, anchor_num=50, random_seed=10):
    anchor_list = []
    np.random.seed(random_seed)
    for _ in range(anchor_num):
        sampling = []
        for i in range(len(pbounds)):
            sampling.append(np.random.rand() * (list(pbounds.values())[i][1] - list(pbounds.values())[i][0]))
        anchor_list.append(sampling)
    return anchor_list


def shrink_theta(pbound, kernel, linear_shrink_multiplier=0.5):
    """
    Shrink the length scale of a kernel

    :param pbound:
    :param kernel:
    :param linear_shrink_multiplier:
    :return:
    """
    xs = []

    anchor_point = random_sampling_anchor(pbound, anchor_num=10)
    total_origin_kernel = 0
    for i in anchor_point:
        for j in anchor_point:
            total_origin_kernel += kernel(i, j)
            xs.append(kernel(i, j))
    total_origin_kernel *= linear_shrink_multiplier

    xs = np.array(xs)
    from scipy.optimize import fsolve

    def func(theta):
        return (xs ** theta).sum() - total_origin_kernel

    root = fsolve(func, np.array([0.9]))
    return np.sqrt(1/root[0])

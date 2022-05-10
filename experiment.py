from problems.standard_test_problems import *
from problems.photocatalysis_problems import *
from surrogate_kernel_bayesian_optimization import SurrogateKernelBayesianOptimization
from bayes_opt.bayesian_optimization import BayesianOptimization

import pandas as pd
import os

test_problems = [
    Ackley,
    Branin,
    Eggholder,
    GoldsteinPrice,
    SixHumpCamel,
    Shekel,
    Hartmann6,
    Michalewicz,
    Rosenbrock,
    StyblinskiTang
]


photocatalysis_problem, input_columns = get_kuka_problem()
bound = {}
for i in input_columns:
    bound[i] = (0, 5)
photocatalysis_problem.bound = bound
photocatalysis_problem.name = 'photocatalysis_experiment'

path = r'C:\Users\darkn\PycharmProjects\SKBO\experiment_results'
import time
datetime = time.localtime()
folder_name = '{}_{}_{}_{}_{}'.format(str(datetime.tm_year), str(datetime.tm_mon), str(datetime.tm_mday),
                                      str(datetime.tm_hour), str(datetime.tm_min))
path = os.path.join(path, folder_name)


def check_model_test_problem(problem, BO, max_iter=50, save_result=True, comparison=False, path=path, acq='ucb', kappa=2.576):
    result_linear_custom = pd.DataFrame([])
    iter = 0
    path = path + '_' + acq + '_' + str(kappa)
    if not os.path.exists(path):
        os.makedirs(path)
    while iter < max_iter:
        if BO == BayesianOptimization:
            optimizer = BO(f=problem, pbounds=problem.bound)
        else:
            optimizer = BO(f=problem, pbounds=problem.bound, custom_list=None, for_comparison=comparison)
            #
        # try:
        optimizer.maximize(n_iter=100, acq=acq, kappa=kappa)
        result_linear_custom = result_linear_custom.append(pd.Series(optimizer.result_dataframe), ignore_index=True)
        iter += 1
        # except:
        #     print('Error occured in iteration {}'.format(iter))
        #     pass

    if save_result:
        result_linear_custom = pd.DataFrame(np.array((result_linear_custom)))

        if BO == BayesianOptimization:
            result_linear_custom.to_csv(os.path.join(path, 'sklearn_result_test_vanilla_{}.csv'.format(problem.name)))
        else:
            if optimizer.for_comparison:
                result_linear_custom.to_csv(os.path.join(path, 'sklearn_result_test_comparison_{}.csv'.format(problem.name)))
            else:
                result_linear_custom.to_csv(os.path.join(path, 'sklearn_result_test_{}.csv'.format(problem.name)))

# Experiments for SKBO


# comparisons = [False, True]
comparisons = [False]
# acq = 'ei'
acq = 'ucb'
kappa = 2.576

for comparison in comparisons:
    # for problem in test_problems:
    #     test_function = TestProblem(problem, minimize=True)
    #     check_model_test_problem(test_function, SurrogateKernelBayesianOptimization, comparison=comparison, acq=acq, kappa=kappa)
    check_model_test_problem(photocatalysis_problem, SurrogateKernelBayesianOptimization, comparison=comparison, acq=acq, kappa=kappa)

# Comparison of standard BO

# for problem in test_problems:
#     test_function = TestProblem(problem, minimize=True)
#     check_model_test_problem(test_function, BayesianOptimization, acq=acq, kappa=kappa)
# check_model_test_problem(photocatalysis_problem, BayesianOptimization, acq=acq, kappa=kappa)

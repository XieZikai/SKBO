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

path = r'C:\Users\darkn\PycharmProjects\SKBO\experiment_results'


def check_model_test_problem(problem, BO, max_iter=50, save_result=True, comparison=False, acq='ucb', path=path):
    result_linear_custom = pd.DataFrame([])
    iter = 0
    while iter < max_iter:
        if BO == BayesianOptimization:
            optimizer = BO(f=problem, pbounds=problem.bound)
        else:
            optimizer = BO(f=problem, pbounds=problem.bound, custom_list=None, for_comparison=comparison)
            #
        try:
            optimizer.maximize(n_iter=100, acq=acq)
            result_linear_custom = result_linear_custom.append(pd.Series(optimizer.result_dataframe), ignore_index=True)
            iter += 1
        except:
            print('Error occured in iteration {}'.format(iter))
            pass

    if save_result:
        result_linear_custom = pd.DataFrame(np.array((result_linear_custom)))

        if BO == BayesianOptimization:
            result_linear_custom.to_csv(os.path.join(path, 'sklearn_result_test_vanilla_{}_{}.csv'.format(problem.name, acq)))
        else:
            if optimizer.for_comparison:
                result_linear_custom.to_csv(os.path.join(path, 'sklearn_result_test_comparison_{}_{}.csv'.format(problem.name, acq)))
            else:
                result_linear_custom.to_csv(os.path.join(path, 'sklearn_result_test_{}_{}.csv'.format(problem.name, acq)))

# Experiments for SKBO


comparisons = [True, False]
acq = 'ei'

for comparison in comparisons:
    for problem in test_problems:
        test_function = TestProblem(problem, minimize=True)
        check_model_test_problem(test_function, SurrogateKernelBayesianOptimization, comparison=comparison, acq=acq)
    check_model_test_problem(photocatalysis_problem, SurrogateKernelBayesianOptimization, comparison=comparison, acq=acq)

# Comparison of standard BO

for problem in test_problems:
    test_function = TestProblem(problem, minimize=True)
    check_model_test_problem(test_function, BayesianOptimization, acq=acq)
check_model_test_problem(photocatalysis_problem, BayesianOptimization, acq=acq)

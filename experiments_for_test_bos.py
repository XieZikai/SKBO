from ensemble_regressors.rfbo import RFBO
from ensemble_regressors.change_kernel_bo import CKBO

from problems.standard_test_problems import *
from problems.photocatalysis_problems import *

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

path = r'/experiment_results'
import time

datetime = time.localtime()
folder_name = '{}_{}_{}_{}_{}'.format(str(datetime.tm_year), str(datetime.tm_mon), str(datetime.tm_mday),
                                      str(datetime.tm_hour), str(datetime.tm_min))
path = os.path.join(path, folder_name)


def check_model_test_problem(problem, BO, max_iter=50, save_result=True, path=path, acq='ucb',
                             kappa=2.576):
    result_linear_custom = pd.DataFrame([])
    error_record = pd.DataFrame([])
    anchor_error_record = pd.DataFrame([])
    iter = 0
    path = path + '_' + acq + '_' + str(kappa)
    if not os.path.exists(path):
        os.makedirs(path)
    while iter < max_iter:
        optimizer = BO(f=problem, pbounds=problem.bound)
        try:
            optimizer.maximize(n_iter=100, acq=acq, kappa=kappa, init_points=5)
            result_linear_custom = result_linear_custom.append(pd.Series(optimizer.result_dataframe, dtype=np.float64), ignore_index=True)
            error_record = error_record.append(pd.Series(optimizer.error_recorder, dtype=np.float64), ignore_index=True)
            anchor_error_record = anchor_error_record.append(pd.Series(optimizer.anchor_error_recorder, dtype=np.float64),
                                                             ignore_index=True)
            iter += 1
        except:
            print('Error occured in iteration {}'.format(iter))
            pass

    if save_result:
        result_linear_custom = pd.DataFrame(np.array(result_linear_custom))
        error_record = pd.DataFrame(np.array(error_record))
        anchor_error_record = pd.DataFrame(np.array(anchor_error_record))

        result_linear_custom.to_csv(os.path.join(path, BO.__name__+'_result_test_{}.csv'.format(problem.name)))
        error_record.to_csv(os.path.join(path, BO.__name__+'_error_record_{}.csv'.format(problem.name)))
        anchor_error_record.to_csv(
            os.path.join(path, BO.__name__+'_anchor_error_record_vanilla_{}.csv'.format(problem.name)))


for problem in test_problems:
    test_function = TestProblem(problem, minimize=True)
    check_model_test_problem(test_function, CKBO)
check_model_test_problem(photocatalysis_problem, CKBO)

import logging
import numpy as np
import csv
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, \
    UniformIntegerHyperparameter

# Import SMAC utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

import sys, os, glob
print(os.path.dirname(sys.path[0]))
sys.path.append(os.path.dirname(sys.path[0]))

from A2C.run_a2c import run_a2c_smac
from run_ple_utils import smac_parser


def a2c_arg_parser():
    parser = smac_parser()
    parser.add_argument('--nenvs', help='Number of envs', type=int, default=1)

    # Comment all variables which shall be optimized. They will be set by the SMAC agent.
    
    # parser.add_argument('--lr', help='Learning Rate', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of samples based on which gradient is updated', )
    # parser.add_argument('--activ_fcn', choices=['relu6', 'elu', 'mixed'], type=str, default='relu6',
    #                     help='Activation functions of network layers', )
    parser.add_argument('--gamma', help='Discount factor for discounting the reward', type=float, default=0.85)
    parser.add_argument('--units_shared_layer1', help='Units in first hidden layer which is shared', type=int,
                        default=24)
    parser.add_argument('--units_shared_layer2', help='Units in second hidden layer which is shared', type=int,
                        default=24)
    parser.add_argument('--units_policy_layer', help='Units in hidden layer in policy head', type=int, default=24)

    return parser.parse_args()


def a2c_smac_wrapper(**params):
    logdir = params["logdir"]

    a2c_output_dir = os.path.join(logdir, 'a2c_output{:02d}'.format(params["instance_id"]))
    if not os.path.isdir(a2c_output_dir):
        os.makedirs(a2c_output_dir)
    smac_output_dir = os.path.join(logdir, 'smac3_output{:02d}'.format(params["instance_id"]))

    def a2c_from_cfg(cfg):
        """ Creates the A2C algorithm based on the given configuration.

        :param cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
            Configuration containing the parameters.
            Configurations are indexable!
        :return: A quality score of the algorithms performance
        """
        # For deactivated paraeters the configuration stores None-values
        # This is not accepted by the a2c algorithm, hence we remove them.
        cfg = {k: cfg[k] for k in cfg if cfg[k]}

        # create run directory
        dir_list = glob.glob(os.path.join(a2c_output_dir, 'run*'))
        rundir = 'run{:02d}'.format(len(dir_list)+1) # + str(len(dir_list) + 1)

        params["logdir"] = os.path.join(a2c_output_dir, rundir)
        os.makedirs(params["logdir"])
        avg_perf, var_perf, max_return = run_a2c_smac(**params, **cfg)
        logger.info('average performance: %s' % avg_perf)
        logger.info('performance variance: %s' % var_perf)
        logger.info('maximum episode return: %s' % max_return)

        # SMAC is minimizing the objective no matter whether run_obj is set to "runtime" or "quality"
        score = - avg_perf # + 0.2 * var_perf - 0.5 * max_return
        logger.info('Quality measure of the current learned agent: %s\n' % score)
        return score

    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

    logger = logging.getLogger()
    logger.propagate = False  # no duplicate logging outputs
    fh = logging.FileHandler(os.path.join(logdir, 'smac.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s:%(name)s: %(message)s'))
    logger.addHandler(fh)

    # Build configuration space and define all hyperparameters
    cs = ConfigurationSpace()
    # batch_size = UniformIntegerHyperparameter("batch_size", 5, 60, default_value=50)
    lr = UniformFloatHyperparameter("lr", 1e-4, 1e-2, default_value=1e-3)
    #units_shared_layer1 = UniformIntegerHyperparameter("units_shared_layer1", 8, 100, default_value=24)
    #units_shared_layer2 = UniformIntegerHyperparameter("units_shared_layer2", 8, 100, default_value=24)
    #units_policy_layer = UniformIntegerHyperparameter("units_policy_layer", 8, 100, default_value=24)
    vf_coeff = UniformFloatHyperparameter("vf_coeff", 1e-2, 0.5, default_value=0.1)
    ent_coeff = UniformFloatHyperparameter("ent_coeff", 5e-6, 1e-4, default_value=1e-5)
    #gamma = UniformFloatHyperparameter("gamma", 0.6, 1., default_value=0.90)
    activ_fcn = CategoricalHyperparameter("activ_fcn", ['relu6', 'elu', 'mixed'], default_value='relu6')
    cs.add_hyperparameters([vf_coeff, ent_coeff, lr, activ_fcn])  # batch_size
#    cs.add_hyperparameters([units_shared_layer1, units_shared_layer2, units_policy_layer,
#                            vf_coeff, ent_coeff, gamma, lr, activ_fcn])  # batch_size

    # Create scenario object
    logger.info('Create scenario object')
    logger.info('Output_dir: %s' % smac_output_dir)
    # print(params["run_parallel"])
    if params["run_parallel"].lower() == "true":
        print("RUN PARALLEL")
        scenario = Scenario({"run_obj": "quality",      # we optimize quality of learned agent
                             "runcount-limit": params["runcount_limit"],     # Maximum function evaluations
                             "cs": cs,                  # configutation space
                             "deterministic": "true",
                             "output_dir": smac_output_dir,
                             "shared_model": True,
                             "input_psmac_dirs": os.path.join(logdir, 'smac3_output*')
                             })
    else:
        scenario = Scenario({"run_obj": "quality",  # we optimize quality of learned agent
                             "runcount-limit": params["runcount_limit"],  # Maximum function evaluations
                             "cs": cs,  # configutation space
                             "deterministic": "true",
                             "output_dir": smac_output_dir,
                             })

    # Optimize using a smac object:
    seed = np.random.RandomState(params["seed"])
    logger.info('Generate SMAC object')
    smac = SMAC(scenario=scenario, rng=seed, tae_runner=a2c_from_cfg)

    logger.info('Start optimizing algorithm configurations\n')
    optimized_cfg = smac.optimize()

    logger.info('##############################################')
    logger.info('Run training with best configuration again')
    logger.info('##############################################')
    optimized_performance = a2c_from_cfg(optimized_cfg)
    logger.info("Optimized config")
    for k in optimized_cfg:
        logger.info(str(k) + ": " + str(optimized_cfg[k]))
    logger.info("Optimized performance: %.2f" % optimized_performance)

    with open(os.path.join(logdir, 'opt_hyperparams.txt'), 'a') as f:
        for k in optimized_cfg:
            f.write(str(k) + ': ' + str(optimized_cfg[k]) + '\n')
        f.write("Optimized performance: %.2f\n\n" % optimized_performance)

    with open(os.path.join(logdir, 'opt_hyperparams.csv'), 'a') as f:
        labels = []
        for k in optimized_cfg:
            labels.append(str(k))
        labels.insert(0, 'performance')
        labels.insert(0, 'instance_id')
        writer = csv.DictWriter(f, fieldnames=labels)
        if params["instance_id"] == 1:
            writer.writeheader()
        optimized_cfg._values["performance"] = optimized_performance
        optimized_cfg._values["instance_id"] = params["instance_id"]
        writer.writerow(optimized_cfg._values)

    return optimized_cfg


def main():
    args = a2c_arg_parser()
    _ = a2c_smac_wrapper(**args.__dict__)


if __name__ == '__main__':
    main()


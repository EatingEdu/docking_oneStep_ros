import os
import sys      
sys.path.append("/home/fyt/project/SPC_MIQL/AlignIQL")

# pdb.set_trace()
# relative_path = "/home/fyt/project/iql_learn/implicit_q_learning/Airdocking/Airdocking_ros/raisim_env"
# rela = os.path.relpath(relative_path)
sys.path.append("/home/fyt/project/SPC_MIQL/AlignIQL/Airdocking/raisim_airdocking") 

import pdb
from datetime import datetime
from typing import Tuple

# from Airdocking.miql.Airdocking_env import *
# #from Airdocking.train.states.expert_dataset import *
# from Airdocking.test.test_params import *

import gym
from gym.spaces import Box

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

# import wrappers
from jaxrl5.agents import BCLearner, IQLLearner, DDPMIQLLearner, ALIGNIQLLearner, SACLearner, PixelBCLearner, MIQLLearner
from jaxrl5.wrappers import wrap_gym, wrap_gym_episode

from env_air_sb3.env_params import rew_coeff_sou #原始的环境位置
FLAGS = flags.FLAGS


flags.DEFINE_string('env_name', 'AirDocking-v6', 'Environment name.')
flags.DEFINE_string('save_dir', './airdocking_test/', 'Tensorboard logging dir.')
flags.DEFINE_string('model_type', 'miql',"policy name")

flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 100,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 2000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 1000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
config_flags.DEFINE_config_file(
    'miql_config',
    '/home/fyt/project/SPC_MIQL/AlignIQL/Airdocking/miql/Miql_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

# gym.envs.register(
#     id='AirDocking-v6',
#     entry_point='env_air_sb3.AMP_sample_contact_velRandom_0113_xz05rot1None:AMP',  # 指向您的环境类
# )

expert_path = "/home/fyt/project/iql_learn/dataset/Joint200"
# python train_offline.py --env_name=halfcheetah-medium-expert-v2 --config=configs/mujoco_config.py 
# python -m pdb train_offline.py --config=configs/airdocking.py
#model_path = "/mnt/workspace_fyt/iql_learn/implicit_q_learning/Airdocking/train/airdocking_test/AirDocking-v0/42/84000/"

#1024
# python test_airdocking.py --config /mnt/workspace_fyt/iql_learn/implicit_q_learning/configs/airdocking_0606_2.py
#model_path = "/home/fyt/project/iql_learn/implicit_q_learning/model/16000/"
#model_path = "/home/fyt/project/iql_learn/implicit_q_learning/model/none_500_forCollect_0618/7000/"
#model_path = "/home/fyt/project/iql_learn/implicit_q_learning/model/0711_none500forCollectPPO_souEnv/15000/"
#model_path = "/home/fyt/project/iql_learn/implicit_q_learning/model/0717_mixPPOJoint200_souEnv/15000/"
model_path= "/home/fyt/project/SPC_MIQL/AlignIQL/Airdocking/miql/model/0114_Miql_ours_numQs4_successRatio07_mergeOnOff_evalUpdate50/6500/" #硬件在环有问题？？

#20260117
model_path = "/home/fyt/project/iql_learn/implicit_q_learning/model/Miql_estimateF_MT_data4+6+7_envTTF/20750/"


# def make_env(env_name: str) -> Tuple[gym.Env]:
#     env = gym.make(env_name, 
#                    rew_coeff=rew_coeff_sou, 
#                    sense_noise='default',
#                    control_name = "forceThrustOmega",
#                    max_step=5000)
    
#     env = wrap_gym_episode(env)
#     return env

# def record_episode(episode_number):
#     return 10 <= episode_number <= 20



kwargs = {'actor_lr': 0.0003, 
          'critic_lr': 0.0003, 
          'discount': 0.99, 
          'expectile': 0.9, 
          'hidden_dims': (512, 256), 
          'tau': 0.005, 
          'temperature': 3.0, 
          'value_lr': 0.0003}
# obs_ = np.array([[-6.30428743e+00, -3.68640137e+00,  1.17203987e+00,
#          1.91885662e+00,  1.33362114e+00, -1.90059423e+00,
#          4.90867943e-02, -8.46407115e-01, -9.71967459e-01,
#          3.75728965e-01,  1.01020716e-01,  1.67156354e-01,
#         -5.59453011e-01,  7.63544083e-01,  8.61168385e-01,
#         -1.70433331e+01, -2.97126389e+01,  7.06873512e+00,
#         -2.60036916e-01, -6.69854581e-01, -9.16066051e-01,
#         -2.27493092e-01, -1.81356907e-01, -3.19415987e-01,
#          4.09910011e+00, -5.29071379e+00,  2.31503630e+00,
#          1.43541157e+00,  4.99348938e-01,  6.11504555e-01,
#          5.92014939e-02, -2.05986083e-01,  7.78089702e-01,
#          6.92137564e-03, -3.24607879e-01, -5.65777063e-01,
#         -1.00781590e-01,  5.22686899e-01,  9.53855753e-01,
#         -3.83684235e+01,  1.45762491e+01, -2.83903694e+01]], dtype=np.float32)
# act_ = np.array([[-0.56669307,  4.7406673 , -5.596482  ,  4.626631  ,  0.89169085,
#         -1.9346521 ,  3.3952477 ,  3.229093  ]], dtype=np.float32)

#pdb.set_trace()
#env = make_env("AirDocking-v6")
def getSpace():
    low = np.array([
            -10., -10., -5.2, -3., -3., -3.,
            -1., -1., -1., -1., -1., -1.,
            -1., -1., -1.,
            -40., -40., -40.,
            -1., -1., -1., -1., -1., -1.,
            -10., -10., -5.2, -3., -3., -3.,
            -1., -1., -1., -1., -1., -1.,
            -1., -1., -1.,
            -40., -40., -40.
        ], dtype=np.float32)

    high = np.array([
            10., 10., 5.2, 3., 3., 3.,
            1., 1., 1., 1., 1., 1.,
            1., 1., 1.,
            40., 40., 40.,
            1., 1., 1., 1., 1., 1.,
            10., 10., 5.2, 3., 3., 3.,
            1., 1., 1., 1., 1., 1.,
            1., 1., 1.,
            40., 40., 40.
        ], dtype=np.float32)

    obs_space = Box(low=low, high=high, dtype=np.float32)

    low = np.array([
            -1., -6.2831855, -6.2831855, -6.2831855,
            -1., -6.2831855, -6.2831855, -6.2831855
        ], dtype=np.float32)

    high = np.array([
            0.939, 6.2831855, 6.2831855, 6.2831855,
            0.939, 6.2831855, 6.2831855, 6.2831855
        ], dtype=np.float32)

    action_space = Box(low=low, high=high, dtype=np.float32)
    return obs_space, action_space
    
obs_space, action_space = getSpace()
agent = MIQLLearner.create(42,
                    obs_space, 
                    action_space, 
                    **kwargs)
agent = agent.load(model_path)
print(f"{model_path} load finish !!!")

def modelPredict(state_error):
    #pdb.set_trace()
    action,_ = agent.eval_actions(np.array(state_error, dtype=np.float32))
    #action = np.zeros(8)
    return action

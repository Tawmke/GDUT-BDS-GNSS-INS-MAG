import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    # experiment setup for RL gnss positioning
    parser.add_argument('--training_stepnum', default=40000, type=int) #40000
    parser.add_argument('--domain_name', default='walker')
    parser.add_argument('--task_name', default='walk')
    parser.add_argument('--traj_type', default='urban', type=str)
    parser.add_argument('--learning_rate_list', default=[8e-5,1e-4], type=list) #[1e-5,5e-5,1e-4,5e-4,1e-3,5e-3]
    parser.add_argument('--running_date', default='test', type=str) #0423_soda_sharelstm/0420
    parser.add_argument('--DA_mode', default='SODA', type=str) #SODA origin RAD SIM
    parser.add_argument('--runtimes', default=1, type=int) #重复实验次数
    parser.add_argument('--Augment', default='RAS-S', type=str) #GN,RAS-S 数据增强的方式
    parser.add_argument('--SNR_list', default=[40], type=int) #[40,30,20,10] 信噪比
    parser.add_argument('--min', default=0.5, type=int)
    parser.add_argument('--max', default=1.5, type=int)

    # 这里下面的不用看
    # agent
    parser.add_argument('--algorithm', default='sac', type=str)
    parser.add_argument('--train_steps', default='500k', type=str)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)

    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)

    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)

    # architecture
    parser.add_argument('--num_shared_layers', default=11, type=int)
    parser.add_argument('--num_head_layers', default=0, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--projection_dim', default=100, type=int)
    parser.add_argument('--encoder_tau', default=0.05, type=float)

    # entropy maximization
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)

    # auxiliary tasks
    parser.add_argument('--aux_lr', default=1e-3, type=float)
    parser.add_argument('--aux_beta', default=0.9, type=float)
    parser.add_argument('--aux_update_freq', default=2, type=int)

    # soda
    parser.add_argument('--soda_batch_size', default=256, type=int)
    parser.add_argument('--soda_tau', default=0.005, type=float)

    # svea
    parser.add_argument('--svea_alpha', default=0.5, type=float)
    parser.add_argument('--svea_beta', default=0.5, type=float)

    # eval
    parser.add_argument('--save_freq', default='100k', type=str)
    parser.add_argument('--eval_freq', default='10k', type=str)
    parser.add_argument('--eval_episodes', default=30, type=int)
    parser.add_argument('--distracting_cs_intensity', default=0., type=float)

    # misc
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--save_video', default=False, action='store_true')

    args = parser.parse_args()

    return args

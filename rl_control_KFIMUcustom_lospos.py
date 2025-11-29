import numpy as np
import pandas as pd
#from env.GSDC_2022_LOSPOS import *
from envRLKF.GSDC_2022_LOSPOSIMU_KF import * # RL环境
# from env.dummy_cec_env_custom import *
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
from model.ppo import PPO
from model.ppo_recurrent_ATF1_AKF import RecurrentPPO
from env.env_param import *
from funcs.utilis_eskf import *
from funcs.PPO_SR import *
from model.model_ATF_KF import *
from collections import deque
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
training_stepnum = 25000 # 30000
# parameter settings
learning_rate_list = [8e-5] #[5e-4,1e-4 8e-5,3e-5]
postraj_num_list = [10] # [10,20,30,40]
# traj type: full 79 urban 32 highway 47 losangel 21 bayarea 58
alltraj = True # use all traj data for training and testing
triptype = 'pixel567urban' # smurban pixel567urban
if alltraj:
    traj_type_target_train = [0, 0.7]  # 轨迹数据的比例
    traj_type_target_test = [0.7, 1]
else:
    traj_type_target_train = [0, 1]  # 轨迹数据的比例
    traj_type_target_test = [0, 1]

trajdata_sort='sorted' # 'randint' 'sorted'
# baseline for RL: bl, wls, kf, kf_igst
baseline_mod='kf' # baseline方法 wls_igst kf_igst
# around 37.52321	-122.35447 scale 1e-5 lat 1.11m, lon 0.88m
# continuous action settings
max_action=100 #最大动作的范围
continuous_actionspace=[-max_action,max_action]
# discrete action settings
discrete_actionspace=7
action_scale = 3e-1 # scale in meters
"""
define parameter for RLKF
envmode: 
1. losposcovR_onlyRNallcorrect: use los pos convR to learn R and correct in the end (or obs)
2. poscovQ_onlyQNallcorrect: use pos convQ to learn Q and correct in the end (or vel)
"""
envmod = 'poscovQ_onlyQNallcorrect'
noise_scale_dic = {'process':1e-4,'measurement':1e-4} # {'process':5e-6,'measurement':0.01}
conv_corr = 'conv_corr_2' # conv_corr_1 conv_corr_2
continuous_Vaction_scale=10e-1 # 过程噪声估计 动作尺度
continuous_action_scale=20e-1 # 测量噪声估计 动作尺度

allcorrect = False # if correct in the end or correct in obs
network_unit = 128 # 神经元数量
net_archppo = [network_unit, network_unit]
ent_coef = 0.0
running_date = 'RL4ESKFIMU_alltraj' # 设置保存路径 RL4ESKFIMU_halftest RL4ESKFIMU_alltraj
reward_setting = 'RMSEadv_re' # 'RMSE' ‘RMSEadv' 'RMSEadv_eskf' 'RMSEadv_re'
detail = f'{reward_setting}_{conv_corr}' # _reversetraj

# select network and environment
discrete_lists=['discrete','discrete_A2C','discrete_lstm','ppo_discrete']
continuous_lists=['continuous','continuous_lstm','continuous_lstm_custom','ppo','continuous_custom','continuous_lstmATF1']
custom_lists=['ppo_discrete','ppo']
networkmod='continuous_lstmATF1'
# recording parameters
# data cata: KF: LatitudeDegrees; robust WLS: LatitudeDegrees_wls; standard WLS: LatitudeDegrees_bl
# parameters for customized ppo
# test settings
moretests=True #True False

if triptype == 'highway':
    tripIDlist = traj_highway
    moreteststypelist = ['highway']
elif triptype == 'urban':
    tripIDlist = traj_urban
    moreteststypelist = ['urban']
elif triptype == 'semiurban':
    tripIDlist = traj_semiurban
    moreteststypelist = ['semiurban']
elif triptype == 'xiaomi':
    tripIDlist = traj_xiaomi
    moreteststypelist = ['xiaomi']
elif triptype == 'xiaomiurban':
    tripIDlist = traj_xiaomiurban
    moreteststypelist = ['xiaomiurban']
elif triptype == 'pixel5urban':
    tripIDlist = traj_pixel5urban
    moreteststypelist = ['pixel5urban']
elif triptype == 'pixelurban':
    tripIDlist = traj_pixelurban
    moreteststypelist = ['pixelurban']
elif triptype == 'pixel4urban':
    tripIDlist = traj_pixel4urban
    moreteststypelist = ['pixel4urban']
elif triptype == 'pixel567urban':
    tripIDlist = traj_pixel567urban
    moreteststypelist = ['pixel567urban']
elif triptype == 'smurban':
    tripIDlist = traj_smurban
    moreteststypelist = ['smurban']

net_archppo = [network_unit, network_unit]

# path for testing
posnum_test = 10
onlytesting = True
interrupt_dic = None # {'start_ratio':0.5,'time':100} # 中断设置，开始中断的时间位置，中断持续时间
testdate = 'RL4ESKFIMU_alltraj'
if onlytesting:
    model_basefolder = 'source=pixel567urban_0.7_1_poscovQ_onlyQNallcorrect_kf_continuous_lstmATF1'
    model_basefolder=f'{dir_path}/records_values/{testdate}/{model_basefolder}'
    model_folderlist=os.listdir(model_basefolder)
    model_folderlist.sort()
    model_folderlist = ['lr=8e-05_pos=10_QS=0.0005_VAS=0.8_RMSEadv_re_conv_corr_2'] # only for testing

# Two agent working for RL env
Twoagent_testing = False
if Twoagent_testing:
    if triptype == 'pixel567urban':
        R_model_basefolder = 'source=pixel567urban_0.7_1_losposcovR_onlyRNallcorrect_kf_continuous_lstmATF1'
        R_model_basefolder = f'{dir_path}/records_values/{testdate}/{R_model_basefolder}'
        Q_model_basefolder = 'source=pixel567urban_0.7_1_poscovQ_onlyQNallcorrect_kf_continuous_lstmATF1'
        Q_model_basefolder = f'{dir_path}/records_values/{testdate}/{Q_model_basefolder}'
        R_model_folder = 'lr=8e-05_pos=10_QS=1e-05_RS=0.0005_XAS=0.8_RMSEadv_re_conv_corr_2' # lr=8e-05_pos=10_RS=0.0001_XAS=1.5_RMSEadv_re_conv_corr_2
        Q_model_folder = 'lr=8e-05_pos=10_QS=0.0005_VAS=0.8_RMSEadv_re_conv_corr_2'
        QR_basefolder = 'source=pixel567urban_0.7_1_losposconvQR_QRcorrect_fullobs_kf_continuous_lstmATF1'
        QR_model_basefolder = f'{dir_path}/records_values/{testdate}/{QR_basefolder}'
    elif triptype == 'smurban':
        R_model_basefolder = 'source=smurban_0.7_1_losposcovR_onlyRNallcorrect_kf_continuous_lstmATF1'
        R_model_basefolder = f'{dir_path}/records_values/{testdate}/{R_model_basefolder}'
        Q_model_basefolder = 'source=smurban_0.7_1_poscovQ_onlyQNallcorrect_kf_continuous_lstmATF1'
        Q_model_basefolder = f'{dir_path}/records_values/{testdate}/{Q_model_basefolder}'
        R_model_folder = 'lr=8e-05_pos=10_RS=0.0001_XAS=0.8_RMSEadv_re_conv_corr_2'
        Q_model_folder = 'lr=8e-05_pos=10_QS=5e-05_VAS=1.5_RMSEadv_re_conv_corr_2'
        QR_basefolder = 'source=smurban_0.7_1_losposconvQR_QRcorrect_fullobs_kf_continuous_lstmATF1'
        QR_model_basefolder = f'{dir_path}/records_values/{testdate}/{QR_basefolder}'

if alltraj:
    ratio = 1
    trajdata_range = [0, len(tripIDlist) - 1]
else:
    ratio = 0.5 # 一半训练一半测试
    trajdata_range= [0,int(np.ceil(len(tripIDlist)*ratio))] # train with half of data

# trajnum_test = 45
# trajdata_range = [trajnum_test,trajnum_test]
# detail = f'trajnum={trajnum_test}'

if networkmod in discrete_lists:
    print(f'Action scale {action_scale:8.2e}, discrete action space {discrete_actionspace}')
elif networkmod in continuous_lists:
    print(f'Action scale {continuous_action_scale:8.2e}, contiuous action space from {continuous_actionspace[0]} to {continuous_actionspace[1]}')

if onlytesting == False:
    for learning_rate in learning_rate_list:
        for posnum in postraj_num_list:
            QS,RS = noise_scale_dic['process'],noise_scale_dic['measurement']

            if envmod == 'lospos_convQR':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR(trajdata_range, traj_type_target_train, triptype, continuous_action_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr)])
                encoder = CustomATF1_AKFRL

            elif envmod == 'losposconvR_onlyR':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyR(trajdata_range, traj_type_target_train, triptype, continuous_action_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr)])
                encoder = CustomATF1_AKFRL_losposcovR  # CustomATF1_AKFRL_measurementR
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_pos = obs['pos'].shape[-1]
                features_dim_R = obs['R_noise'].shape[-1]
                policy_dim = features_dim_gnss + features_dim_pos + features_dim_R
                policy_kwargs = dict(features_extractor_class=encoder,
                                     features_extractor_kwargs=dict(features_dim=policy_dim),ATF_trig=networkmod)

            elif envmod == 'losconvR_onlyR':
                env = DummyVecEnv([lambda: GPSPosition_continuous_los_convQR_onlyR(trajdata_range,traj_type_target_train, triptype, continuous_action_scale,
                                                                                      continuous_actionspace,  reward_setting, trajdata_sort, baseline_mod, posnum,noise_scale_dic, conv_corr)])
                encoder = CustomATF1_AKFRL_loscovR  # CustomATF1_AKFRL_measurementR
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_R = obs['R_noise'].shape[-1]
                policy_kwargs = dict(features_extractor_class=encoder,
                                     features_extractor_kwargs=dict(features_dim=features_dim_R + features_dim_gnss),ATF_trig=networkmod)

            elif envmod == 'losposconvQ_onlyQ':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQ(trajdata_range,traj_type_target_train, triptype, continuous_action_scale,
                                                                                      continuous_actionspace,  reward_setting, trajdata_sort, baseline_mod, posnum,noise_scale_dic, conv_corr)])
                encoder = CustomATF1_AKFRL_losposcovQ  # CustomATF1_AKFRL_measurementR
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_pos = obs['pos'].shape[-1]
                features_dim_Q = obs['Q_noise'].shape[-1]
                net_arch = [network_unit, network_unit]
                policy_kwargs = dict(features_extractor_class=encoder, features_extractor_kwargs=dict(features_dim=features_dim_gnss+features_dim_Q + features_dim_pos),
                                     ATF_trig=networkmod,net_arch=net_arch)

            elif envmod == 'losposcovR_onlyRNallcorrect':
                tensorboard_log = f'{dir_path}records_values/{running_date}/source={triptype}_{traj_type_target_train[1]}_{ratio}_{envmod}_{baseline_mod}_{networkmod}/' \
                                  f'lr={learning_rate}_pos={posnum}_RS={RS}_XAS={continuous_action_scale}_{detail}'
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyRNallcorrect(trajdata_range, traj_type_target_train, triptype, continuous_action_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr, allcorrect=allcorrect)])
                encoder = CustomATF1_AKFRL_losposcovRinnovation
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_pos = obs['pos'].shape[-1]
                features_dim_R = obs['R_noise'].shape[-1]
                features_dim_ino = obs['innovation'].shape[-1]
                policy_dim = features_dim_gnss + features_dim_pos + features_dim_R + features_dim_ino
                net_arch = [network_unit, network_unit]
                policy_kwargs = dict(features_extractor_class=encoder,  features_extractor_kwargs=dict(features_dim=policy_dim),
                                     ATF_trig=networkmod, net_arch=net_arch)

            elif envmod == 'poscovQ_onlyQNallcorrect' or envmod == 'losposcovQ_onlyQNallcorrect':
                tensorboard_log = f'{dir_path}records_values/{running_date}/source={triptype}_{traj_type_target_train[1]}_{ratio}_{envmod}_{baseline_mod}_{networkmod}/' \
                                  f'lr={learning_rate}_pos={posnum}_QS={QS}_VAS={continuous_Vaction_scale}_{detail}'
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQNallcorrect(trajdata_range, traj_type_target_train, triptype, continuous_Vaction_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr, allcorrect=allcorrect)])
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_pos = obs['pos'].shape[-1]
                features_dim_Q = obs['Q_noise'].shape[-1]
                features_dim_ino = obs['innovation'].shape[-1]
                encoder = CustomATF1_AKFRL_poscovQinnovation
                policy_dim = features_dim_pos + features_dim_Q + features_dim_ino
                if envmod == 'losposcovQ_onlyQNallcorrect':
                    tensorboard_log = f'{dir_path}records_values/{running_date}/source={triptype}_{traj_type_target_train[1]}_{ratio}_{envmod}_{baseline_mod}_{networkmod}/' \
                                      f'lr={learning_rate}_pos={posnum}_QS={QS}_VAS={continuous_Vaction_scale}_{detail}'
                    encoder = CustomATF1_AKFRL_losposcovRinnovation
                    policy_dim = features_dim_pos + features_dim_Q + features_dim_gnss + features_dim_ino
                net_arch = [network_unit, network_unit]
                policy_kwargs = dict(features_extractor_class=encoder,  features_extractor_kwargs=dict(features_dim=policy_dim),
                                     ATF_trig=networkmod, net_arch=net_arch)

            elif envmod == 'losposcovQR_QRNallcorrect':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_QRNallcorrect(trajdata_range, traj_type_target_train, triptype, continuous_action_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr, allcorrect=allcorrect)])
                encoder = CustomATF1_AKFRL
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_pos = obs['pos'].shape[-1]
                features_dim_R = obs['R_noise'].shape[-1]
                features_dim_Q = obs['Q_noise'].shape[-1]
                policy_dim = features_dim_gnss + features_dim_pos + features_dim_R + features_dim_Q
                net_arch = net_archppo
                policy_kwargs = dict(features_extractor_class=encoder,  features_extractor_kwargs=dict(features_dim=policy_dim),
                                     ATF_trig=networkmod, net_arch=net_arch)

            elif envmod == 'losposcovR_RNobsendcorrect':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_RNobsendcorrect(trajdata_range, traj_type_target_train, triptype, continuous_action_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr, allcorrect=allcorrect)])
                encoder = CustomATF1_AKFRL_losposcovR
                obs = env.reset()
                features_dim_gnss = obs['gnss'].shape[-1]
                features_dim_pos = obs['pos'].shape[-1]
                features_dim_R = obs['R_noise'].shape[-1]
                features_dim_Q = obs['Q_noise'].shape[-1]
                policy_dim = features_dim_gnss + features_dim_pos + features_dim_R
                policy_kwargs = dict(features_extractor_class=encoder,  features_extractor_kwargs=dict(features_dim=policy_dim),ATF_trig=networkmod)

            elif envmod == 'lospos':
                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos(trajdata_range, traj_type_target_train, triptype, continuous_action_scale, continuous_actionspace,
                                                                       reward_setting,trajdata_sort,baseline_mod, posnum, noise_scale_dic, conv_corr)])
                encoder = CustomATF1

            model = RecurrentPPO("MlpLstmPolicy", env, verbose=2, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log, learning_rate=learning_rate, ent_coef=ent_coef)
            model.learn(total_timesteps=training_stepnum, eval_log_path=tensorboard_log)

            #print and save training results
            logdirname=model.logger.dir+'/train_'
            # logdirname='./'
            print('Training finished.')

            #record model
            # params=model.get_parameters()
            if networkmod in discrete_lists:
                model.save(model.logger.dir+f"/{networkmod}_{reward_setting}_action{discrete_actionspace}_{action_scale:0.1e}_trainingnum{training_stepnum:0.1e}"
                                            f"_env_{baseline_mod}{envmod}range{trajdata_range[0]}_{trajdata_range[-1]}{trajdata_sort}_lr{learning_rate:0.1e}")
            elif networkmod in continuous_lists:
                model.save(model.logger.dir+f"/{networkmod}_{reward_setting}_action{continuous_actionspace[0]}_{continuous_actionspace[1]}"
                                            f"_{continuous_action_scale:0.1e}_trainingnum{training_stepnum:0.1e}"
                                            f"_env_{baseline_mod}{envmod}range{trajdata_range[0]}_{trajdata_range[-1]}{trajdata_sort}_lr{learning_rate:0.1e}")

            recording_results_ecef_RL4KF(dir_path, data_truth_dic,trajdata_range,tripIDlist,logdirname,baseline_mod,traj_record=False)

            # more tests
            if moretests:
                for testtype in moreteststypelist:
                    print(f'more test for {testtype} env begin here')
                    if testtype == 'highway':
                        tripIDlist_test = traj_highway
                    elif testtype == 'urban':
                        tripIDlist_test = traj_urban
                    elif testtype == 'xiaomi':
                        tripIDlist_test = traj_xiaomi
                    elif testtype == 'xiaomiurban':
                        tripIDlist_test = traj_xiaomiurban
                    elif testtype == 'pixel5urban':
                        tripIDlist_test = traj_pixel5urban
                    elif testtype == 'pixelurban':
                        tripIDlist_test = traj_pixelurban
                    elif testtype == 'pixel4urban':
                        tripIDlist_test = traj_pixel4urban
                    elif testtype == 'pixel567urban':
                        tripIDlist_test = traj_pixel567urban
                    elif testtype == 'smurban':
                        tripIDlist_test = traj_smurban

                    if alltraj:
                        more_test_trajrange = [0, len(tripIDlist) - 1]
                    else:
                        more_test_trajrange = [int(np.ceil(len(tripIDlist_test)*ratio))+1, len(tripIDlist_test) - 1]

                    if testtype == triptype:
                        traj_type = traj_type_target_test  # 独立同分布测试
                    else:
                        traj_type = [0, 1]  # 域外分布测试范围

                    test_trajlist=range(more_test_trajrange[0],more_test_trajrange[-1]+1)#[0,1,2,3,4,5]
                    for test_traj in test_trajlist:
                        test_trajdata_range = [test_traj, test_traj]
                        if networkmod in discrete_lists:
                            env = DummyVecEnv([lambda: GPSPosition_discrete_lospos(test_trajdata_range, traj_type, action_scale, discrete_actionspace,
                                                                               reward_setting,trajdata_sort,baseline_mod)])
                        elif networkmod in continuous_lists:
                            if envmod == 'lospos_convQR':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic, conv_corr)])
                            elif envmod == 'losposconvR_onlyR':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyR(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic, conv_corr)])
                            elif envmod == 'losconvR_onlyR':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_los_convQR_onlyR(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic, conv_corr)])
                            elif envmod == 'losposconvQ_onlyQ':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQ(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic, conv_corr)])

                            elif envmod == 'losposcovR_onlyRNallcorrect':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyRNallcorrect(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic,
                                                                                                                 conv_corr, allcorrect=allcorrect)])
                            elif envmod == 'losposcovQ_onlyQNallcorrect':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQNallcorrect(test_trajdata_range, traj_type, testtype, continuous_Vaction_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic,
                                                                                                                 conv_corr, allcorrect=allcorrect)])
                            elif envmod == 'losposcovQR_QRNallcorrect':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_QRNallcorrect(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic,
                                                                                                                 conv_corr, allcorrect=allcorrect)])
                            elif envmod == 'losposcovR_RNobsendcorrect':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_RNobsendcorrect(test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                                                                         continuous_actionspace,reward_setting, trajdata_sort, baseline_mod, posnum, noise_scale_dic,
                                                                                                                 conv_corr, allcorrect=allcorrect)])
                            elif envmod == 'lospos':
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos(test_trajdata_range,traj_type, testtype,continuous_action_scale,continuous_actionspace,reward_setting,
                                                                                                    trajdata_sort,baseline_mod,posnum,noise_scale_dic,conv_corr)])

                        obs = env.reset()
                        maxiter = 100000
                        for iter in range(maxiter):
                            if iter == 0:  # reset state for a perid of iterations
                                action, _states = model.predict(obs, deterministic=True)
                            else:
                                action, _states = model.predict(obs, deterministic=True, state=_states)  # , state=_states
                            obs, rewards, done, info = env.step(action)
                            tmp = info[0]['tripIDnum']
                            if iter <= 1 or iter % np.ceil(maxiter / 10) == 0:
                                # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                                print(f'Iter {iter}, traj {tmp} reward is {rewards}')
                            elif done:
                                print(f'Iter {iter}, traj {tmp} reward is {rewards}, done')
                                break

                    logdirname=model.logger.dir + f'/testmore_{testtype}_'
                    recording_results_ecef_RL4KF(dir_path, data_truth_dic,[test_trajlist[0],test_trajlist[-1]],tripIDlist_test,logdirname,baseline_mod,traj_record=False)

            cnt=1
    print('More Test for different phonetype finished.')

elif onlytesting:
    for model_folder in model_folderlist:
        if f'pos={posnum_test}' not in model_folder:
            continue
        if Twoagent_testing:
            break
        #record model
        # if networkmod in model_folder:
        model_sepfolderlist=os.listdir(f'{model_basefolder}/{model_folder}') # PPO_1
        model_sepfolderlist.sort()
        # model_sepfolderlist=['RecurrentPPO_2']
        if f'lr=' in model_folder and f'{posnum_test}' in model_folder:
            # network_unit_test = int(model_folder.split(f'_unit=')[1].split('_layer')[0])
            # laynum_test = int(model_folder.split('_layer=')[1].split('_QS')[0])
            if envmod == 'losposcovR_onlyRNallcorrect':
                continuous_action_scale = float(model_folder.split('XAS=')[1].split('_RMSEadv_re')[0])
                conv_corr = model_folder.split('_RMSEadv_re_')[1]
                noise_scale_dic = {'process':1,'measurement':float(model_folder.split('_RS=')[1].split('_XAS')[0])}
            elif envmod == 'poscovQ_onlyQNallcorrect':
                continuous_Vaction_scale = float(model_folder.split('VAS=')[1].split('_RMSEadv_re')[0])
                conv_corr = model_folder.split('_RMSEadv_re_')[1]
                noise_scale_dic = {'process': float(model_folder.split('_QS=')[1].split('_VAS')[0]), 'measurement': 1}
        else:
            continue

        for model_sepfolder in model_sepfolderlist:
            process_trig = False
            if ('csv' not in model_sepfolder) and ('txt' not in model_sepfolder):
                model_filelist=os.listdir(f'{model_basefolder}/{model_folder}/{model_sepfolder}')
                model_filelist.sort()
                for model_file in model_filelist:
                    if networkmod in model_file:
                        model_filename=model_file
                        process_trig = True
                        break
                    else:
                        process_trig = False

            if process_trig:
                model_loggerdir=f'{model_basefolder}/{model_folder}/{model_sepfolder}'
                t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f'{model_loggerdir}, {t}')
                model_filenamepath=f'{model_loggerdir}/{model_filename}'
                if networkmod=='discrete_A2C':
                    model = A2C.load(model_filenamepath)
                elif networkmod in {'continuous_lstmATF1'}:
                    if envmod == 'losposcovR_onlyRNallcorrect':
                        env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyRNallcorrect(trajdata_range,traj_type_target_train,triptype,continuous_action_scale,
                                                                continuous_actionspace,reward_setting,trajdata_sort,baseline_mod, posnum_test,noise_scale_dic,conv_corr,allcorrect=allcorrect)])
                        encoder = CustomATF1_AKFRL_losposcovRinnovation # innovation
                        obs = env.reset()
                        features_dim_gnss = obs['gnss'].shape[-1]
                        features_dim_pos = obs['pos'].shape[-1]
                        features_dim_R = obs['R_noise'].shape[-1]
                        features_dim_ino = obs['innovation'].shape[-1]
                        policy_dim = features_dim_gnss + features_dim_pos + features_dim_R + features_dim_ino
                        net_arch = [network_unit, network_unit]
                        policy_kwargs = dict(features_extractor_class=encoder,
                                             features_extractor_kwargs=dict(features_dim=policy_dim),
                                             ATF_trig=networkmod, net_arch=net_arch)
                        model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs)
                        model.policy.features_extractor.attention1.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention1.attwts.weight.squeeze())
                        model.policy.features_extractor.attention2.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention2.attwts.weight.squeeze())
                        model.policy.features_extractor.attention3.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention3.attwts.weight.squeeze())
                        model.policy.features_extractor.attention4.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention4.attwts.weight.squeeze())

                    elif envmod == 'poscovQ_onlyQNallcorrect' or envmod == 'losposcovQ_onlyQNallcorrect':
                        env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQNallcorrect(trajdata_range,traj_type_target_train,triptype,
                              continuous_Vaction_scale,continuous_actionspace,reward_setting,trajdata_sort,baseline_mod,posnum_test,noise_scale_dic,conv_corr,allcorrect=allcorrect)])

                        obs = env.reset()
                        features_dim_gnss = obs['gnss'].shape[-1]
                        features_dim_pos = obs['pos'].shape[-1]
                        features_dim_Q = obs['Q_noise'].shape[-1]
                        features_dim_ino = obs['innovation'].shape[-1]
                        encoder = CustomATF1_AKFRL_poscovQinnovation
                        policy_dim = features_dim_pos + features_dim_Q + features_dim_ino
                        if envmod == 'losposcovQ_onlyQNallcorrect':
                            encoder = CustomATF1_AKFRL_losposcovQ
                            policy_dim = features_dim_pos + features_dim_Q + features_dim_gnss
                        net_arch = [network_unit, network_unit]
                        policy_kwargs = dict(features_extractor_class=encoder,features_extractor_kwargs=dict(features_dim=policy_dim),ATF_trig=networkmod, net_arch=net_arch)

                        model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs)
                        model.policy.features_extractor.attention1.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention1.attwts.weight.squeeze())
                        model.policy.features_extractor.attention2.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention2.attwts.weight.squeeze())
                        model.policy.features_extractor.attention3.attwts.weight = torch.nn.Parameter(model.policy.features_extractor.attention3.attwts.weight.squeeze())

                    model.load(model_filenamepath,env=env)

                # more tests
                if moretests:
                    for testtype in moreteststypelist:
                        print(f'more test for {testtype} env begin here')
                        if testtype == 'highway':
                            tripIDlist_test = traj_highway
                        elif testtype == 'urban':
                            tripIDlist_test = traj_urban
                        elif testtype == 'xiaomiurban':
                            tripIDlist_test = traj_xiaomiurban
                        elif testtype == 'pixelurban':
                            tripIDlist_test = traj_pixelurban
                        elif testtype == 'pixel567urban':
                            tripIDlist_test = traj_pixel567urban
                        elif testtype == 'smurban':
                            tripIDlist_test = traj_smurban

                        if alltraj:
                            more_test_trajrange = [0, len(tripIDlist) - 1]
                        else:
                            more_test_trajrange = [int(np.ceil(len(tripIDlist_test) * ratio)) + 1,len(tripIDlist_test) - 1]

                        if testtype == triptype:
                            traj_type = traj_type_target_test  # 独立同分布测试
                        else:
                            traj_type = [0, 1]  # 域外分布测试范围

                        test_trajlist = range(more_test_trajrange[0], more_test_trajrange[-1] + 1)  # [0,1,2,3,4,5]
                        for test_traj in test_trajlist:
                            test_trajdata_range = [test_traj, test_traj]
                            if networkmod in discrete_lists:
                                env = DummyVecEnv([lambda: GPSPosition_discrete_lospos(test_trajdata_range, traj_type,action_scale,discrete_actionspace,
                                                                                       reward_setting, trajdata_sort, baseline_mod)])
                            elif networkmod in continuous_lists:
                                if envmod == 'losposcovR_onlyRNallcorrect':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyRNallcorrect(
                                        test_trajdata_range, traj_type, testtype, continuous_action_scale,
                                        continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, posnum_test,
                                        noise_scale_dic, conv_corr, interrupt_dic=interrupt_dic)])
                                elif envmod == 'poscovQ_onlyQNallcorrect':
                                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQNallcorrect(
                                        test_trajdata_range, traj_type, testtype, continuous_Vaction_scale,
                                        continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, posnum_test,
                                        noise_scale_dic,conv_corr, interrupt_dic=interrupt_dic)])

                            obs = env.reset()
                            maxiter = 100000
                            for iter in range(maxiter):
                                if iter  == 0:  # reset state for a perid of iterations % 10
                                    action, _states = model.predict(obs, deterministic=True)
                                else:
                                    action, _states = model.predict(obs, deterministic=True, state=_states) # , state=_states
                                obs, rewards, done, info = env.step(action)
                                tmp = info[0]['tripIDnum']
                                if iter <= 1 or iter % 100 == 0:
                                    # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}')
                                elif rewards == 0:
                                    # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}')
                                elif done:
                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}, done')
                                    break

                        logdirname = model_loggerdir + f'/testmore_{testtype}_'
                        recording_results_ecef_RL4KF(dir_path, data_truth_dic, [test_trajlist[0], test_trajlist[-1]],tripIDlist_test, logdirname, baseline_mod, traj_record=True)

    print('only test finish!')

if Twoagent_testing:
    R_model_sepfolderlist=os.listdir(f'{R_model_basefolder}/{R_model_folder}') # PPO_1
    R_model_sepfolderlist.sort()
    Q_model_sepfolderlist=os.listdir(f'{Q_model_basefolder}/{Q_model_folder}') # PPO_1
    Q_model_sepfolderlist.sort()
    # model_sepfolderlist=['RecurrentPPO_2']
    if f'lr=' in R_model_folder and f'{posnum_test}' in R_model_folder:
            continuous_Xaction_scale = float(R_model_folder.split('XAS=')[1].split('_RMSEadv_re')[0])
            continuous_Vaction_scale = float(Q_model_folder.split('VAS=')[1].split('_RMSEadv_re')[0])
            QS = float(Q_model_folder.split('_QS=')[1].split('_VAS')[0])
            RS = float(R_model_folder.split('_RS=')[1].split('_XAS')[0])
            noise_scale_dic = {'process': QS,'measurement': RS}
    else:
        print('R_model_folder error')

    QR_model_logger = f'{QR_model_basefolder}/lr={learning_rate_list[0]}_pos={posnum_test}_QS={QS}_RS={RS}_XAS={continuous_Xaction_scale}_VAS={continuous_Vaction_scale}_twoagent'
    if not os.path.exists(QR_model_logger):
        os.makedirs(QR_model_logger)

    for R_model_sepfolder in R_model_sepfolderlist:
        for Q_model_sepfolder in Q_model_sepfolderlist:
            R_process_trig = False
            Q_process_trig = False
            if ('csv' not in R_model_sepfolder) and ('txt' not in R_model_sepfolder):
                R_model_filelist=os.listdir(f'{R_model_basefolder}/{R_model_folder}/{R_model_sepfolder}')
                R_model_filelist.sort()
                for R_model_file in R_model_filelist:
                    if networkmod in R_model_file:
                        R_model_filename=R_model_file
                        R_process_trig = True
                        break
                    else:
                        R_process_trig = False

            if ('csv' not in Q_model_sepfolder) and ('txt' not in Q_model_sepfolder):
                Q_model_filelist=os.listdir(f'{Q_model_basefolder}/{Q_model_folder}/{Q_model_sepfolder}')
                Q_model_filelist.sort()
                for Q_model_file in Q_model_filelist:
                    if networkmod in Q_model_file:
                        Q_model_filename = Q_model_file
                        Q_process_trig = True
                        break
                    else:
                        Q_process_trig = False

            if R_process_trig and Q_process_trig:
                R_model_loggerdir = f'{R_model_basefolder}/{R_model_folder}/{R_model_sepfolder}'
                t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f'{R_model_loggerdir}, {t}')
                R_model_filenamepath=f'{R_model_loggerdir}/{R_model_filename}'

                Q_model_loggerdir = f'{Q_model_basefolder}/{Q_model_folder}/{Q_model_sepfolder}'
                t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f'{Q_model_loggerdir}, {t}')
                Q_model_filenamepath=f'{Q_model_loggerdir}/{Q_model_filename}'

                if networkmod in {'continuous_lstmATF1'}:
                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyRNallcorrect(trajdata_range,traj_type_target_train,triptype,continuous_action_scale, continuous_actionspace,
                                       reward_setting,trajdata_sort, baseline_mod,posnum_test,noise_scale_dic,conv_corr,allcorrect=allcorrect)])
                    obs = env.reset()
                    features_dim_gnss = obs['gnss'].shape[-1]
                    features_dim_pos = obs['pos'].shape[-1]
                    features_dim_R = obs['R_noise'].shape[-1]
                    features_dim_ino = obs['innovation'].shape[-1]

                    encoder = CustomATF1_AKFRL_losposcovRinnovation  # innovation
                    policy_dim = features_dim_gnss + features_dim_pos + features_dim_R + features_dim_ino
                    net_arch = [network_unit, network_unit]
                    policy_kwargs = dict(features_extractor_class=encoder,features_extractor_kwargs=dict(features_dim=policy_dim),ATF_trig=networkmod, net_arch=net_arch)
                    R_model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs)
                    R_model.policy.features_extractor.attention1.attwts.weight = torch.nn.Parameter(R_model.policy.features_extractor.attention1.attwts.weight.squeeze())
                    R_model.policy.features_extractor.attention2.attwts.weight = torch.nn.Parameter(R_model.policy.features_extractor.attention2.attwts.weight.squeeze())
                    R_model.policy.features_extractor.attention3.attwts.weight = torch.nn.Parameter(R_model.policy.features_extractor.attention3.attwts.weight.squeeze())
                    R_model.policy.features_extractor.attention4.attwts.weight = torch.nn.Parameter(R_model.policy.features_extractor.attention4.attwts.weight.squeeze())
                    R_model.load(R_model_filenamepath, env=env)

                    env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_onlyQNallcorrect(trajdata_range,traj_type_target_train,triptype,continuous_Vaction_scale,continuous_actionspace,
                                            reward_setting,trajdata_sort,baseline_mod,posnum_test,noise_scale_dic,conv_corr,allcorrect=allcorrect)])
                    obs = env.reset()
                    features_dim_gnss = obs['gnss'].shape[-1]
                    features_dim_pos = obs['pos'].shape[-1]
                    features_dim_Q = obs['Q_noise'].shape[-1]
                    features_dim_ino = obs['innovation'].shape[-1]
                    encoder = CustomATF1_AKFRL_poscovQinnovation
                    policy_dim = features_dim_pos + features_dim_Q + features_dim_ino
                    net_arch = [network_unit, network_unit]
                    policy_kwargs = dict(features_extractor_class=encoder,features_extractor_kwargs=dict(features_dim=policy_dim),ATF_trig=networkmod, net_arch=net_arch)
                    Q_model = RecurrentPPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs)
                    Q_model.policy.features_extractor.attention1.attwts.weight = torch.nn.Parameter(Q_model.policy.features_extractor.attention1.attwts.weight.squeeze())
                    Q_model.policy.features_extractor.attention2.attwts.weight = torch.nn.Parameter(Q_model.policy.features_extractor.attention2.attwts.weight.squeeze())
                    Q_model.policy.features_extractor.attention3.attwts.weight = torch.nn.Parameter(Q_model.policy.features_extractor.attention3.attwts.weight.squeeze())
                    Q_model.load(Q_model_filenamepath,env=env)

                # more tests
                if moretests:
                    for testtype in moreteststypelist:
                        print(f'more test for {testtype} env begin here')
                        if testtype == 'highway':
                            tripIDlist_test = traj_highway
                        elif testtype == 'urban':
                            tripIDlist_test = traj_urban
                        elif testtype == 'xiaomiurban':
                            tripIDlist_test = traj_xiaomiurban
                        elif testtype == 'pixelurban':
                            tripIDlist_test = traj_pixelurban
                        elif testtype == 'pixel567urban':
                            tripIDlist_test = traj_pixel567urban
                        elif testtype == 'smurban':
                            tripIDlist_test = traj_smurban

                        if alltraj:
                            more_test_trajrange = [0, len(tripIDlist) - 1]
                        else:
                            more_test_trajrange = [int(np.ceil(len(tripIDlist_test) * ratio)) + 1,len(tripIDlist_test) - 1]

                        if testtype == triptype:
                            traj_type = traj_type_target_test  # 独立同分布测试
                        else:
                            traj_type = [0, 1]  # 域外分布测试范围

                        test_trajlist = range(more_test_trajrange[0], more_test_trajrange[-1] + 1)  # [0,1,2,3,4,5]
                        for test_traj in test_trajlist:
                            test_trajdata_range = [test_traj, test_traj]
                            if networkmod in continuous_lists:
                                env = DummyVecEnv([lambda: GPSPosition_continuous_lospos_convQR_QRNallcorrect(test_trajdata_range, traj_type, testtype, continuous_Xaction_scale,
                                    continuous_Vaction_scale,continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, posnum_test,
                                    noise_scale_dic, conv_corr, interrupt_dic=interrupt_dic, allcorrect=allcorrect)])

                            obs = env.reset()
                            maxiter = 100000
                            for iter in range(maxiter):
                                if iter  == 0:  # reset state for a perid of iterations % 10
                                    R_action, _Rstates = R_model.predict(obs, deterministic=True)
                                    Q_action, _Qstates = Q_model.predict(obs, deterministic=True)
                                else:
                                    R_action, _Rstates = R_model.predict(obs, deterministic=True, state=_Rstates)
                                    Q_action, _Qstates = Q_model.predict(obs, deterministic=True, state=_Qstates)
                                action = np.concatenate((Q_action,R_action), axis=1)
                                obs, rewards, done, info = env.step(action)
                                tmp = info[0]['tripIDnum']
                                if iter <= 1 or iter % 100 == 0:
                                    # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}')
                                elif rewards == 0:
                                    # print(f'Iter {:.1f} reward is {:.2e}'.format(iter, rewards))
                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}')
                                elif done:
                                    print(f'Iter {iter}, traj {tmp} reward is {rewards}, done')
                                    break

                        model_loggerdir = QR_model_logger + f'/R_{R_model_sepfolder}_Q_{Q_model_sepfolder}'
                        if not os.path.exists(model_loggerdir):
                            os.makedirs(model_loggerdir)
                        logdirname = model_loggerdir + f'/testmore_{testtype}_'
                        if interrupt_dic is not None:
                            inter_time = interrupt_dic['time']
                            logdirname = model_loggerdir + f'/testmore_{testtype}_interrupt{inter_time}_'
                        recording_results_ecef_RL4KF(dir_path, data_truth_dic, [test_trajlist[0], test_trajlist[-1]],tripIDlist_test, logdirname, baseline_mod, traj_record=True)

    print('only test finish!')
# 强化学习定位环境构建
import gym
from gym import spaces
import random
import pickle
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob as gl
from envRLKF.env_param_IMU import *
from scipy.spatial import distance
import math
import pymap3d as pm
import torch
pid = os.getpid()
print("当前程序的 PID:", pid)
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import lightgbm as lgb
# from sklearn.metrics import mean_absolute_error
import simdkalman

step_print = False
mutipath = True
# 导入数据
dir_path = '/mnt/sdb/home/tangjh/GSDC2023/'  # '/home/tangjh/smartphone-decimeter-2022/''D:/jianhao/smartphone-decimeter-2022/'
# load raw data
with open(dir_path + 'envRLKF/raw_baseline_IMU_multic_enu.pkl', "rb") as file:
    data_truth_dic = pickle.load(file)
file.close()
# load raw accel data
with open(dir_path + 'envRLKF/raw_acc_enu.pkl', "rb") as file:
    data_raw_acc_enu_dic = pickle.load(file)
file.close()
with open(dir_path + 'envRLKF/raw_rotation_matrix_dic.pkl', "rb") as file:
    data_RM_HF_dic = pickle.load(file)
file.close()

gnss_trig = False
if gnss_trig:
    with open(dir_path + 'env/raw_gnss_gpsl1.pkl', "rb") as file:
        gnss_dic = pickle.load(file)
    file.close()

with open(dir_path + 'envRLKF/processed_features_multicCN0AA_lla_cid.pkl', "rb") as file:
    losfeature_all = pickle.load(file)
file.close()

"""
processed_features_multic_Allfeature_mutipath.pkl:
0:svid; 1:residuals; 2-4:los vector; 5:CN0 6: RawPseudorangeUncertaintyMeters; 7: AA; 8: EA; 9: mutipath_indicator
"""
random.seed(0)

traj_sum_all_df = pd.read_csv(f'{dir_path}/env/traj_summary.csv')
traj_sum_IMU_df = pd.read_csv(f'{dir_path}envRLKF/raw_tripID_IMU_enu.csv')
traj_sum_df = pd.merge(traj_sum_all_df, traj_sum_IMU_df, on='tripId')

# divise with env type
traj_urban = traj_sum_df.loc[traj_sum_df['Type'] == 'urban']['tripId'].values.tolist()
traj_highway = traj_sum_df.loc[traj_sum_df['Type'] == 'highway']['tripId'].values.tolist()
traj_semiurban = traj_sum_df.loc[traj_sum_df['Type'] == 'semiurban']['tripId'].values.tolist()

# divisi with phone type
traj_xiaomi = traj_sum_df[traj_sum_df['phone'].str.contains('mi', na=False)]['tripId'].values.tolist()
traj_xiaomiurban = traj_sum_df[(traj_sum_df['phone'].str.contains('mi', na=False)) & (traj_sum_df['Type'] == 'urban')]['tripId'].values.tolist()
traj_pixelurban = traj_sum_df[(traj_sum_df['phone'].str.contains('pixel', na=False)) & (traj_sum_df['Type'] == 'urban')]['tripId'].values.tolist()
traj_pixel5urban = traj_sum_df[(traj_sum_df['phone'].str.contains('pixel5', na=False)) & (traj_sum_df['Type'] == 'urban')]['tripId'].values.tolist()
traj_pixel4urban = traj_sum_df[(traj_sum_df['phone'].str.contains('pixel4', na=False)) & (traj_sum_df['Type'] == 'urban')][
    'tripId'].values.tolist()
traj_pixel6urban = traj_sum_df[(traj_sum_df['phone'].str.contains('pixel6', na=False)) & (traj_sum_df['Type'] == 'urban')][
    'tripId'].values.tolist()
traj_pixel7urban = traj_sum_df[(traj_sum_df['phone'].str.contains('pixel7', na=False)) & (traj_sum_df['Type'] == 'urban')][
    'tripId'].values.tolist()
traj_pixel567urban = [item for item in traj_pixelurban if "pixel4" not in item]

traj_smurban = traj_sum_df[(traj_sum_df['phone'].str.contains('sm', na=False)) & (traj_sum_df['Type'] == 'urban')]['tripId'].values.tolist()

KF_colums = ['XEcefMeters_kf', 'YEcefMeters_kf', 'ZEcefMeters_kf']
KF_colums_igst = ['XEcefMeters_kf_igst', 'YEcefMeters_kf_igst', 'ZEcefMeters_kf_igst']
ground_truth_colums = ['ecefX', 'ecefY', 'ecefZ']

CN0PRUAAEA_num = 8  # 伪距残差+LOS（3D）+CN0+伪距不确定性+高度角+方位角
CN0PRUEA_num = 7  # 伪距残差+LOS（3D）+CN0+伪距不确定性+高度角
CN0EA_num = 6  # 伪距残差+LOS（3D）+CN0+高度角
sigma_v = 1
sigma_x = 6
dim_Q = 12
dim_R = 6
IM = np.eye(3)  # 3维单位阵，用于拼接
ZM = np.zeros([3, 3]) # 3维零矩阵，拼接用
record_feature = True

"""
动作6维度：wls位置修正3维度+协方差矩阵误差加噪声2维度
"""

class GPSPosition_continuous_lospos_convQR_QRNallcorrect(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, trajdata_range, traj_type, triptype, continuous_Xaction_scale, continuous_Vaction_scale,
                 continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len, noise_scale_dic,
                 conv_corr, interrupt_dic=None, allcorrect=False):
        # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(GPSPosition_continuous_lospos_convQR_QRNallcorrect, self).__init__()
        self.max_visible_sat = 13
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * CN0PRUEA_num)),
             'pos': spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float),
             'Q_noise': spaces.Box(low=0, high=1, shape=(1, dim_Q), dtype=np.float),
             'R_noise': spaces.Box(low=0, high=1, shape=(1, dim_R), dtype=np.float),
             'innovation': spaces.Box(low=0, high=1, shape=(1, dim_R), dtype=np.float)})

        self.allcorrect = allcorrect  # correct in the end or in the obs
        if triptype == 'highway':
            self.tripIDlist = traj_highway
        elif triptype == 'urban':
            self.tripIDlist = traj_urban
        elif triptype == 'xiaomi':
            self.tripIDlist = traj_xiaomi
        elif triptype == 'xiaomiurban':
            self.tripIDlist = traj_xiaomiurban
        elif triptype == 'pixel567urban':
            self.tripIDlist = traj_pixel567urban
        elif triptype == 'smurban':
            self.tripIDlist = traj_smurban

        self.triptype = triptype
        self.traj_type = traj_type
        # continuous action
        if trajdata_range == 'full':
            self.trajdata_range = [0, len(self.tripIDlist) - 1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.process_noise_scale = noise_scale_dic['process']
        self.measurement_noise_scale = noise_scale_dic['measurement']
        self.continuous_action_scale = continuous_Xaction_scale
        self.continuous_Vaction_scale = continuous_Vaction_scale
        # 动作维度，6维度位置速度+6维度对应的R调整
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 6 + 6 + 12 + 12),dtype=np.float)
        self.total_reward = 0
        self.reward_setting = reward_setting
        self.trajdata_sort = trajdata_sort
        self.baseline_mod = baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
        elif self.trajdata_sort == 'randint':
            sublist = self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]]
            random.shuffle(sublist)
            self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]] = sublist
            self.tripIDnum = self.trajdata_range[0]

        self.conv_corr = conv_corr
        if interrupt_dic is not None:
            self.cnt_interrupt = 0
            self.interrupt_test = True
            self.start_ratio = interrupt_dic['start_ratio']
            self.interrupt_time = interrupt_dic['time']
        else:
            self.interrupt_test = False
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort == 'randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum = random.randint(self.trajdata_range[0], self.trajdata_range[1])
        elif self.trajdata_sort == 'sorted':
            self.tripIDnum = self.tripIDnum + 1
            if self.tripIDnum > self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        self.baseline = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature_all[self.tripIDlist[self.tripIDnum]].copy()
        self.raw_acc_enu = data_raw_acc_enu_dic[self.tripIDlist[self.tripIDnum]].values
        self.RM_HF_dic = data_RM_HF_dic[self.tripIDlist[self.tripIDnum]]

        self.datatime = self.baseline['UnixTimeMillis']
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UnixTimeMillis'].values) - 1, 'UnixTimeMillis']
        # normalize baseline
        # self.baseline['LatitudeDegrees_norm'] = (self.baseline['LatitudeDegrees']-lat_min)/(lat_max-lat_min)
        # self.baseline['LongitudeDegrees_norm'] = (self.baseline['LongitudeDegrees']-lon_min)/(lon_max-lon_min)
        # gen pred
        if self.baseline_mod == 'bl':
            self.baseline['E_RLpredict'] = self.baseline['E_bl']
            self.baseline['N_RLpredict'] = self.baseline['N_bl']
            self.baseline['U_RLpredict'] = self.baseline['U_bl']
        elif self.baseline_mod == 'wls':
            self.baseline['E_RLpredict'] = self.baseline['E_wls']
            self.baseline['N_RLpredict'] = self.baseline['N_wls']
            self.baseline['U_RLpredict'] = self.baseline['U_wls']
        elif self.baseline_mod == 'kf':
            self.baseline['E_RLpredict'] = self.baseline['E_kf']
            self.baseline['N_RLpredict'] = self.baseline['N_kf']
            self.baseline['U_RLpredict'] = self.baseline['U_kf']

        if gnss_trig:
            self.gnss = gnss_dic[self.tripIDlist[self.tripIDnum]]
        # Set the current step to a random point within the data frame
        # revise 1017: need the specific percent of the traj
        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        # set interrupt time
        if self.interrupt_test:
            self.interrupt_start_step = np.ceil(len(self.baseline) * (self.traj_type[0] + self.start_ratio * (self.traj_type[-1]-self.traj_type[0])))
            self.interrupt_start_utc = self.datatime[self.interrupt_start_step + (self.pos_num - 1)]

        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['E_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['N_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['U_RLpredict']] = None

        # 协方差矩阵初始化
        self.covv = sigma_v ** 1.250 * np.eye(dim_Q)
        self.covx = sigma_x ** 1.250 * np.eye(dim_R)
        self.covx[2, 2] = self.covx[2, 2] * 2

        # initial kf state
        self.state_llh = self.baseline.loc[0,['LatitudeDegrees','LongitudeDegrees','AltitudeMeters']].values # 初始点经纬度，转坐标系用
        self.P = 5.0 ** 2 * np.eye(dim_Q)  # initial State covariance
        self.I = np.eye(dim_Q)
        self.H = np.hstack((np.eye(dim_R),np.zeros((dim_R,dim_Q-dim_R))))
        self.dx = np.zeros([dim_Q, 1]) # initial error state

        self.p_imu = self.baseline.loc[self.current_step+(self.pos_num - 2),['E_RLpredict','N_RLpredict','U_RLpredict']].values.reshape((3, 1))
        self.v_imu = self.baseline.loc[self.current_step+(self.pos_num - 2),['VE_wls','VN_wls','VU_wls']].values.reshape((3, 1))

        self.raw_acc_utc = self.raw_acc_enu[:,0] # 加速度的时间戳矩阵
        utc_init = self.datatime[self.current_step + (self.pos_num - 2)]
        self.idx_acc = np.argmin(np.abs(self.raw_acc_utc - utc_init)) # 初始 加速度索引

        self.innovation = np.ones([1,dim_R])
        obs = self._next_observation()
        # must return in observation scale
        return obs  # self.tripIDnum#, obs#, {}

    def _normalize_pos(self, state):
        state[0] = state[0] / 10000
        state[1] = state[1] / 10000
        state[2] = state[2] / 100
        return state

    def _normalize_noise(self, obs_noise_R, obs_noise_Q):
        obs_noise_R = obs_noise_R / 100
        obs_noise_Q = obs_noise_Q / 10
        return obs_noise_R, obs_noise_Q

    def _normalize_los(self, gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        ## max normalize
        gnss[:, 1] = (gnss[:, 1]) / max(res_max, np.abs(res_min))
        gnss[:, 2] = (gnss[:, 2]) / max(losx_max, np.abs(losx_min))
        gnss[:, 3] = (gnss[:, 3]) / max(losy_max, np.abs(losy_min))
        gnss[:, 4] = (gnss[:, 4]) / max(losz_max, np.abs(losz_min))
        gnss[:, 5] = (gnss[:, 5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:, 6] = (gnss[:, 6]) / max(PRU_max, np.abs(PRU_min))
        gnss[:, 7] = (gnss[:, 7]) / max(AA_max, np.abs(AA_min))
        gnss[:, 8] = (gnss[:, 8]) / max(EA_max, np.abs(EA_min))
        # zero-score normalize
        # for i in range(gnss.shape[1]): # zero-score normalize
        #     if (i==0) or (i==gnss.shape[1]-1):
        #         continue
        #     mean,std = np.mean(gnss[:,i]),np.std(gnss[:,i])
        #     gnss[:,i] = (gnss[:,i]-mean) / std
        return gnss

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'E_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'N_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'U_RLpredict'].values])

        if self.baseline_mod == 'bl':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_bl']]],axis=1)
        elif self.baseline_mod == 'wls':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_wls']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_wls']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_wls']]],axis=1)
        elif self.baseline_mod == 'kf':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_kf']]],axis=1)

        obs = self._normalize_pos(obs)

        # gnss feature process
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features'].copy()
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        obs_feature = np.zeros([(self.max_visible_sat), CN0PRUEA_num])
        feature_index_list = [1, 2, 3, 4, 5, 6, 8]
        GPSL1_index = feature_tmp[:, 0] < 100
        feature_tmp = feature_tmp[:,feature_index_list]  # only GPSL1
        feature_tmp = feature_tmp[GPSL1_index,:]  # only GPSL1
        obs_feature[0:len(feature_tmp), :] = feature_tmp

        # noise cov feature process
        obs_Xnoise_pre = self.covx
        obs_Vnoise_pre = self.covv
        obs_Xnoise_cur = sigma_x ** 1.250 * np.eye(dim_R)
        obs_Xnoise_cur[2,2] = obs_Xnoise_cur[2,2] * 2
        obs_Vnoise_cur = sigma_v ** 1.250 * np.eye(dim_Q)
        if self.conv_corr == 'conv_corr_2': # 使用之前的不断迭代修改
            obs_noise_Q = np.diag(obs_Vnoise_pre)
            obs_noise_R = np.diag(obs_Xnoise_pre)
        elif self.conv_corr == 'conv_corr_1': # 每次都是固定的修改
            obs_noise_Q = np.diag(obs_Vnoise_cur)
            obs_noise_R = np.diag(obs_Xnoise_cur)

        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'),
                   'gnss': obs_feature.reshape(1, CN0PRUEA_num * self.max_visible_sat, order='C'),
                   'Q_noise': obs_noise_Q.reshape(1, dim_Q, order='C'), 'R_noise': obs_noise_R.reshape(1, dim_R, order='C'),
                   'innovation': self.innovation}

        return obs_all

    def step(self, action):  # modified in 3.3
        # judge if end #
        done = (self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) * self.traj_type[-1] - (
            self.pos_num) - outlayer_in_end_ecef)
        timestep = self.baseline.loc[self.current_step + (self.pos_num - 1), 'UnixTimeMillis']
        # action for new prediction

        action = np.reshape(action, [1, dim_Q * 2 + dim_R * 2])
        # action for process noise, 位置、速度、姿态、加速度
        predict_pe = action[0, 0] * self.continuous_Vaction_scale
        predict_pn = action[0, 1] * self.continuous_Vaction_scale
        predict_pu = action[0, 2] * self.continuous_Vaction_scale * 2
        predict_ve = action[0, 3] * self.continuous_Vaction_scale * 1e-1
        predict_vn = action[0, 4] * self.continuous_Vaction_scale * 1e-1
        predict_vu = action[0, 5] * self.continuous_Vaction_scale * 1e-2
        predict_pitch = action[0, 6] * self.continuous_Vaction_scale
        predict_roll = action[0, 7] * self.continuous_Vaction_scale
        predict_yaw = action[0, 8] * self.continuous_Vaction_scale
        predict_ae = action[0, 9] * self.continuous_Vaction_scale * 1e-2
        predict_an = action[0, 10] * self.continuous_Vaction_scale * 1e-2
        predict_au = action[0, 11] * self.continuous_Vaction_scale * 1e-2
        processnoise = np.array([predict_pe,predict_pn,predict_pu,predict_ve,predict_vn,predict_vu,
                                 predict_pitch,predict_roll,predict_yaw,predict_ae,predict_an,predict_au]).reshape(12,1)
        # 过程协方差调整
        diag = [action[0, 12],action[0, 13],action[0, 14],action[0, 15],action[0, 16],action[0, 17],
                action[0, 18],action[0, 19],action[0, 20],action[0, 21],action[0, 22],action[0, 23]]
        predict_Q_noise = self.process_noise_scale * np.diagflat(diag)

        # action for measurement noise, gnss 位置、速度
        predict_pe = action[0, 24] * self.continuous_action_scale
        predict_pn = action[0, 25] * self.continuous_action_scale
        predict_pu = action[0, 26] * self.continuous_action_scale * 2
        predict_ve = action[0, 27] * self.continuous_action_scale * 1e-1
        predict_vn = action[0, 28] * self.continuous_action_scale * 1e-1
        predict_vu = action[0, 29] * self.continuous_action_scale * 1e-2
        # 协方差调整
        diag = [action[0, 30], action[0, 31], action[0, 32], action[0, 33], action[0, 34], action[0, 35]]
        predict_R_noise = self.measurement_noise_scale * np.diagflat(diag)

        if self.baseline_mod == 'bl':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_bl']
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_bl']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_bl']
        elif self.baseline_mod == 'wls':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_wls']  # WLS结果作为观测
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_wls']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_wls']
        elif self.baseline_mod == 'kf':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_kf']
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_kf']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_kf']

        eskf_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_eskf_hfv2']
        eskf_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_eskf_hfv2']
        eskf_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_eskf_hfv2']

        v_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VE_wls']
        v_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VN_wls']
        v_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VU_wls']
        gro_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_gt']
        gro_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_gt']
        gro_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_gt']

        # modified for RLKF
        # modified for RLKF
        p_wls = np.array([obs_e + predict_pe, obs_n + predict_pn, obs_u + predict_pu])
        v_wls = np.array([v_e + predict_ve, v_n + predict_vn, v_u + predict_vu])
        rl_e, rl_n, rl_u = self.RL4ESKF_IMU(p_wls, v_wls, predict_R_noise, predict_Q_noise, processnoise)
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['E_RLpredict']] = rl_e
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['N_RLpredict']] = rl_n
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['U_RLpredict']] = rl_u

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['E_RLpredict']] = rl_e
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['N_RLpredict']] = rl_n
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['U_RLpredict']] = rl_u

        # reward function
        if self.reward_setting == 'RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2)) * 1e0  # *1e5
        # TODO 水平和高程是否需要加比例因子
        elif self.reward_setting == 'RMSEadv':
            reward = np.sqrt(((obs_e - gro_e) ** 2 + (obs_n - gro_n) ** 2 + (obs_u - gro_u) ** 2)) - \
                     np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))
        elif self.reward_setting == 'RMSEadv_eskf':
            reward = np.sqrt(((eskf_e - gro_e) ** 2 + (eskf_n - gro_n) ** 2 + (eskf_u - gro_u) ** 2)) - \
                     np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))
        elif self.reward_setting == 'RMSEadv_re': # 高程误差优势除2
            reward = np.sqrt(((eskf_e - gro_e) ** 2 + (eskf_n - gro_n) ** 2)) - np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2)) \
                      + (np.sqrt((eskf_u - gro_u) ** 2) - np.sqrt((rl_u - gro_u) ** 2)) * 5e-1

        error = np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))
        if step_print:
            print(
                f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                f'RL dist: [{np.abs(rl_e - gro_e):.2f}, {np.abs(rl_n - gro_n):.2f}, {np.abs(rl_u - gro_u):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,
                                   'baseline': self.baseline, 'error': error,'tripid': self.tripIDnum}  # self.info#, {}# , 'data_truth_dic':data_truth_dic

    def RL4ESKF_IMU(self, zs, us, predict_R_noise, predict_Q_noise, processnoise, policy=True):  # RL for KF modified in 0303
        # Parameters
        sigma_mahalanobis = 100  # Mahalanobis distance for rejecting innovation
        # position and velocity by imu
        acc_utc = self.raw_acc_utc[self.idx_acc]
        gnss_utc = self.datatime[self.current_step + (self.pos_num - 1)]  # current gnss utc
        gnss_utc_pre = self.datatime[self.current_step + (self.pos_num - 2)]  # previous gnss utc
        while abs(acc_utc - gnss_utc) > 11:
            # 无gnss，进行imu捷联推理
            acc_enu = self.raw_acc_enu[self.idx_acc, 1:4].reshape((3, 1))
            acc_enu = acc_enu - self.dx[9:12]
            self.v_imu = self.v_imu + acc_enu * (acc_utc - self.raw_acc_utc[self.idx_acc - 1]) * 1e-3  # 时间差 # to tensor
            self.p_imu = self.p_imu + self.v_imu * (acc_utc - self.raw_acc_utc[self.idx_acc - 1]) * 1e-3
            self.idx_acc = self.idx_acc + 1
            acc_utc = self.raw_acc_utc[self.idx_acc]

        # don't process eskf if interrupt
        if self.interrupt_test:
            if self.interrupt_start_utc < gnss_utc:
                self.cnt_interrupt += 1
                if self.cnt_interrupt <= self.interrupt_time:
                    return self.p_imu[0, 0], self.p_imu[1, 0], self.p_imu[2, 0]
                elif self.cnt_interrupt == self.interrupt_time+1:
                    # 中断后，重新用gnss位置速度复位
                    print(f'interrupt end in {gnss_utc}')
                    self.p_imu = self.baseline.loc[self.current_step + (self.pos_num - 1), ['E_kf', 'N_kf','U_kf']].values.reshape((3, 1))
                    self.v_imu = self.baseline.loc[self.current_step + (self.pos_num - 1), ['VE_wls', 'VN_wls', 'VU_wls']].values.reshape((3, 1))
                    return self.p_imu[0, 0], self.p_imu[1, 0], self.p_imu[2, 0]

        # 观测信息提取
        z = zs.reshape((3, 1))  # 位置
        u = us.reshape((3, 1))  # 速度
        # TODO 可以直接把矩阵导进来，减少计算量
        # update RM RN
        ecef_wls = np.array(pm.enu2ecef(z[0], z[1], z[2], self.state_llh[0], self.state_llh[1], self.state_llh[2]))
        llh_wls = np.array(pm.ecef2geodetic(ecef_wls[0], ecef_wls[1], ecef_wls[2]))
        RM = calculate_RM(llh_wls[0])  # 计算曲率半径
        RN = calculate_RN(llh_wls[0])

        # 计算状态矩阵
        Fv = np.array([[0, acc_enu[2, 0], -acc_enu[1, 0]],
                       [-acc_enu[2, 0], 0, acc_enu[0, 0]],
                       [acc_enu[1, 0], -acc_enu[0, 0], 0]])
        lat_rad = math.radians(z[0, 0])  # 将纬度转为弧度
        tan_lat = math.tan(lat_rad)
        Fd = np.array([[0, 1 / (RM + z[2, 0]), 0],
                       [-1 / (RN + z[2, 0]), 0, 0],
                       [-tan_lat / (RN + z[2, 0]), 0, 0]])

        tol = (gnss_utc - gnss_utc_pre) * 1e-3  # 时间间隔
        F = np.block([[IM, IM * tol, ZM, ZM],
                      [ZM, IM, Fv * tol, IM * tol],
                      [ZM, Fd * tol, IM, ZM],
                      [ZM, ZM, ZM, IM]])

        if self.conv_corr == 'conv_corr_1':  # Estimated WLS position covariance
            Q = sigma_v ** 1.250 * np.eye(dim_Q)
            Q = Q + predict_Q_noise
            R = sigma_x ** 1.250 * np.eye(dim_R)
            R[2, 2] = R[2, 2] * 2
            R = R + predict_R_noise
        elif self.conv_corr == 'conv_corr_2':
            Q = self.covv + predict_Q_noise
            R = self.covx + predict_R_noise
        Q = np.where(Q < 0, 0.01, Q)
        R = np.where(R < 0, 0.01, R)
        self.covv = Q
        self.covx = R

        # ESKF predict stage
        self.dx = F @ self.dx + processnoise
        self.P = (F @ self.P) @ F.T + self.covv
        dz = np.vstack((self.p_imu - z, self.v_imu - u))  # imu和gnss计算的位置和速度误差
        d = distance.mahalanobis(dz[:, 0], self.H @ self.dx, np.linalg.inv(self.P[:6, :6]))
        # KF: Update step
        if d < sigma_mahalanobis:
            # R = torch.from_numpy(R).double().to(self.device) # to tensor
            y = dz - self.H @ self.dx
            S = (self.H @ self.P) @ self.H.T + self.covx
            K = (self.P @ self.H.T) @ np.linalg.inv(S)
            # S_inv = torch.from_numpy(np.linalg.inv(np.array(S.cpu()))).to(self.device)
            # K = (self.P @ self.H.T) @ S_inv # to tensor
            self.dx = self.dx + K @ y
            self.P = (self.I - (K @ self.H)) @ self.P
        else:
            # If observation update is not available, increase covariance
            self.P += 10 ** 2 * self.covv
            y = dz - self.H @ self.dx

        self.p_imu = self.p_imu - self.dx[0:3]
        self.v_imu = self.v_imu - self.dx[3:6]
        self.innovation = y.reshape((1, dim_R)) / 10  # normalization
        return self.p_imu[0, 0], self.p_imu[1, 0], self.p_imu[2, 0]

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

class GPSPosition_continuous_lospos_convQR_onlyRNallcorrect(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace,
                 reward_setting, trajdata_sort, baseline_mod, traj_len, noise_scale_dic,
                 conv_corr, interrupt_dic=None, allcorrect=False):
        # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(GPSPosition_continuous_lospos_convQR_onlyRNallcorrect, self).__init__()
        self.max_visible_sat = 13
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * CN0PRUEA_num)),
             'pos': spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float),
             'Q_noise': spaces.Box(low=0, high=1, shape=(1, 12), dtype=np.float),
             'R_noise': spaces.Box(low=0, high=1, shape=(1, 6), dtype=np.float),
             'innovation': spaces.Box(low=0, high=1, shape=(1, dim_R), dtype=np.float)})

        self.allcorrect = allcorrect  # correct in the end or in the obs
        if triptype == 'highway':
            self.tripIDlist = traj_highway
        elif triptype == 'urban':
            self.tripIDlist = traj_urban
        elif triptype == 'xiaomi':
            self.tripIDlist = traj_xiaomi
        elif triptype == 'xiaomiurban':
            self.tripIDlist = traj_xiaomiurban
        elif triptype == 'pixel567urban':
            self.tripIDlist = traj_pixel567urban
        elif triptype == 'smurban':
            self.tripIDlist = traj_smurban

        self.triptype = triptype
        self.traj_type = traj_type
        # continuous action
        if trajdata_range == 'full':
            self.trajdata_range = [0, len(self.tripIDlist) - 1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.process_noise_scale = noise_scale_dic['process']
        self.measurement_noise_scale = noise_scale_dic['measurement']
        self.continuous_action_scale = continuous_action_scale
        # 动作维度，6维度位置速度+6维度对应的R调整
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 6 + 6),dtype=np.float)
        self.total_reward = 0
        self.reward_setting = reward_setting
        self.trajdata_sort = trajdata_sort
        self.baseline_mod = baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
        elif self.trajdata_sort == 'randint':
            sublist = self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]]
            random.shuffle(sublist)
            self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]] = sublist
            self.tripIDnum = self.trajdata_range[0]

        self.conv_corr = conv_corr
        if interrupt_dic is not None:
            self.cnt_interrupt = 0
            self.interrupt_test = True
            self.start_ratio = interrupt_dic['start_ratio']
            self.interrupt_time = interrupt_dic['time']
        else:
            self.interrupt_test = False
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort == 'randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum = random.randint(self.trajdata_range[0], self.trajdata_range[1])
        elif self.trajdata_sort == 'sorted':
            self.tripIDnum = self.tripIDnum + 1
            if self.tripIDnum > self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        self.baseline = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature_all[self.tripIDlist[self.tripIDnum]].copy()
        self.raw_acc_enu = data_raw_acc_enu_dic[self.tripIDlist[self.tripIDnum]].values
        self.RM_HF_dic = data_RM_HF_dic[self.tripIDlist[self.tripIDnum]]

        self.datatime = self.baseline['UnixTimeMillis']
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UnixTimeMillis'].values) - 1, 'UnixTimeMillis']
        # normalize baseline
        # self.baseline['LatitudeDegrees_norm'] = (self.baseline['LatitudeDegrees']-lat_min)/(lat_max-lat_min)
        # self.baseline['LongitudeDegrees_norm'] = (self.baseline['LongitudeDegrees']-lon_min)/(lon_max-lon_min)
        # gen pred
        if self.baseline_mod == 'bl':
            self.baseline['E_RLpredict'] = self.baseline['E_bl']
            self.baseline['N_RLpredict'] = self.baseline['N_bl']
            self.baseline['U_RLpredict'] = self.baseline['U_bl']
        elif self.baseline_mod == 'wls':
            self.baseline['E_RLpredict'] = self.baseline['E_wls']
            self.baseline['N_RLpredict'] = self.baseline['N_wls']
            self.baseline['U_RLpredict'] = self.baseline['U_wls']
        elif self.baseline_mod == 'kf':
            self.baseline['E_RLpredict'] = self.baseline['E_kf']
            self.baseline['N_RLpredict'] = self.baseline['N_kf']
            self.baseline['U_RLpredict'] = self.baseline['U_kf']

        if gnss_trig:
            self.gnss = gnss_dic[self.tripIDlist[self.tripIDnum]]
        # Set the current step to a random point within the data frame
        # revise 1017: need the specific percent of the traj
        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        # setting for interrupt time
        if self.interrupt_test:
            self.interrupt_start_step = np.ceil(len(self.baseline) * (self.traj_type[0] + self.start_ratio * (self.traj_type[-1]-self.traj_type[0])))
            self.interrupt_start_utc = self.datatime[self.interrupt_start_step + (self.pos_num - 1)]

        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num -  1, ['E_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['N_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['U_RLpredict']] = None

        # 协方差矩阵初始化
        self.covv = sigma_v ** 1.250 * np.eye(dim_Q)
        self.covx = sigma_x ** 1.250 * np.eye(dim_R)
        self.covx[2, 2] = self.covx[2, 2] * 2

        # initial kf state
        self.state_llh = self.baseline.loc[0,['LatitudeDegrees','LongitudeDegrees','AltitudeMeters']].values # 初始点经纬度，转坐标系用
        self.P = 5.0 ** 2 * np.eye(dim_Q)  # initial State covariance
        self.I = np.eye(dim_Q)
        self.H = np.hstack((np.eye(dim_R),np.zeros((dim_R,dim_Q-dim_R))))
        self.dx = np.zeros([dim_Q, 1]) # initial erro state
        # trasfer to tensor
        # self.P = 5.0 ** 2 * torch.eye(dim_Q).double().to(self.device)  # initial State covariance
        # self.I = torch.eye(dim_Q).double().to(self.device)
        # self.H = torch.from_numpy(np.hstack((np.eye(6),np.zeros((6,dim_Q-6))))).to(self.device)
        # self.dx = torch.zeros([dim_Q, 1]).double().to(self.device) # initial erro state

        self.p_imu = self.baseline.loc[self.current_step+(self.pos_num - 2),['E_RLpredict','N_RLpredict','U_RLpredict']].values.reshape((3, 1))
        self.v_imu = self.baseline.loc[self.current_step+(self.pos_num - 2),['VE_wls','VN_wls','VU_wls']].values.reshape((3, 1))
        # trasfer to tensor
        # self.p_imu = torch.from_numpy(self.p_imu).double().to(self.device)
        # self.v_imu = torch.from_numpy(self.v_imu).double().to(self.device)
        self.raw_acc_utc = self.raw_acc_enu[:,0] # 加速度的时间戳矩阵
        utc_init = self.datatime[self.current_step + (self.pos_num - 2)]
        self.idx_acc = np.argmin(np.abs(self.raw_acc_utc - utc_init)) # 初始 加速度索引

        self.innovation = np.ones([1,dim_R])
        obs = self._next_observation()
        # must return in observation scale
        return obs  # self.tripIDnum#, obs#, {}

    def _normalize_pos(self, state):
        state[0] = state[0] / 10000
        state[1] = state[1] / 10000
        state[2] = state[2] / 100
        return state

    def _normalize_noise(self, obs_noise_R, obs_noise_Q):
        obs_noise_R = obs_noise_R / 100
        obs_noise_Q = obs_noise_Q / 10
        return obs_noise_R, obs_noise_Q

    def _normalize_los(self, gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        ## max normalize
        gnss[:, 1] = (gnss[:, 1]) / max(res_max, np.abs(res_min))
        gnss[:, 2] = (gnss[:, 2]) / max(losx_max, np.abs(losx_min))
        gnss[:, 3] = (gnss[:, 3]) / max(losy_max, np.abs(losy_min))
        gnss[:, 4] = (gnss[:, 4]) / max(losz_max, np.abs(losz_min))
        gnss[:, 5] = (gnss[:, 5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:, 6] = (gnss[:, 6]) / max(PRU_max, np.abs(PRU_min))
        gnss[:, 7] = (gnss[:, 7]) / max(AA_max, np.abs(AA_min))
        gnss[:, 8] = (gnss[:, 7]) / max(EA_max, np.abs(EA_min))
        # zero-score normalize
        # for i in range(gnss.shape[1]): # zero-score normalize
        #     if (i==0) or (i==gnss.shape[1]-1):
        #         continue
        #     mean,std = np.mean(gnss[:,i]),np.std(gnss[:,i])
        #     gnss[:,i] = (gnss[:,i]-mean) / std
        return gnss

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'E_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'N_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'U_RLpredict'].values])

        if self.baseline_mod == 'bl':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_bl']]],axis=1)
        elif self.baseline_mod == 'wls':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_wls']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_wls']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_wls']]],axis=1)
        elif self.baseline_mod == 'kf':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_kf']]],axis=1)

        obs = self._normalize_pos(obs)

        # gnss feature process
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features'].copy()
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        obs_feature = np.zeros([(self.max_visible_sat), CN0PRUEA_num])
        feature_index_list = [1, 2, 3, 4, 5, 6, 8]
        GPSL1_index = feature_tmp[:, 0] < 100
        feature_tmp = feature_tmp[:,feature_index_list]  # only GPSL1
        feature_tmp = feature_tmp[GPSL1_index,:]  # only GPSL1
        obs_feature[0:len(feature_tmp), :] = feature_tmp

        # noise cov feature process
        obs_Xnoise_pre = self.covx
        # obs_Vnoise_pre = self.covv
        obs_Xnoise_cur = sigma_x ** 1.250 * np.eye(dim_R)
        obs_Xnoise_cur[2,2] = obs_Xnoise_cur[2,2] * 2
        # obs_Vnoise_cur = sigma_v ** 1.250 * np.eye(dim_Q)
        if self.conv_corr == 'conv_corr_2': # 使用之前的不断迭代修改
            obs_noise_Q = np.diag(self.covv)
            obs_noise_R = np.diag(obs_Xnoise_pre)
        elif self.conv_corr == 'conv_corr_1': # 每次都是固定的修改
            obs_noise_Q = np.diag(self.covv)
            obs_noise_R = np.diag(obs_Xnoise_cur)

        # obs_feature = np.array([np.where(self.visible_sat[i] in feature_tmp[:,0],feature_tmp[feature_tmp[:,0]==self.visible_sat[i],1:]
        #                         ,np.zeros_like(feature_tmp[0,1:])) for i in range(len(self.visible_sat))])
        # obs_all={'pos':obs, 'gnss':obs_feature}
        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'),
                   'gnss': obs_feature.reshape(1, CN0PRUEA_num * self.max_visible_sat, order='C'),
                   'Q_noise': obs_noise_Q.reshape(1, 12, order='C'), 'R_noise': obs_noise_R.reshape(1, 6, order='C'),
                   'innovation': self.innovation}

        # obs = obs.reshape(-1, 1) # + (traj_len-1)  + (traj_len-1)
        # obs=np.array([self.baseline.loc[self.current_step, 'LatitudeDegrees_norm'],self.baseline.loc[self.current_step, 'LongitudeDegrees_norm']])
        # obs=obs.reshape(2,1)
        # TODO latDeg lngDeg ... latDeg lngDeg
        return obs_all

    def step(self, action):  # modified in 3.3
        # judge if end #
        done = (self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) * self.traj_type[-1] - (
            self.pos_num) - outlayer_in_end_ecef)
        timestep = self.baseline.loc[self.current_step + (self.pos_num - 1), 'UnixTimeMillis']
        # action for new prediction
        action = np.reshape(action, [1, 6 + 6])
        predict_pe = action[0, 0] * self.continuous_action_scale
        predict_pn = action[0, 1] * self.continuous_action_scale
        predict_pu = action[0, 2] * self.continuous_action_scale * 2
        predict_ve = action[0, 3] * self.continuous_action_scale * 1e-1
        predict_vn = action[0, 4] * self.continuous_action_scale * 1e-1
        predict_vu = action[0, 5] * self.continuous_action_scale * 1e-2
        # 协方差调整
        diag = [action[0, 6],action[0, 7],action[0, 8],action[0, 9],action[0, 10],action[0, 11]]
        predict_R_noise = self.measurement_noise_scale * np.diagflat(diag)

        if self.baseline_mod == 'bl':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_bl']
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_bl']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_bl']
        elif self.baseline_mod == 'wls':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_wls']  # WLS结果作为观测
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_wls']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_wls']
        elif self.baseline_mod == 'kf':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_kf']
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_kf']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_kf']

        eskf_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_eskf_hfv2']
        eskf_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_eskf_hfv2']
        eskf_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_eskf_hfv2']

        v_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VE_wls']
        v_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VN_wls']
        v_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VU_wls']
        gro_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_gt']
        gro_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_gt']
        gro_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_gt']

        # modified for RLKF
        p_wls = np.array([obs_e + predict_pe, obs_n + predict_pn, obs_u + predict_pu])
        v_wls = np.array([v_e + predict_ve, v_n + predict_vn, v_u + predict_vu])
        rl_e, rl_n, rl_u = self.RL4ESKF_IMU(p_wls, v_wls, predict_R_noise)
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['E_RLpredict']] = rl_e
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['N_RLpredict']] = rl_n
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['U_RLpredict']] = rl_u

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['E_RLpredict']] = rl_e
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['N_RLpredict']] = rl_n
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['U_RLpredict']] = rl_u
        # reward function
        if self.reward_setting == 'RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2)) * 1e0  # *1e5
        # TODO 水平和高程是否需要加比例因子
        elif self.reward_setting == 'RMSEadv':
            reward = np.sqrt(((obs_e - gro_e) ** 2 + (obs_n - gro_n) ** 2 + (obs_u - gro_u) ** 2)) - \
                     np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))
        elif self.reward_setting == 'RMSEadv_eskf':
            reward = np.sqrt(((eskf_e - gro_e) ** 2 + (eskf_n - gro_n) ** 2 + (eskf_u - gro_u) ** 2)) - \
                     np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))
        elif self.reward_setting == 'RMSEadv_re': # 高程误差优势除2
            reward = np.sqrt(((eskf_e - gro_e) ** 2 + (eskf_n - gro_n) ** 2)) - np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2)) \
                      + (np.sqrt((eskf_u - gro_u) ** 2) - np.sqrt((rl_u - gro_u) ** 2)) * 5e-1

        error = np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))

        if step_print:
            print(
                f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                f'RL dist: [{np.abs(rl_e - gro_e):.2f}, {np.abs(rl_n - gro_n):.2f}, {np.abs(rl_u - gro_u):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,'baseline': self.baseline,
                                   'error':error, 'tripid':self.tripIDnum}  # self.info#, {}# , 'data_truth_dic':data_truth_dic

    def RL4ESKF_IMU(self, zs, us, predict_R_noise):  # RL for KF modified in 0303
        # Parameters
        sigma_mahalanobis = 10000000000000000000000.0  # Mahalanobis distance for rejecting innovation

        # position and velocity by imu
        acc_utc = self.raw_acc_utc[self.idx_acc]
        gnss_utc = self.datatime[self.current_step + (self.pos_num - 1)] # current gnss utc
        gnss_utc_pre = self.datatime[self.current_step + (self.pos_num - 2)]  # previous gnss utc
        while abs(acc_utc - gnss_utc) > 11:
            acc_enu = self.raw_acc_enu[self.idx_acc,1:4].reshape((3, 1))
            # acc_enu = acc_enu - np.array(self.dx[9:12].cpu()) # to tensor
            acc_enu = acc_enu - self.dx[9:12]
            # acc_enu_tensor = torch.from_numpy(acc_enu).double().to(self.device)  # to tensor
            self.v_imu = self.v_imu + acc_enu * (acc_utc - self.raw_acc_utc[self.idx_acc - 1]) * 1e-3  # 时间差 # to tensor
            self.p_imu = self.p_imu + self.v_imu * (acc_utc - self.raw_acc_utc[self.idx_acc - 1]) * 1e-3
            self.idx_acc = self.idx_acc + 1
            acc_utc = self.raw_acc_utc[self.idx_acc]

        if self.interrupt_test:
            if self.interrupt_start_utc < gnss_utc:
                self.cnt_interrupt += 1
                if self.cnt_interrupt <= self.interrupt_time:
                    return self.p_imu[0, 0], self.p_imu[1, 0], self.p_imu[2, 0]
                elif self.cnt_interrupt == self.interrupt_time+1:
                    # 中断后，重新用gnss位置速度复位
                    print(f'interrupt end in {gnss_utc}')
                    self.p_imu = self.baseline.loc[self.current_step + (self.pos_num - 1), ['E_RLpredict', 'N_RLpredict','U_RLpredict']].values.reshape((3, 1))
                    self.v_imu = self.baseline.loc[self.current_step + (self.pos_num - 1), ['VE_wls', 'VN_wls', 'VU_wls']].values.reshape((3, 1))
                    return self.p_imu[0, 0], self.p_imu[1, 0], self.p_imu[2, 0]

        # 观测信息提取
        z = zs.reshape((3, 1))  # 位置
        u = us.reshape((3, 1))  # 速度
        # TODO 可以直接把矩阵导进来，减少计算量
        # update RM RN
        ecef_wls = np.array(pm.enu2ecef(z[0], z[1], z[2],self.state_llh[0], self.state_llh[1],self.state_llh[2]))
        llh_wls = np.array(pm.ecef2geodetic(ecef_wls[0], ecef_wls[1], ecef_wls[2]))
        RM = calculate_RM(llh_wls[0])  # 计算曲率半径
        RN = calculate_RN(llh_wls[0])

        # 计算状态矩阵
        Fv = np.array([[0, acc_enu[2, 0], -acc_enu[1, 0]],
                       [-acc_enu[2, 0], 0, acc_enu[0, 0]],
                       [acc_enu[1, 0], -acc_enu[0, 0], 0]])
        lat_rad = math.radians(z[0, 0])  # 将纬度转为弧度
        tan_lat = math.tan(lat_rad)
        Fd = np.array([[0, 1 / (RM + z[2, 0]), 0],
                       [-1 / (RN + z[2, 0]), 0, 0],
                       [-tan_lat / (RN + z[2, 0]), 0, 0]])

        tol = (gnss_utc - gnss_utc_pre) * 1e-3  # 时间间隔
        F = np.block([[IM, IM * tol, ZM, ZM],
                      [ZM, IM, Fv * tol, IM * tol],
                      [ZM, Fd * tol, IM, ZM],
                      [ZM, ZM, ZM, IM]])

        # F = torch.from_numpy(F).to(self.device) # trasfer to tensor
        self.dx = F @ self.dx
        self.P = (F @ self.P) @ F.T + self.covv
        # self.P = (F @ self.P) @ F.T + torch.from_numpy(self.covv).to(self.device) # to tensor

        dz = np.vstack((self.p_imu - z, self.v_imu - u))  # imu和gnss计算的位置和速度误差
        # dz = np.vstack((np.array(self.p_imu.cpu()) - z, np.array(self.v_imu.cpu()) - u))  # imu和gnss计算的位置和速度误差 # to tensor
        # dz = torch.from_numpy(dz).to(self.device)
        d = distance.mahalanobis(dz[:,0], self.H @ self.dx, np.linalg.inv(self.P[:6,:6]))
        # KF: Update step
        if d < sigma_mahalanobis:
            if self.conv_corr == 'conv_corr_1':  # Estimated WLS position covariance
                R = sigma_x ** 1.250 * np.eye(dim_R)
                R[2, 2] = R[2, 2] * 2
                R = R  + predict_R_noise
            elif self.conv_corr == 'conv_corr_2':
                R = self.covx + predict_R_noise
            R = np.where(R < 0, 0.01, R)
            self.covx = R

            # R = torch.from_numpy(R).double().to(self.device) # to tensor
            y = dz - self.H @ self.dx
            S = (self.H @ self.P) @ self.H.T + R
            K = (self.P @ self.H.T) @ np.linalg.inv(S)
            # S_inv = torch.from_numpy(np.linalg.inv(np.array(S.cpu()))).to(self.device)
            # K = (self.P @ self.H.T) @ S_inv # to tensor
            self.dx = self.dx + K @ y
            self.P = (self.I - (K @ self.H)) @ self.P
        else:
            # If observation update is not available, increase covariance
            self.P += 10 ** 2 * self.covv
            y = dz - self.H @ self.dx

        self.p_imu = self.p_imu - self.dx[0:3]
        self.v_imu = self.v_imu - self.dx[3:6]
        self.innovation = y.reshape((1, dim_R))/10 # normalization
        return self.p_imu[0,0], self.p_imu[1,0], self.p_imu[2,0]

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

class GPSPosition_continuous_lospos_convQR_onlyQNallcorrect(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len, noise_scale_dic,
                 conv_corr, interrupt_dic=None, allcorrect=True):
    # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(GPSPosition_continuous_lospos_convQR_onlyQNallcorrect, self).__init__()
        self.max_visible_sat=13
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * CN0PRUEA_num)),
             'pos': spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float),
             'Q_noise': spaces.Box(low=0, high=1, shape=(1, dim_Q), dtype=np.float),
             'R_noise': spaces.Box(low=0, high=1, shape=(1, dim_R), dtype=np.float),
             'innovation': spaces.Box(low=0, high=1, shape=(1, dim_R), dtype=np.float)})

        self.allcorrect = allcorrect  # correct in the end or in the obs
        if triptype == 'highway':
            self.tripIDlist = traj_highway
        elif triptype == 'urban':
            self.tripIDlist = traj_urban
        elif triptype == 'xiaomi':
            self.tripIDlist = traj_xiaomi
        elif triptype == 'xiaomiurban':
            self.tripIDlist = traj_xiaomiurban
        elif triptype == 'pixel567urban':
            self.tripIDlist = traj_pixel567urban
        elif triptype == 'smurban':
            self.tripIDlist = traj_smurban

        self.triptype = triptype
        self.traj_type = traj_type
        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.process_noise_scale = noise_scale_dic['process']
        self.measurement_noise_scale = noise_scale_dic['measurement']
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, dim_Q+dim_Q), dtype=np.float) # modified for RLKF
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
        elif self.trajdata_sort == 'randint':
            sublist = self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]]
            random.shuffle(sublist)
            self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]] = sublist
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)
        # noise cov correction parameter
        self.conv_corr = conv_corr

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort == 'randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum = random.randint(self.trajdata_range[0], self.trajdata_range[1])
        elif self.trajdata_sort == 'sorted':
            self.tripIDnum = self.tripIDnum + 1
            if self.tripIDnum > self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        self.baseline = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature_all[self.tripIDlist[self.tripIDnum]].copy()
        self.raw_acc_enu = data_raw_acc_enu_dic[self.tripIDlist[self.tripIDnum]].values
        self.RM_HF_dic = data_RM_HF_dic[self.tripIDlist[self.tripIDnum]]

        self.datatime = self.baseline['UnixTimeMillis']
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UnixTimeMillis'].values) - 1, 'UnixTimeMillis']
        # normalize baseline
        # self.baseline['LatitudeDegrees_norm'] = (self.baseline['LatitudeDegrees']-lat_min)/(lat_max-lat_min)
        # self.baseline['LongitudeDegrees_norm'] = (self.baseline['LongitudeDegrees']-lon_min)/(lon_max-lon_min)
        # gen pred
        if self.baseline_mod == 'bl':
            self.baseline['E_RLpredict'] = self.baseline['E_bl']
            self.baseline['N_RLpredict'] = self.baseline['N_bl']
            self.baseline['U_RLpredict'] = self.baseline['U_bl']
        elif self.baseline_mod == 'wls':
            self.baseline['E_RLpredict'] = self.baseline['E_wls']
            self.baseline['N_RLpredict'] = self.baseline['N_wls']
            self.baseline['U_RLpredict'] = self.baseline['U_wls']
        elif self.baseline_mod == 'kf':
            self.baseline['E_RLpredict'] = self.baseline['E_kf']
            self.baseline['N_RLpredict'] = self.baseline['N_kf']
            self.baseline['U_RLpredict'] = self.baseline['U_kf']

        if gnss_trig:
            self.gnss = gnss_dic[self.tripIDlist[self.tripIDnum]]
        # Set the current step to a random point within the data frame
        # revise 1017: need the specific percent of the traj
        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['E_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['N_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['U_RLpredict']] = None

        # 协方差矩阵初始化
        self.covv = sigma_v ** 1.250 * np.eye(dim_Q)
        self.covx = sigma_x ** 1.250 * np.eye(dim_R)
        self.covx[2, 2] = self.covx[2, 2] * 2

        # initial kf state
        self.state_llh = self.baseline.loc[0,['LatitudeDegrees','LongitudeDegrees','AltitudeMeters']].values # 初始点经纬度，转坐标系用
        self.P = 5.0 ** 2 * np.eye(dim_Q)  # initial State covariance
        self.I = np.eye(dim_Q)
        self.H = np.hstack((np.eye(dim_R),np.zeros((dim_R,dim_Q-dim_R))))
        self.dx = np.zeros([dim_Q, 1]) # initial erro state
        # trasfer to tensor
        # self.P = 5.0 ** 2 * torch.eye(dim_Q).double().to(self.device)  # initial State covariance
        # self.I = torch.eye(dim_Q).double().to(self.device)
        # self.H = torch.from_numpy(np.hstack((np.eye(6),np.zeros((6,dim_Q-6))))).to(self.device)
        # self.dx = torch.zeros([dim_Q, 1]).double().to(self.device) # initial erro state

        self.p_imu = self.baseline.loc[self.current_step+(self.pos_num - 2),['E_RLpredict','N_RLpredict','U_RLpredict']].values.reshape((3, 1))
        self.v_imu = self.baseline.loc[self.current_step+(self.pos_num - 2),['VE_wls','VN_wls','VU_wls']].values.reshape((3, 1))
        # trasfer to tensor
        # self.p_imu = torch.from_numpy(self.p_imu).double().to(self.device)
        # self.v_imu = torch.from_numpy(self.v_imu).double().to(self.device)
        self.raw_acc_utc = self.raw_acc_enu[:,0] # 加速度的时间戳矩阵
        utc_init = self.datatime[self.current_step + (self.pos_num - 2)]
        self.idx_acc = np.argmin(np.abs(self.raw_acc_utc - utc_init)) # 初始 加速度索引

        self.innovation = np.ones([1,dim_R])
        obs = self._next_observation()
        # must return in observation scale
        return obs  # self.tripIDnum#, obs#, {}

    def _normalize_pos(self, state):
        state[0] = state[0] / 10000
        state[1] = state[1] / 10000
        state[2] = state[2] / 100
        return state

    def _normalize_noise(self, obs_noise_R, obs_noise_Q):
        obs_noise_R = obs_noise_R / 100
        obs_noise_Q = obs_noise_Q / 10
        return obs_noise_R, obs_noise_Q

    def _normalize_los(self, gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        ## max normalize
        gnss[:, 1] = (gnss[:, 1]) / max(res_max, np.abs(res_min))
        gnss[:, 2] = (gnss[:, 2]) / max(losx_max, np.abs(losx_min))
        gnss[:, 3] = (gnss[:, 3]) / max(losy_max, np.abs(losy_min))
        gnss[:, 4] = (gnss[:, 4]) / max(losz_max, np.abs(losz_min))
        gnss[:, 5] = (gnss[:, 5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:, 6] = (gnss[:, 6]) / max(PRU_max, np.abs(PRU_min))
        gnss[:, 7] = (gnss[:, 7]) / max(AA_max, np.abs(AA_min))
        gnss[:, 8] = (gnss[:, 7]) / max(EA_max, np.abs(EA_min))
        # zero-score normalize
        # for i in range(gnss.shape[1]): # zero-score normalize
        #     if (i==0) or (i==gnss.shape[1]-1):
        #         continue
        #     mean,std = np.mean(gnss[:,i]),np.std(gnss[:,i])
        #     gnss[:,i] = (gnss[:,i]-mean) / std
        return gnss

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'E_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'N_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'U_RLpredict'].values])

        if self.baseline_mod == 'bl':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_bl']]],axis=1)
        elif self.baseline_mod == 'wls':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_wls']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_wls']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_wls']]],axis=1)
        elif self.baseline_mod == 'kf':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_kf']]],axis=1)

        obs = self._normalize_pos(obs)

        # gnss feature process
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features'].copy()
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        obs_feature = np.zeros([(self.max_visible_sat), CN0PRUEA_num])
        feature_index_list = [1, 2, 3, 4, 5, 6, 8]
        GPSL1_index = feature_tmp[:, 0] < 100
        feature_tmp = feature_tmp[:,feature_index_list]  # only GPSL1
        feature_tmp = feature_tmp[GPSL1_index,:]  # only GPSL1
        obs_feature[0:len(feature_tmp), :] = feature_tmp

        # noise cov feature process
        obs_Vnoise_pre = self.covv
        obs_Vnoise_cur = sigma_v ** 1.250 * np.eye(dim_Q)

        # obs_Vnoise_cur = sigma_v ** 1.250 * np.eye(dim_Q)
        if self.conv_corr == 'conv_corr_2': # 使用之前的不断迭代修改
            obs_noise_Q = np.diag(obs_Vnoise_pre)
            obs_noise_R = np.diag(self.covx)
        elif self.conv_corr == 'conv_corr_1': # 每次都是固定的修改
            obs_noise_Q = np.diag(obs_Vnoise_cur)
            obs_noise_R = np.diag(self.covx)

        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'),
                   'gnss': obs_feature.reshape(1, CN0PRUEA_num * self.max_visible_sat, order='C'),
                   'Q_noise': obs_noise_Q.reshape(1, dim_Q, order='C'), 'R_noise': obs_noise_R.reshape(1, dim_R, order='C'),
                   'innovation': self.innovation}

        # obs = obs.reshape(-1, 1) # + (traj_len-1)  + (traj_len-1)
        # obs=np.array([self.baseline.loc[self.current_step, 'LatitudeDegrees_norm'],self.baseline.loc[self.current_step, 'LongitudeDegrees_norm']])
        # obs=obs.reshape(2,1)
        # TODO latDeg lngDeg ... latDeg lngDeg
        return obs_all

    def step(self, action): # modified in 20250521
        done = (self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) * self.traj_type[-1] - (
            self.pos_num) - outlayer_in_end_ecef)
        timestep = self.baseline.loc[self.current_step + (self.pos_num - 1), 'UnixTimeMillis']
        # action for new prediction
        action = np.reshape(action, [1, dim_Q + dim_Q])
        predict_pe = action[0, 0] * self.continuous_action_scale
        predict_pn = action[0, 1] * self.continuous_action_scale
        predict_pu = action[0, 2] * self.continuous_action_scale * 2
        predict_ve = action[0, 3] * self.continuous_action_scale * 1e-1
        predict_vn = action[0, 4] * self.continuous_action_scale * 1e-1
        predict_vu = action[0, 5] * self.continuous_action_scale * 1e-2
        predict_pitch = action[0, 6] * self.continuous_action_scale
        predict_roll = action[0, 7] * self.continuous_action_scale
        predict_yaw = action[0, 8] * self.continuous_action_scale
        predict_ae = action[0, 9] * self.continuous_action_scale * 1e-2
        predict_an = action[0, 10] * self.continuous_action_scale * 1e-2
        predict_au = action[0, 11] * self.continuous_action_scale * 1e-2
        processnoise = np.array([predict_pe,predict_pn,predict_pu,predict_ve,predict_vn,predict_vu,
                                 predict_pitch,predict_roll,predict_yaw,predict_ae,predict_an,predict_au]).reshape(12,1)

        # 协方差调整
        diag = [action[0, 12],action[0, 13],action[0, 14],action[0, 15],action[0, 16],action[0, 17],
                action[0, 18],action[0, 19],action[0, 20],action[0, 21],action[0, 22],action[0, 23]]
        predict_Q_noise = self.process_noise_scale * np.diagflat(diag)

        if self.baseline_mod == 'bl':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_bl']
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_bl']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_bl']
        elif self.baseline_mod == 'wls':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_wls']  # WLS结果作为观测
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_wls']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_wls']
        elif self.baseline_mod == 'kf':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_kf']
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_kf']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_kf']

        eskf_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_eskf_hfv2']
        eskf_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_eskf_hfv2']
        eskf_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_eskf_hfv2']

        v_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VE_wls']
        v_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VN_wls']
        v_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VU_wls']
        gro_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_gt']
        gro_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_gt']
        gro_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_gt']

        # modified for RLKF
        p_wls = np.array([obs_e, obs_n, obs_u])
        v_wls = np.array([v_e, v_n, v_u])
        rl_e, rl_n, rl_u = self.RL4ESKF_IMU(p_wls, v_wls, processnoise, predict_Q_noise)
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['E_RLpredict']] = rl_e
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['N_RLpredict']] = rl_n
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['U_RLpredict']] = rl_u

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['E_RLpredict']] = rl_e
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['N_RLpredict']] = rl_n
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['U_RLpredict']] = rl_u
        # reward function
        if self.reward_setting == 'RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2)) * 1e0  # *1e5
        # TODO 水平和高程是否需要加比例因子
        elif self.reward_setting == 'RMSEadv':
            reward = np.sqrt(((obs_e - gro_e) ** 2 + (obs_n - gro_n) ** 2 + (obs_u - gro_u) ** 2)) - \
                     np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))
        elif self.reward_setting == 'RMSEadv_eskf':
            reward = np.sqrt(((eskf_e - gro_e) ** 2 + (eskf_n - gro_n) ** 2 + (eskf_u - gro_u) ** 2)) - \
                     np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))
        elif self.reward_setting == 'RMSEadv_re': # 高程误差优势除2
            reward = np.sqrt(((eskf_e - gro_e) ** 2 + (eskf_n - gro_n) ** 2)) - np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2)) \
                      + (np.sqrt((eskf_u - gro_u) ** 2) - np.sqrt((rl_u - gro_u) ** 2)) * 5e-1

        error = np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))

        if step_print:
            print(
                f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                f'RL dist: [{np.abs(rl_e - gro_e):.2f}, {np.abs(rl_n - gro_n):.2f}, {np.abs(rl_u - gro_u):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,'baseline': self.baseline,
                                   'error':error, 'tripid':self.tripIDnum}  # self.info#, {}# , 'data_truth_dic':data_truth_dic

    def RL4ESKF_IMU(self,zs, us, processnoise, predict_Q_noise, policy=True): # RL for KF modified in 0303
        # Parameters
        sigma_mahalanobis = 100  # Mahalanobis distance for rejecting innovation
        # position and velocity by imu
        acc_utc = self.raw_acc_utc[self.idx_acc]
        gnss_utc = self.datatime[self.current_step + (self.pos_num - 1)]  # current gnss utc
        gnss_utc_pre = self.datatime[self.current_step + (self.pos_num - 2)]  # previous gnss utc
        while abs(acc_utc - gnss_utc) > 11:
            acc_enu = self.raw_acc_enu[self.idx_acc, 1:4].reshape((3, 1))
            # acc_enu = acc_enu - np.array(self.dx[9:12].cpu()) # to tensor
            acc_enu = acc_enu - self.dx[9:12]
            # acc_enu_tensor = torch.from_numpy(acc_enu).double().to(self.device)  # to tensor
            self.v_imu = self.v_imu + acc_enu * (acc_utc - self.raw_acc_utc[self.idx_acc - 1]) * 1e-3  # 时间差 # to tensor
            self.p_imu = self.p_imu + self.v_imu * (acc_utc - self.raw_acc_utc[self.idx_acc - 1]) * 1e-3
            self.idx_acc = self.idx_acc + 1
            acc_utc = self.raw_acc_utc[self.idx_acc]

        # 观测信息提取
        z = zs.reshape((3, 1))  # 位置
        u = us.reshape((3, 1))  # 速度
        # TODO 可以直接把矩阵导进来，减少计算量
        # update RM RN
        ecef_wls = np.array(pm.enu2ecef(z[0], z[1], z[2], self.state_llh[0], self.state_llh[1], self.state_llh[2]))
        llh_wls = np.array(pm.ecef2geodetic(ecef_wls[0], ecef_wls[1], ecef_wls[2]))
        RM = calculate_RM(llh_wls[0])  # 计算曲率半径
        RN = calculate_RN(llh_wls[0])

        # 计算状态矩阵
        Fv = np.array([[0, acc_enu[2, 0], -acc_enu[1, 0]],
                       [-acc_enu[2, 0], 0, acc_enu[0, 0]],
                       [acc_enu[1, 0], -acc_enu[0, 0], 0]])
        lat_rad = math.radians(z[0, 0])  # 将纬度转为弧度
        tan_lat = math.tan(lat_rad)
        Fd = np.array([[0, 1 / (RM + z[2, 0]), 0],
                       [-1 / (RN + z[2, 0]), 0, 0],
                       [-tan_lat / (RN + z[2, 0]), 0, 0]])

        tol = (gnss_utc - gnss_utc_pre) * 1e-3  # 时间间隔
        F = np.block([[IM, IM * tol, ZM, ZM],
                      [ZM, IM, Fv * tol, IM * tol],
                      [ZM, Fd * tol, IM, ZM],
                      [ZM, ZM, ZM, IM]])

        if self.conv_corr == 'conv_corr_1':  # Estimated WLS position covariance
            Q = sigma_v ** 1.250 * np.eye(dim_R)
            Q = Q + predict_Q_noise
        elif self.conv_corr == 'conv_corr_2':
            Q = self.covv + predict_Q_noise
        Q = np.where(Q < 0, 0.01, Q)
        self.covv = Q
        # F = torch.from_numpy(F).to(self.device) # trasfer to tensor
        self.dx = F @ self.dx + processnoise
        self.P = (F @ self.P) @ F.T + self.covv
        # self.P = (F @ self.P) @ F.T + torch.from_numpy(self.covv).to(self.device) # to tensor

        dz = np.vstack((self.p_imu - z, self.v_imu - u))  # imu和gnss计算的位置和速度误差
        # dz = np.vstack((np.array(self.p_imu.cpu()) - z, np.array(self.v_imu.cpu()) - u))  # imu和gnss计算的位置和速度误差 # to tensor
        # dz = torch.from_numpy(dz).to(self.device)
        d = distance.mahalanobis(dz[:, 0], self.H @ self.dx, np.linalg.inv(self.P[:6, :6]))
        # KF: Update step
        if d < sigma_mahalanobis:
            # R = torch.from_numpy(R).double().to(self.device) # to tensor
            y = dz - self.H @ self.dx
            S = (self.H @ self.P) @ self.H.T + self.covx
            K = (self.P @ self.H.T) @ np.linalg.inv(S)
            # S_inv = torch.from_numpy(np.linalg.inv(np.array(S.cpu()))).to(self.device)
            # K = (self.P @ self.H.T) @ S_inv # to tensor
            self.dx = self.dx + K @ y
            self.P = (self.I - (K @ self.H)) @ self.P
        else:
            # If observation update is not available, increase covariance
            self.P += 10 ** 2 * self.covv
            y = dz - self.H @ self.dx

        self.p_imu = self.p_imu - self.dx[0:3]
        self.v_imu = self.v_imu - self.dx[3:6]
        self.innovation = y.reshape((1, dim_R)) / 10  # normalization
        return self.p_imu[0, 0], self.p_imu[1, 0], self.p_imu[2, 0]

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')


class GPSPosition_continuous_lospos(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace,
                 reward_setting, trajdata_sort, baseline_mod, traj_len, noise_scale_dic,
                 conv_corr):
        # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(GPSPosition_continuous_lospos, self).__init__()
        self.max_visible_sat = 13
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict({'gnss': spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * 4)),
                                              'pos': spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num),
                                                                dtype=np.float)})

        if triptype == 'highway':
            self.tripIDlist = traj_highway
        elif triptype == 'urban':
            self.tripIDlist = traj_else

        self.triptype = triptype
        self.traj_type = traj_type
        # continuous action
        if trajdata_range == 'full':
            self.trajdata_range = [0, len(self.tripIDlist) - 1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.process_noise_scale = noise_scale_dic['process']
        self.measurement_noise_scale = noise_scale_dic['measurement']
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 6),
                                       dtype=np.float)  # modified for RLKF
        self.total_reward = 0
        self.reward_setting = reward_setting
        self.trajdata_sort = trajdata_sort
        self.baseline_mod = baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)
        # noise cov correction parameter
        self.conv_corr = conv_corr

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort == 'randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum = random.randint(self.trajdata_range[0], self.trajdata_range[1])
        elif self.trajdata_sort == 'sorted':
            self.tripIDnum = self.tripIDnum + 1
            if self.tripIDnum > self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        self.baseline = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.kf_realtime = data_kf_realtime[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature_KF[self.tripIDlist[self.tripIDnum]].copy()
        self.baseline_speed = data_truth_velocity_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.covxv = cov_xv_dic[self.tripIDlist[self.tripIDnum]].copy()
        # For some unknown reason, the speed estimation is wrong only for XiaomiMi8
        # so the variance is increased
        VX = self.baseline_speed['VXEcefMeters_wls_igst'].values
        VY = self.baseline_speed['VYEcefMeters_wls_igst'].values
        VZ = self.baseline_speed['VZEcefMeters_wls_igst'].values
        if 'XiaomiMi8' in self.tripIDlist[self.tripIDnum]:
            VX = np.append((VX[:-1] + VX[1:]) / 2, 0)
            VY = np.append((VY[:-1] + VY[1:]) / 2, 0)
            VZ = np.append((VZ[:-1] + VZ[1:]) / 2, 0)
            self.covxv['covv'] = 1000.0 ** 2 * self.covxv['covv']
        self.baseline_speed['VXEcefMeters_wls_igst'] = np.insert((VX[:-1] + VX[1:]) / 2, 0, 0)
        self.baseline_speed['VYEcefMeters_wls_igst'] = np.insert((VY[:-1] + VY[1:]) / 2, 0, 0)
        self.baseline_speed['VZEcefMeters_wls_igst'] = np.insert((VZ[:-1] + VZ[1:]) / 2, 0, 0)
        self.P = 5.0 ** 2 * np.eye(3)  # initial State covariance

        self.datatime = self.baseline['UnixTimeMillis']
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UnixTimeMillis'].values) - 1, 'UnixTimeMillis']
        # normalize baseline
        # self.baseline['LatitudeDegrees_norm'] = (self.baseline['LatitudeDegrees']-lat_min)/(lat_max-lat_min)
        # self.baseline['LongitudeDegrees_norm'] = (self.baseline['LongitudeDegrees']-lon_min)/(lon_max-lon_min)
        # gen pred
        if self.baseline_mod == 'bl':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_bl']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_bl']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_bl']
        elif self.baseline_mod == 'wls_igst':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_wls_igst']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_wls_igst']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_wls_igst']
        elif self.baseline_mod == 'kf':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_kf']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_kf']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_kf_igst']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_kf_igst']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_kf_igst']

        if gnss_trig:
            self.gnss = gnss_dic[self.tripIDlist[self.tripIDnum]]
        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(0, len(self.df.loc[:, 'latDeg_norm'].values) - (traj_len-1))
        self.visible_sat = satnumKF_df.loc[satnumKF_df.loc[:, self.tripIDlist[self.tripIDnum]] > 0, 'Svid'].to_numpy()
        # revise 1017: need the specific percent of the traj
        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Z_RLpredict']] = None

        obs = self._next_observation()
        # must return in observation scale
        return obs  # self.tripIDnum#, obs#, {}

    def _normalize_pos(self, state):
        state[0] = (state[0] - xecef_min) / (xecef_max - xecef_min)
        state[1] = (state[1] - yecef_min) / (yecef_max - yecef_min)
        state[2] = (state[2] - zecef_min) / (zecef_max - zecef_min)
        return state

    def _normalize_los(self, gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        ## max normalize
        gnss[:, 1] = (gnss[:, 1]) / max(res_max, np.abs(res_min))
        gnss[:, 2] = (gnss[:, 2]) / max(losx_max, np.abs(losx_min))
        gnss[:, 3] = (gnss[:, 3]) / max(losy_max, np.abs(losy_min))
        gnss[:, 4] = (gnss[:, 4]) / max(losz_max, np.abs(losz_min))
        # zero-score normalize
        # for i in range(gnss.shape[1]): # zero-score normalize
        #     if (i==0) or (i==gnss.shape[1]-1):
        #         continue
        #     mean,std = np.mean(gnss[:,i]),np.std(gnss[:,i])
        #     gnss[:,i] = (gnss[:,i]-mean) / std
        return gnss

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'Z_RLpredict'].values])

        if self.baseline_mod == 'bl':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_bl']]],
                            axis=1)
        elif self.baseline_mod == 'wls_igst':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_wls_igst']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_wls_igst']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_wls_igst']]],
                            axis=1)
        elif self.baseline_mod == 'kf':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf']]],
                            axis=1)
        elif self.baseline_mod == 'kf_igst':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf_igst']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf_igst']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf_igst']]],
                            axis=1)

        obs = self._normalize_pos(obs)
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        feature_tmp = self.losfeature[self.datatime[self.current_step + (self.pos_num - 1)]]['features']
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        obs_feature = np.zeros([(self.max_visible_sat), 4])
        for i in range(len(self.visible_sat)):
            if self.visible_sat[i] in feature_tmp[:, 0]:
                obs_feature[i, :] = feature_tmp[feature_tmp[:, 0] == self.visible_sat[i], 1:]

        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'),
                   'gnss': obs_feature.reshape(1, 4 * self.max_visible_sat, order='C')}

        # obs = obs.reshape(-1, 1) # + (traj_len-1)  + (traj_len-1)
        # obs=np.array([self.baseline.loc[self.current_step, 'LatitudeDegrees_norm'],self.baseline.loc[self.current_step, 'LongitudeDegrees_norm']])
        # obs=obs.reshape(2,1)
        # TODO latDeg lngDeg ... latDeg lngDeg
        return obs_all

    def step(self, action):  # modified in 3.3
        # judge if end #
        done = (self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) * self.traj_type[-1] - (
            self.pos_num) - outlayer_in_end_ecef)
        timestep = self.baseline.loc[self.current_step + (self.pos_num - 1), 'UnixTimeMillis']
        # action for new prediction
        action = np.reshape(action, [1, 6])
        predict_x = action[0, 0] * self.continuous_action_scale
        predict_y = action[0, 1] * self.continuous_action_scale
        predict_z = action[0, 2] * self.continuous_action_scale
        predict_vx = action[0, 3] * self.continuous_action_scale * 1e-6
        predict_vy = action[0, 4] * self.continuous_action_scale * 1e-6
        predict_vz = action[0, 5] * self.continuous_action_scale * 1e-6

        Q_noise_vector = np.array([[predict_vx], [predict_vy], [predict_vz]])
        R_noise_vector = np.array([[predict_x], [predict_y], [predict_z]])
        predict_Q_noise = self.process_noise_scale * Q_noise_vector @ Q_noise_vector.T
        predict_R_noise = self.measurement_noise_scale * R_noise_vector @ R_noise_vector.T

        if self.baseline_mod == 'bl':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_bl']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_bl']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_bl']
        elif self.baseline_mod == 'wls_igst':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_wls_igst']  # WLS结果作为观测
            obs_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_wls_igst']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_wls_igst']
            kf_x = self.kf_realtime.loc[
                self.current_step + (self.pos_num - 1), 'XEcefMeters_KF_realtime']  # 实时KF作为baselime
            kf_y = self.kf_realtime.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_KF_realtime']
            kf_z = self.kf_realtime.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_KF_realtime']
            v_x = self.baseline_speed.loc[self.current_step + (self.pos_num - 1), 'VXEcefMeters_wls_igst']
            v_y = self.baseline_speed.loc[self.current_step + (self.pos_num - 1), 'VYEcefMeters_wls_igst']
            v_z = self.baseline_speed.loc[self.current_step + (self.pos_num - 1), 'VZEcefMeters_wls_igst']
        elif self.baseline_mod == 'kf':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf_igst']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf_igst']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf_igst']
        gro_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ecefX']
        gro_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ecefY']
        gro_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ecefZ']
        # modified for RLKF
        # rl_x = obs_x + predict_x
        # rl_y = obs_y + predict_y
        # rl_z = obs_z + predict_z
        x_wls = np.array([obs_x + predict_x, obs_y + predict_y, obs_z + predict_z])
        v_wls = np.array([v_x + predict_vx, v_y + predict_vy, v_z + predict_vz])
        rl_x, rl_y, rl_z = self.RL4KFGSDC(x_wls, v_wls, predict_Q_noise, predict_R_noise)

        self.baseline.loc[self.current_step + (self.pos_num - 1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Z_RLpredict']] = rl_z
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[
            self.current_step + (self.pos_num - 1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[
            self.current_step + (self.pos_num - 1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[
            self.current_step + (self.pos_num - 1), ['Z_RLpredict']] = rl_z
        # reward function
        if self.reward_setting == 'RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)) * 1e0  # *1e5
        elif self.reward_setting == 'RMSEadv':
            reward = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2)) * 1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)) * 1e0
        elif self.reward_setting == 'RMSEadv_kf':
            reward = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2)) * 1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)) * 1e0

        if step_print:
            print(
                f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                f'RL dist: [{np.abs(rl_x - gro_x):.2f}, {np.abs(rl_y - gro_y):.2f}, {np.abs(rl_z - gro_z):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,
                                   'baseline': self.baseline}  # self.info#, {}# , 'data_truth_dic':data_truth_dic

    def RL4KFGSDC(self, zs, us, predict_Q_noise, predict_R_noise):  # RL for KF modified in 0303
        # Parameters
        sigma_mahalanobis = 30.0  # Mahalanobis distance for rejecting innovation
        dim_x = zs.shape[0]
        F = np.eye(dim_x)  # Transition matrix
        H = np.eye(dim_x)  # Measurement function
        # Initial state and covariance
        x = np.array([self.baseline.loc[self.current_step + (self.pos_num - 2), 'X_RLpredict'],
                      self.baseline.loc[self.current_step + (self.pos_num - 2), 'Y_RLpredict'],
                      self.baseline.loc[
                          self.current_step + (self.pos_num - 2), 'Z_RLpredict']])  # State: 使用上一个时刻RL预测的位置

        I = np.eye(dim_x)

        # KF: Prediction step
        ## Estimated WLS velocity covariance
        if self.conv_corr == 'conv_corr_1':
            Q = self.covxv['covv'][int(self.current_step + (self.pos_num - 1))] - predict_Q_noise  #
        elif self.conv_corr == 'conv_corr_2':
            Q_ = self.covxv['covv'][int(self.current_step + (self.pos_num - 2))]
            Q = (Q_ + self.covxv['covv'][
                int(self.current_step + (self.pos_num - 1))] - predict_Q_noise) * 0.5  # 前一时刻和当前时刻相加求均值
        elif self.conv_corr == 'conv_corr_3':
            Q = self.covxv['covv'][int(self.current_step + (self.pos_num - 2))] - predict_Q_noise  #
        elif self.conv_corr == 'conv_corr_4':
            Q = np.abs(np.abs(self.covxv['covv'][int(self.current_step + (self.pos_num - 1))]) - np.abs(
                predict_Q_noise))  # 保证修正后协方差往小的方向走

        self.covxv['covv'][int(self.current_step + (self.pos_num - 1))] = Q

        x = F @ x + us.T
        self.P = (F @ self.P) @ F.T + Q
        d = distance.mahalanobis(zs, H @ x, np.linalg.inv(self.P))
        # KF: Update step
        if d < sigma_mahalanobis:
            if self.conv_corr == 'conv_corr_1':  # Estimated WLS position covariance
                R = self.covxv['covx'][int(self.current_step + (self.pos_num - 1))] - predict_R_noise
            elif self.conv_corr == 'conv_corr_2':
                R_ = self.covxv['covx'][int(self.current_step + (self.pos_num - 2))]
                R = (R_ + self.covxv['covx'][
                    int(self.current_step + (self.pos_num - 1))] - predict_R_noise) * 0.5  # 前一时刻和当前时刻相加求均值
            elif self.conv_corr == 'conv_corr_3':
                R = self.covxv['covx'][int(self.current_step + (self.pos_num - 2))] - predict_R_noise
            elif self.conv_corr == 'conv_corr_4':
                R = np.abs(np.abs(self.covxv['covx'][int(self.current_step + (self.pos_num - 1))]) - np.abs(
                    predict_R_noise))  # 保证修正后协方差往小的方向走

            self.covxv['covx'][int(self.current_step + (self.pos_num - 1))] = R

            y = zs.T - H @ x
            S = (H @ self.P) @ H.T + R
            K = (self.P @ H.T) @ np.linalg.inv(S)
            x = x + K @ y
            self.P = (I - (K @ H)) @ self.P
        else:
            # If observation update is not available, increase covariance
            self.P += 10 ** 2 * Q

        return x[0], x[1], x[2]

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')


class GPSPosition_continuous_lospos_onlyR(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace,
                 reward_setting, trajdata_sort, baseline_mod, traj_len, noise_scale_dic,
                 conv_corr):
        # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(GPSPosition_continuous_lospos_onlyR, self).__init__()
        self.max_visible_sat = 13
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict({'gnss': spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * 4)),
                                              'pos': spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num),
                                                                dtype=np.float)})

        if triptype == 'highway':
            self.tripIDlist = traj_highway
        elif triptype == 'urban':
            self.tripIDlist = traj_else

        self.triptype = triptype
        self.traj_type = traj_type
        # continuous action
        if trajdata_range == 'full':
            self.trajdata_range = [0, len(self.tripIDlist) - 1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.process_noise_scale = noise_scale_dic['process']
        self.measurement_noise_scale = noise_scale_dic['measurement']
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, 3),
                                       dtype=np.float)  # modified for RLKF
        self.total_reward = 0
        self.reward_setting = reward_setting
        self.trajdata_sort = trajdata_sort
        self.baseline_mod = baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)
        # noise cov correction parameter
        self.conv_corr = conv_corr

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort == 'randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum = random.randint(self.trajdata_range[0], self.trajdata_range[1])
        elif self.trajdata_sort == 'sorted':
            self.tripIDnum = self.tripIDnum + 1
            if self.tripIDnum > self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        self.baseline = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.kf_realtime = data_kf_realtime[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature_KF[self.tripIDlist[self.tripIDnum]].copy()
        self.baseline_speed = data_truth_velocity_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.covxv = cov_xv_dic[self.tripIDlist[self.tripIDnum]].copy()
        # For some unknown reason, the speed estimation is wrong only for XiaomiMi8
        # so the variance is increased
        VX = self.baseline_speed['VXEcefMeters_wls_igst'].values
        VY = self.baseline_speed['VYEcefMeters_wls_igst'].values
        VZ = self.baseline_speed['VZEcefMeters_wls_igst'].values
        if 'XiaomiMi8' in self.tripIDlist[self.tripIDnum]:
            VX = np.append((VX[:-1] + VX[1:]) / 2, 0)
            VY = np.append((VY[:-1] + VY[1:]) / 2, 0)
            VZ = np.append((VZ[:-1] + VZ[1:]) / 2, 0)
            self.covxv['covv'] = 1000.0 ** 2 * self.covxv['covv']
        self.baseline_speed['VXEcefMeters_wls_igst'] = np.insert((VX[:-1] + VX[1:]) / 2, 0, 0)
        self.baseline_speed['VYEcefMeters_wls_igst'] = np.insert((VY[:-1] + VY[1:]) / 2, 0, 0)
        self.baseline_speed['VZEcefMeters_wls_igst'] = np.insert((VZ[:-1] + VZ[1:]) / 2, 0, 0)
        self.P = 5.0 ** 2 * np.eye(3)  # initial State covariance

        self.datatime = self.baseline['UnixTimeMillis']
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UnixTimeMillis'].values) - 1, 'UnixTimeMillis']
        # normalize baseline
        # self.baseline['LatitudeDegrees_norm'] = (self.baseline['LatitudeDegrees']-lat_min)/(lat_max-lat_min)
        # self.baseline['LongitudeDegrees_norm'] = (self.baseline['LongitudeDegrees']-lon_min)/(lon_max-lon_min)
        # gen pred
        if self.baseline_mod == 'bl':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_bl']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_bl']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_bl']
        elif self.baseline_mod == 'wls_igst':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_wls_igst']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_wls_igst']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_wls_igst']
        elif self.baseline_mod == 'kf':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_kf']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_kf']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            self.baseline['X_RLpredict'] = self.baseline['XEcefMeters_kf_igst']
            self.baseline['Y_RLpredict'] = self.baseline['YEcefMeters_kf_igst']
            self.baseline['Z_RLpredict'] = self.baseline['ZEcefMeters_kf_igst']

        if gnss_trig:
            self.gnss = gnss_dic[self.tripIDlist[self.tripIDnum]]
        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(0, len(self.df.loc[:, 'latDeg_norm'].values) - (traj_len-1))
        self.visible_sat = satnumKF_df.loc[satnumKF_df.loc[:, self.tripIDlist[self.tripIDnum]] > 0, 'Svid'].to_numpy()
        # revise 1017: need the specific percent of the traj
        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['X_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Y_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step - 1, ['Z_RLpredict']] = None

        obs = self._next_observation()
        # must return in observation scale
        return obs  # self.tripIDnum#, obs#, {}

    def _normalize_pos(self, state):
        state[0] = (state[0] - xecef_min) / (xecef_max - xecef_min)
        state[1] = (state[1] - yecef_min) / (yecef_max - yecef_min)
        state[2] = (state[2] - zecef_min) / (zecef_max - zecef_min)
        return state

    def _normalize_los(self, gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        ## max normalize
        gnss[:, 1] = (gnss[:, 1]) / max(res_max, np.abs(res_min))
        gnss[:, 2] = (gnss[:, 2]) / max(losx_max, np.abs(losx_min))
        gnss[:, 3] = (gnss[:, 3]) / max(losy_max, np.abs(losy_min))
        gnss[:, 4] = (gnss[:, 4]) / max(losz_max, np.abs(losz_min))
        # zero-score normalize
        # for i in range(gnss.shape[1]): # zero-score normalize
        #     if (i==0) or (i==gnss.shape[1]-1):
        #         continue
        #     mean,std = np.mean(gnss[:,i]),np.std(gnss[:,i])
        #     gnss[:,i] = (gnss[:,i]-mean) / std
        return gnss

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'X_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'Y_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'Z_RLpredict'].values])

        if self.baseline_mod == 'bl':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_bl']]],
                            axis=1)
        elif self.baseline_mod == 'wls_igst':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_wls_igst']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_wls_igst']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_wls_igst']]],
                            axis=1)
        elif self.baseline_mod == 'kf':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf']]],
                            axis=1)
        elif self.baseline_mod == 'kf_igst':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf_igst']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf_igst']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf_igst']]],
                            axis=1)

        obs = self._normalize_pos(obs)
        # obs_f=self.losfeature[self.datatime[self.current_step + (traj_len-1)]]
        feature_tmp = self.losfeature[self.datatime[self.current_step + (self.pos_num - 1)]]['features']
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        obs_feature = np.zeros([(self.max_visible_sat), 4])
        for i in range(len(self.visible_sat)):
            if self.visible_sat[i] in feature_tmp[:, 0]:
                obs_feature[i, :] = feature_tmp[feature_tmp[:, 0] == self.visible_sat[i], 1:]

        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'),
                   'gnss': obs_feature.reshape(1, 4 * self.max_visible_sat, order='C')}

        # obs = obs.reshape(-1, 1) # + (traj_len-1)  + (traj_len-1)
        # obs=np.array([self.baseline.loc[self.current_step, 'LatitudeDegrees_norm'],self.baseline.loc[self.current_step, 'LongitudeDegrees_norm']])
        # obs=obs.reshape(2,1)
        # TODO latDeg lngDeg ... latDeg lngDeg
        return obs_all

    def step(self, action):  # modified in 3.3
        # judge if end #
        done = (self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) * self.traj_type[-1] - (
            self.pos_num) - outlayer_in_end_ecef)
        timestep = self.baseline.loc[self.current_step + (self.pos_num - 1), 'UnixTimeMillis']
        # action for new prediction
        action = np.reshape(action, [1, 3])
        predict_x = action[0, 0] * self.continuous_action_scale
        predict_y = action[0, 1] * self.continuous_action_scale
        predict_z = action[0, 2] * self.continuous_action_scale

        R_noise_vector = np.array([[predict_x], [predict_y], [predict_z]])
        predict_R_noise = self.measurement_noise_scale * R_noise_vector @ R_noise_vector.T

        if self.baseline_mod == 'bl':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_bl']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_bl']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_bl']
        elif self.baseline_mod == 'wls_igst':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_wls_igst']  # WLS结果作为观测
            obs_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_wls_igst']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_wls_igst']
            kf_x = self.kf_realtime.loc[
                self.current_step + (self.pos_num - 1), 'XEcefMeters_KF_realtime']  # 实时KF作为baselime
            kf_y = self.kf_realtime.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_KF_realtime']
            kf_z = self.kf_realtime.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_KF_realtime']
            v_x = self.baseline_speed.loc[self.current_step + (self.pos_num - 1), 'VXEcefMeters_wls_igst']
            v_y = self.baseline_speed.loc[self.current_step + (self.pos_num - 1), 'VYEcefMeters_wls_igst']
            v_z = self.baseline_speed.loc[self.current_step + (self.pos_num - 1), 'VZEcefMeters_wls_igst']
        elif self.baseline_mod == 'kf':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf']
        elif self.baseline_mod == 'kf_igst':
            obs_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'XEcefMeters_kf_igst']
            obs_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'YEcefMeters_kf_igst']
            obs_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ZEcefMeters_kf_igst']
        gro_x = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ecefX']
        gro_y = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ecefY']
        gro_z = self.baseline.loc[self.current_step + (self.pos_num - 1), 'ecefZ']
        # modified for RLKF
        # rl_x = obs_x + predict_x
        # rl_y = obs_y + predict_y
        # rl_z = obs_z + predict_z
        x_wls = np.array([obs_x + predict_x, obs_y + predict_y, obs_z + predict_z])
        v_wls = np.array([v_x, v_y, v_z])
        rl_x, rl_y, rl_z = self.RL4KFGSDC(x_wls, v_wls, predict_R_noise)

        self.baseline.loc[self.current_step + (self.pos_num - 1), ['X_RLpredict']] = rl_x
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Y_RLpredict']] = rl_y
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['Z_RLpredict']] = rl_z
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[
            self.current_step + (self.pos_num - 1), ['X_RLpredict']] = rl_x
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[
            self.current_step + (self.pos_num - 1), ['Y_RLpredict']] = rl_y
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[
            self.current_step + (self.pos_num - 1), ['Z_RLpredict']] = rl_z
        # reward function
        if self.reward_setting == 'RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)) * 1e0  # *1e5
        elif self.reward_setting == 'RMSEadv':
            reward = np.sqrt(((obs_x - gro_x) ** 2 + (obs_y - gro_y) ** 2 + (obs_z - gro_z) ** 2)) * 1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)) * 1e0
        elif self.reward_setting == 'RMSEadv_kf':
            reward = np.sqrt(((kf_x - gro_x) ** 2 + (kf_y - gro_y) ** 2 + (kf_z - gro_z) ** 2)) * 1e0 - \
                     np.sqrt(((rl_x - gro_x) ** 2 + (rl_y - gro_y) ** 2 + (rl_z - gro_z) ** 2)) * 1e0

        if step_print:
            print(
                f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                f'RL dist: [{np.abs(rl_x - gro_x):.2f}, {np.abs(rl_y - gro_y):.2f}, {np.abs(rl_z - gro_z):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()
        return obs, reward, done, {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,
                                   'baseline': self.baseline}  # self.info#, {}# , 'data_truth_dic':data_truth_dic

    def RL4KFGSDC(self, zs, us, predict_R_noise):  # RL for KF modified in 0303
        # Parameters
        sigma_mahalanobis = 30.0  # Mahalanobis distance for rejecting innovation
        dim_x = zs.shape[0]
        F = np.eye(dim_x)  # Transition matrix
        H = np.eye(dim_x)  # Measurement function
        # Initial state and covariance
        x = np.array([self.baseline.loc[self.current_step + (self.pos_num - 2), 'X_RLpredict'],
                      self.baseline.loc[self.current_step + (self.pos_num - 2), 'Y_RLpredict'],
                      self.baseline.loc[
                          self.current_step + (self.pos_num - 2), 'Z_RLpredict']])  # State: 使用上一个时刻RL预测的位置

        I = np.eye(dim_x)

        # KF: Prediction step
        ## Estimated WLS velocity covariance
        Q = self.covxv['covv'][int(self.current_step + (self.pos_num - 1))]  #

        x = F @ x + us.T
        self.P = (F @ self.P) @ F.T + Q
        d = distance.mahalanobis(zs, H @ x, np.linalg.inv(self.P))
        # KF: Update step
        if d < sigma_mahalanobis:
            if self.conv_corr == 'conv_corr_1':  # Estimated WLS position covariance
                R = self.covxv['covx'][int(self.current_step + (self.pos_num - 1))] - predict_R_noise
            elif self.conv_corr == 'conv_corr_2':
                R_ = self.covxv['covx'][int(self.current_step + (self.pos_num - 2))]
                R = (R_ + self.covxv['covx'][
                    int(self.current_step + (self.pos_num - 1))] - predict_R_noise) * 0.5  # 前一时刻和当前时刻相加求均值
            elif self.conv_corr == 'conv_corr_3':
                R = self.covxv['covx'][int(self.current_step + (self.pos_num - 2))] - predict_R_noise
            elif self.conv_corr == 'conv_corr_4':
                R = np.abs(np.abs(self.covxv['covx'][int(self.current_step + (self.pos_num - 1))]) - np.abs(
                    predict_R_noise))  # 保证修正后协方差往小的方向走

            self.covxv['covx'][int(self.current_step + (self.pos_num - 1))] = R

            y = zs.T - H @ x
            S = (H @ self.P) @ H.T + R
            K = (self.P @ H.T) @ np.linalg.inv(S)
            x = x + K @ y
            self.P = (I - (K @ H)) @ self.P
        else:
            # If observation update is not available, increase covariance
            self.P += 10 ** 2 * Q

        return x[0], x[1], x[2]

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

class Sage_GPSPosition_continuous_lospos_convQR_onlyQNallcorrect(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len, noise_scale_dic,
                 conv_corr, interrupt_dic=None, allcorrect=True, forgetting_factor = 0.99):
    # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(Sage_GPSPosition_continuous_lospos_convQR_onlyQNallcorrect, self).__init__()

        self.max_visible_sat=13
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * CN0PRUEA_num)),
             'pos': spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float32),
             'Q_noise': spaces.Box(low=0, high=1, shape=(1, dim_Q), dtype=np.float32),
             'R_noise': spaces.Box(low=0, high=1, shape=(1, dim_R), dtype=np.float32),
             'innovation': spaces.Box(low=0, high=1, shape=(1, dim_R), dtype=np.float32)})

        self.forgetting_factor = forgetting_factor  # 这个可以在0.95-0.99的范围之内调节
        self.allcorrect = allcorrect  # correct in the end or in the obs
        if triptype == 'highway':
            self.tripIDlist = traj_highway
        elif triptype == 'urban':
            self.tripIDlist = traj_urban
        elif triptype == 'xiaomi':
            self.tripIDlist = traj_xiaomi
        elif triptype == 'xiaomiurban':
            self.tripIDlist = traj_xiaomiurban
        elif triptype == 'pixel567urban':
            self.tripIDlist = traj_pixel567urban
        elif triptype == 'smurban':
            self.tripIDlist = traj_smurban

        self.triptype = triptype
        self.traj_type = traj_type
        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.process_noise_scale = noise_scale_dic['process']
        self.measurement_noise_scale = noise_scale_dic['measurement']
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, dim_Q+dim_Q), dtype=np.float32) # modified for RLKF
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
        elif self.trajdata_sort == 'randint':
            sublist = self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]]
            random.shuffle(sublist)
            self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]] = sublist
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)
        # noise cov correction parameter
        self.conv_corr = conv_corr

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort == 'randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum = random.randint(self.trajdata_range[0], self.trajdata_range[1])
        elif self.trajdata_sort == 'sorted':
            self.tripIDnum = self.tripIDnum + 1
            if self.tripIDnum > self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        self.baseline = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature_all[self.tripIDlist[self.tripIDnum]].copy()
        self.raw_acc_enu = data_raw_acc_enu_dic[self.tripIDlist[self.tripIDnum]].values
        self.RM_HF_dic = data_RM_HF_dic[self.tripIDlist[self.tripIDnum]]

        self.datatime = self.baseline['UnixTimeMillis']
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UnixTimeMillis'].values) - 1, 'UnixTimeMillis']
        # normalize baseline
        # self.baseline['LatitudeDegrees_norm'] = (self.baseline['LatitudeDegrees']-lat_min)/(lat_max-lat_min)
        # self.baseline['LongitudeDegrees_norm'] = (self.baseline['LongitudeDegrees']-lon_min)/(lon_max-lon_min)
        # gen pred
        if self.baseline_mod == 'bl':
            self.baseline['E_RLpredict'] = self.baseline['E_bl']
            self.baseline['N_RLpredict'] = self.baseline['N_bl']
            self.baseline['U_RLpredict'] = self.baseline['U_bl']
        elif self.baseline_mod == 'wls':
            self.baseline['E_RLpredict'] = self.baseline['E_wls']
            self.baseline['N_RLpredict'] = self.baseline['N_wls']
            self.baseline['U_RLpredict'] = self.baseline['U_wls']
        elif self.baseline_mod == 'kf':
            self.baseline['E_RLpredict'] = self.baseline['E_kf']
            self.baseline['N_RLpredict'] = self.baseline['N_kf']
            self.baseline['U_RLpredict'] = self.baseline['U_kf']

        if gnss_trig:
            self.gnss = gnss_dic[self.tripIDlist[self.tripIDnum]]
        # Set the current step to a random point within the data frame
        # revise 1017: need the specific percent of the traj
        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['E_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['N_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['U_RLpredict']] = None

        # 协方差矩阵初始化
        self.covv = sigma_v ** 1.250 * np.eye(dim_Q)
        self.covx = sigma_x ** 1.250 * np.eye(dim_R)
        self.covx[2, 2] = self.covx[2, 2] * 2

        # 添加上遗忘因子
        self.lamba = 0.95 # 一般设置为0。95-0.99

        # initial kf state
        self.state_llh = self.baseline.loc[0,['LatitudeDegrees','LongitudeDegrees','AltitudeMeters']].values # 初始点经纬度，转坐标系用
        self.P = 5.0 ** 2 * np.eye(dim_Q)  # initial State covariance
        self.P_prev = self.P.copy()
        self.I = np.eye(dim_Q)
        self.H = np.hstack((np.eye(dim_R),np.zeros((dim_R,dim_Q-dim_R))))
        self.dx = np.zeros([dim_Q, 1]) # initial erro state
        # trasfer to tensor
        # self.P = 5.0 ** 2 * torch.eye(dim_Q).double().to(self.device)  # initial State covariance
        # self.I = torch.eye(dim_Q).double().to(self.device)
        # self.H = torch.from_numpy(np.hstack((np.eye(6),np.zeros((6,dim_Q-6))))).to(self.device)
        # self.dx = torch.zeros([dim_Q, 1]).double().to(self.device) # initial erro state

        self.p_imu = self.baseline.loc[self.current_step+(self.pos_num - 2),['E_RLpredict','N_RLpredict','U_RLpredict']].values.reshape((3, 1))
        self.v_imu = self.baseline.loc[self.current_step+(self.pos_num - 2),['VE_wls','VN_wls','VU_wls']].values.reshape((3, 1))
        # trasfer to tensor
        # self.p_imu = torch.from_numpy(self.p_imu).double().to(self.device)
        # self.v_imu = torch.from_numpy(self.v_imu).double().to(self.device)
        self.raw_acc_utc = self.raw_acc_enu[:,0] # 加速度的时间戳矩阵
        utc_init = self.datatime[self.current_step + (self.pos_num - 2)]
        self.idx_acc = np.argmin(np.abs(self.raw_acc_utc - utc_init)) # 初始 加速度索引

        self.innovation = np.ones([1,dim_R])
        obs = self._next_observation()
        # must return in observation scale
        return obs  # self.tripIDnum#, obs#, {}

    def _normalize_pos(self, state):
        state[0] = state[0] / 10000
        state[1] = state[1] / 10000
        state[2] = state[2] / 100
        return state

    def _normalize_noise(self, obs_noise_R, obs_noise_Q):
        obs_noise_R = obs_noise_R / 100
        obs_noise_Q = obs_noise_Q / 10
        return obs_noise_R, obs_noise_Q

    def _normalize_los(self, gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        ## max normalize
        gnss[:, 1] = (gnss[:, 1]) / max(res_max, np.abs(res_min))
        gnss[:, 2] = (gnss[:, 2]) / max(losx_max, np.abs(losx_min))
        gnss[:, 3] = (gnss[:, 3]) / max(losy_max, np.abs(losy_min))
        gnss[:, 4] = (gnss[:, 4]) / max(losz_max, np.abs(losz_min))
        gnss[:, 5] = (gnss[:, 5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:, 6] = (gnss[:, 6]) / max(PRU_max, np.abs(PRU_min))
        gnss[:, 7] = (gnss[:, 7]) / max(AA_max, np.abs(AA_min))
        gnss[:, 8] = (gnss[:, 7]) / max(EA_max, np.abs(EA_min))
        # zero-score normalize
        # for i in range(gnss.shape[1]): # zero-score normalize
        #     if (i==0) or (i==gnss.shape[1]-1):
        #         continue
        #     mean,std = np.mean(gnss[:,i]),np.std(gnss[:,i])
        #     gnss[:,i] = (gnss[:,i]-mean) / std
        return gnss

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'E_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'N_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'U_RLpredict'].values])

        if self.baseline_mod == 'bl':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_bl']]],axis=1)
        elif self.baseline_mod == 'wls':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_wls']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_wls']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_wls']]],axis=1)
        elif self.baseline_mod == 'kf':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_kf']]],axis=1)

        obs = self._normalize_pos(obs)

        # gnss feature process
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features'].copy()
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        obs_feature = np.zeros([(self.max_visible_sat), CN0PRUEA_num])
        feature_index_list = [1, 2, 3, 4, 5, 6, 8]
        GPSL1_index = feature_tmp[:, 0] < 100
        feature_tmp = feature_tmp[:,feature_index_list]  # only GPSL1
        feature_tmp = feature_tmp[GPSL1_index,:]  # only GPSL1
        obs_feature[0:len(feature_tmp), :] = feature_tmp

        # noise cov feature process
        obs_Vnoise_pre = self.covv
        obs_Vnoise_cur = sigma_v ** 1.250 * np.eye(dim_Q)

        # obs_Vnoise_cur = sigma_v ** 1.250 * np.eye(dim_Q)
        if self.conv_corr == 'conv_corr_2': # 使用之前的不断迭代修改
            obs_noise_Q = np.diag(obs_Vnoise_pre)
            obs_noise_R = np.diag(self.covx)
        elif self.conv_corr == 'conv_corr_1': # 每次都是固定的修改
            obs_noise_Q = np.diag(obs_Vnoise_cur)
            obs_noise_R = np.diag(self.covx)

        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'),
                   'gnss': obs_feature.reshape(1, CN0PRUEA_num * self.max_visible_sat, order='C'),
                   'Q_noise': obs_noise_Q.reshape(1, dim_Q, order='C'), 'R_noise': obs_noise_R.reshape(1, dim_R, order='C'),
                   'innovation': self.innovation}

        # obs = obs.reshape(-1, 1) # + (traj_len-1)  + (traj_len-1)
        # obs=np.array([self.baseline.loc[self.current_step, 'LatitudeDegrees_norm'],self.baseline.loc[self.current_step, 'LongitudeDegrees_norm']])
        # obs=obs.reshape(2,1)
        # TODO latDeg lngDeg ... latDeg lngDeg
        return obs_all

    def step(self,action): # modified in 20250521

        # 忽略动作

        done = (self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) * self.traj_type[-1] - (
            self.pos_num) - outlayer_in_end_ecef)
        timestep = self.baseline.loc[self.current_step + (self.pos_num - 1), 'UnixTimeMillis']
        # # action for new prediction
        # action = np.reshape(action, [1, dim_Q + dim_Q])
        # predict_pe = action[0, 0] * self.continuous_action_scale
        # predict_pn = action[0, 1] * self.continuous_action_scale
        # predict_pu = action[0, 2] * self.continuous_action_scale * 2
        # predict_ve = action[0, 3] * self.continuous_action_scale * 1e-1
        # predict_vn = action[0, 4] * self.continuous_action_scale * 1e-1
        # predict_vu = action[0, 5] * self.continuous_action_scale * 1e-2
        # predict_pitch = action[0, 6] * self.continuous_action_scale
        # predict_roll = action[0, 7] * self.continuous_action_scale
        # predict_yaw = action[0, 8] * self.continuous_action_scale
        # predict_ae = action[0, 9] * self.continuous_action_scale * 1e-2
        # predict_an = action[0, 10] * self.continuous_action_scale * 1e-2
        # predict_au = action[0, 11] * self.continuous_action_scale * 1e-2
        # processnoise = np.array([predict_pe,predict_pn,predict_pu,predict_ve,predict_vn,predict_vu,
        #                          predict_pitch,predict_roll,predict_yaw,predict_ae,predict_an,predict_au]).reshape(12,1)

        # 协方差调整
        # diag = [action[0, 12],action[0, 13],action[0, 14],action[0, 15],action[0, 16],action[0, 17],
        #         action[0, 18],action[0, 19],action[0, 20],action[0, 21],action[0, 22],action[0, 23]]
        # predict_Q_noise = self.process_noise_scale * np.diagflat(diag)

        # 添加上processnoise
        prcessnoise = self.process_noise_scale*np.eye(dim_Q,1)

        if self.baseline_mod == 'bl':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_bl']
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_bl']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_bl']
        elif self.baseline_mod == 'wls':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_wls']  # WLS结果作为观测
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_wls']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_wls']
        elif self.baseline_mod == 'kf':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_kf']
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_kf']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_kf']

        eskf_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_eskf_hfv2']
        eskf_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_eskf_hfv2']
        eskf_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_eskf_hfv2']

        v_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VE_wls']
        v_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VN_wls']
        v_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VU_wls']
        gro_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_gt']
        gro_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_gt']
        gro_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_gt']

        # modified for RLKF  ????
        p_wls = np.array([obs_e, obs_n, obs_u])
        v_wls = np.array([v_e, v_n, v_u])

        #### 这一部分需要修改  #####
        z = np.array([obs_e, obs_n,obs_u])
        rl_e, rl_n, rl_u = self.SageHusaAKF(p_wls, v_wls, prcessnoise)

        self.baseline.loc[self.current_step + (self.pos_num - 1), ['E_RLpredict']] = rl_e
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['N_RLpredict']] = rl_n
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['U_RLpredict']] = rl_u

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['E_RLpredict']] = rl_e
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['N_RLpredict']] = rl_n
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['U_RLpredict']] = rl_u
        # reward function
        if self.reward_setting == 'RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2)) * 1e0  # *1e5
        # TODO 水平和高程是否需要加比例因子
        elif self.reward_setting == 'RMSEadv':
            reward = np.sqrt(((obs_e - gro_e) ** 2 + (obs_n - gro_n) ** 2 + (obs_u - gro_u) ** 2)) - \
                     np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))
        elif self.reward_setting == 'RMSEadv_eskf':
            reward = np.sqrt(((eskf_e - gro_e) ** 2 + (eskf_n - gro_n) ** 2 + (eskf_u - gro_u) ** 2)) - \
                     np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))
        elif self.reward_setting == 'RMSEadv_re': # 高程误差优势除2
            reward = np.sqrt(((eskf_e - gro_e) ** 2 + (eskf_n - gro_n) ** 2)) - np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2)) \
                      + (np.sqrt((eskf_u - gro_u) ** 2) - np.sqrt((rl_u - gro_u) ** 2)) * 5e-1

        error = np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))

        if step_print:
            print(
                f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                f'RL dist: [{np.abs(rl_e - gro_e):.2f}, {np.abs(rl_n - gro_n):.2f}, {np.abs(rl_u - gro_u):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1

        if done:
            obs = []
        else:
            obs = self._next_observation()

        return obs, reward, done, {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,'baseline': self.baseline,
                                   'error':error, 'tripid':self.tripIDnum}  # self.info#, {}# , 'data_truth_dic':data_truth_dic

    def SageHusaAKF(self,zs, us, processnoise): # RL for KF modified in 0303
        # Parameters
        sigma_mahalanobis = 100  # Mahalanobis distance for rejecting innovation,通过马氏距离判断当前观测值是否异常
        # position and velocity by imu
        acc_utc = self.raw_acc_utc[self.idx_acc]
        gnss_utc = self.datatime[self.current_step + (self.pos_num - 1)]  # current gnss utc
        gnss_utc_pre = self.datatime[self.current_step + (self.pos_num - 2)]  # previous gnss utc
        while abs(acc_utc - gnss_utc) > 11:
            acc_enu = self.raw_acc_enu[self.idx_acc, 1:4].reshape((3, 1))
            # acc_enu = acc_enu - np.array(self.dx[9:12].cpu()) # to tensor
            acc_enu = acc_enu - self.dx[9:12]
            # acc_enu_tensor = torch.from_numpy(acc_enu).double().to(self.device)  # to tensor
            self.v_imu = self.v_imu + acc_enu * (acc_utc - self.raw_acc_utc[self.idx_acc - 1]) * 1e-3  # 时间差 # to tensor
            self.p_imu = self.p_imu + self.v_imu * (acc_utc - self.raw_acc_utc[self.idx_acc - 1]) * 1e-3
            self.idx_acc = self.idx_acc + 1
            acc_utc = self.raw_acc_utc[self.idx_acc]

        # 观测信息提取
        z = zs.reshape((3, 1))  # 位置
        u = us.reshape((3, 1))  # 速度
        # TODO 可以直接把矩阵导进来，减少计算量
        # update RM RN
        ecef_wls = np.array(pm.enu2ecef(z[0], z[1], z[2], self.state_llh[0], self.state_llh[1], self.state_llh[2]))
        llh_wls = np.array(pm.ecef2geodetic(ecef_wls[0], ecef_wls[1], ecef_wls[2]))
        RM = calculate_RM(llh_wls[0])  # 计算曲率半径
        RN = calculate_RN(llh_wls[0])

        # 计算状态矩阵
        Fv = np.array([[0, acc_enu[2, 0], -acc_enu[1, 0]],
                       [-acc_enu[2, 0], 0, acc_enu[0, 0]],
                       [acc_enu[1, 0], -acc_enu[0, 0], 0]])
        lat_rad = math.radians(z[0, 0])  # 将纬度转为弧度
        tan_lat = math.tan(lat_rad)
        Fd = np.array([[0, 1 / (RM + z[2, 0]), 0],
                       [-1 / (RN + z[2, 0]), 0, 0],
                       [-tan_lat / (RN + z[2, 0]), 0, 0]])

        tol = (gnss_utc - gnss_utc_pre) * 1e-3  # 时间间隔
        F = np.block([[IM, IM * tol, ZM, ZM],
                      [ZM, IM, Fv * tol, IM * tol],
                      [ZM, Fd * tol, IM, ZM],
                      [ZM, ZM, ZM, IM]])

        # 这一部分不需要，因为其中的Q是不根据RL给出的数据
        # if self.conv_corr == 'conv_corr_1':  # Estimated WLS position covariance
        #     Q = sigma_v ** 1.250 * np.eye(dim_R)
        #     Q = Q + predict_Q_noise
        # elif self.conv_corr == 'conv_corr_2':
        #     Q = self.covv + predict_Q_noise
        # Q = np.where(Q < 0, 0.01, Q)
        # self.covv = Q
        # F = torch.from_numpy(F).to(self.device) # trasfer to tensor
        # 预测步骤
        self.dx = F @ self.dx + processnoise
        self.P = (F @ self.P) @ F.T + self.covv

        dz = np.vstack((self.p_imu - z, self.v_imu - u))  # imu和gnss计算的位置和速度误差
        y = dz - self.H @ self.dx
        # dz = np.vstack((np.array(self.p_imu.cpu()) - z, np.array(self.v_imu.cpu()) - u))  # imu和gnss计算的位置和速度误差 # to tensor
        # dz = torch.from_numpy(dz).to(self.device)
        # mahalanobis距离检测
        d = distance.mahalanobis(dz[:, 0], self.H @ self.dx, np.linalg.inv(self.P[:6, :6]))

        # d = distance.mahalanobis(dz[:, 0], self.H @ self.dx, np.linalg.inv(self.P[:6, :6]))
        # KF: Update step
        if d < sigma_mahalanobis:
            # 计算其中的卡尔曼增益
            # R = torch.from_numpy(R).double().to(self.device) # to tensor
            # y = dz - self.H @ self.dx
            S = (self.H @ self.P) @ self.H.T + self.covx
            K = (self.P @ self.H.T) @ np.linalg.inv(S)
            # S_inv = torch.from_numpy(np.linalg.inv(np.array(S.cpu()))).to(self.device)
            # K = (self.P @ self.H.T) @ S_inv # to tensor
            # 添加上Sage_husa 自适应部分： 更新过程噪声协方差
            innovation_outer = y @ y.T
            Q_hat = K @ innovation_outer @ K.T +self.P - F @ self.P_prev @ F.T
            self.covv = (1-self.forgetting_factor) * self.covv + self.forgetting_factor * Q_hat
            self.covv = (self.covv + self.covv.T) / 2
            # 状态更新
            self.dx = self.dx + K @ y
            self.P = (self.I - (K @ self.H)) @ self.P
            # 确保协方差阵定

            min_eig = np.min(np.real(np.linalg.eigvals(self.covv)))
            if min_eig<0:
                self.covv -= 10* min_eig * np.eye(*self.covv.shape)

        else:
            # If observation update is not available, increase covariance
            self.P += 10 ** 2 * self.covv
            # y = dz - self.H @ self.dx
        self.P_prev = self.P.copy()  # 留一个备份,作为下次迭代
        self.p_imu = self.p_imu - self.dx[0:3]
        self.v_imu = self.v_imu - self.dx[3:6]
        self.innovation = y.reshape((1, dim_R)) / 10  # normalization
        return self.p_imu[0, 0], self.p_imu[1, 0], self.p_imu[2, 0]

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

class SAC_GPSPosition_continuous_lospos_convQR_onlyQNallcorrect(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len, noise_scale_dic,
                 conv_corr, interrupt_dic=None, allcorrect=True, isloss = False):
    # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(SAC_GPSPosition_continuous_lospos_convQR_onlyQNallcorrect, self).__init__()
        self.max_visible_sat=13
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * CN0PRUEA_num)),
             'pos': spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float32),
             'Q_noise': spaces.Box(low=0, high=1, shape=(1, dim_Q), dtype=np.float32),
             'R_noise': spaces.Box(low=0, high=1, shape=(1, dim_R), dtype=np.float32),
             'innovation': spaces.Box(low=0, high=1, shape=(1, dim_R), dtype=np.float32),
             'kf_pred':spaces.Box(low=0, high=1, shape=(1, dim_Q), dtype=np.float32)})

        self.allcorrect = allcorrect  # correct in the end or in the obs
        if triptype == 'highway':
            self.tripIDlist = traj_highway
        elif triptype == 'urban':
            self.tripIDlist = traj_urban
        elif triptype == 'xiaomi':
            self.tripIDlist = traj_xiaomi
        elif triptype == 'xiaomiurban':
            self.tripIDlist = traj_xiaomiurban
        elif triptype == 'pixel567urban':
            self.tripIDlist = traj_pixel567urban
        elif triptype == 'smurban':
            self.tripIDlist = traj_smurban

        self.triptype = triptype
        self.traj_type = traj_type
        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.process_noise_scale = noise_scale_dic['process']
        self.measurement_noise_scale = noise_scale_dic['measurement']
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, dim_Q+dim_Q), dtype=np.float32) # modified for RLKF
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
        elif self.trajdata_sort == 'randint':
            sublist = self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]]
            random.shuffle(sublist)
            self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]] = sublist
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)
        # noise cov correction parameter
        self.conv_corr = conv_corr

        if interrupt_dic is not None:
            self.cnt_interrupt = 0
            self.interrupt_test = True
            self.start_ratio = interrupt_dic['start_ratio']
            self.interrupt_time = interrupt_dic['time']
        else:
            self.interrupt_test = False

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort == 'randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum = random.randint(self.trajdata_range[0], self.trajdata_range[1])
        elif self.trajdata_sort == 'sorted':
            self.tripIDnum = self.tripIDnum + 1
            if self.tripIDnum > self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        self.baseline = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature_all[self.tripIDlist[self.tripIDnum]].copy()
        self.raw_acc_enu = data_raw_acc_enu_dic[self.tripIDlist[self.tripIDnum]].values
        self.RM_HF_dic = data_RM_HF_dic[self.tripIDlist[self.tripIDnum]]

        self.datatime = self.baseline['UnixTimeMillis']
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UnixTimeMillis'].values) - 1, 'UnixTimeMillis']
        # normalize baseline
        # self.baseline['LatitudeDegrees_norm'] = (self.baseline['LatitudeDegrees']-lat_min)/(lat_max-lat_min)
        # self.baseline['LongitudeDegrees_norm'] = (self.baseline['LongitudeDegrees']-lon_min)/(lon_max-lon_min)
        # gen pred
        if self.baseline_mod == 'bl':
            self.baseline['E_RLpredict'] = self.baseline['E_bl']
            self.baseline['N_RLpredict'] = self.baseline['N_bl']
            self.baseline['U_RLpredict'] = self.baseline['U_bl']
        elif self.baseline_mod == 'wls':
            self.baseline['E_RLpredict'] = self.baseline['E_wls']
            self.baseline['N_RLpredict'] = self.baseline['N_wls']
            self.baseline['U_RLpredict'] = self.baseline['U_wls']
        elif self.baseline_mod == 'kf':
            self.baseline['E_RLpredict'] = self.baseline['E_kf']
            self.baseline['N_RLpredict'] = self.baseline['N_kf']
            self.baseline['U_RLpredict'] = self.baseline['U_kf']

        if gnss_trig:
            self.gnss = gnss_dic[self.tripIDlist[self.tripIDnum]]
        # Set the current step to a random point within the data frame
        # revise 1017: need the specific percent of the traj
            # setting for interrupt time
        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        if self.interrupt_test:
            self.interrupt_start_step = np.ceil(len(self.baseline) * (
                        self.traj_type[0] + self.start_ratio * (self.traj_type[-1] - self.traj_type[0])))
            self.interrupt_start_utc = self.datatime[self.interrupt_start_step + (self.pos_num - 1)]

        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['E_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['N_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['U_RLpredict']] = None

        # 协方差矩阵初始化
        self.covv = sigma_v ** 1.250 * np.eye(dim_Q)
        self.covx = sigma_x ** 1.250 * np.eye(dim_R)
        self.covx[2, 2] = self.covx[2, 2] * 2

        # initial kf state
        self.state_llh = self.baseline.loc[0,['LatitudeDegrees','LongitudeDegrees','AltitudeMeters']].values # 初始点经纬度，转坐标系用
        self.P = 5.0 ** 2 * np.eye(dim_Q)  # initial State covariance
        self.I = np.eye(dim_Q)
        self.H = np.hstack((np.eye(dim_R),np.zeros((dim_R,dim_Q-dim_R))))
        self.dx = np.zeros([dim_Q, 1]) # initial erro state
        # trasfer to tensor
        # self.P = 5.0 ** 2 * torch.eye(dim_Q).double().to(self.device)  # initial State covariance
        # self.I = torch.eye(dim_Q).double().to(self.device)
        # self.H = torch.from_numpy(np.hstack((np.eye(6),np.zeros((6,dim_Q-6))))).to(self.device)
        # self.dx = torch.zeros([dim_Q, 1]).double().to(self.device) # initial erro state

        self.p_imu = self.baseline.loc[self.current_step+(self.pos_num - 2),['E_RLpredict','N_RLpredict','U_RLpredict']].values.reshape((3, 1))
        self.v_imu = self.baseline.loc[self.current_step+(self.pos_num - 2),['VE_wls','VN_wls','VU_wls']].values.reshape((3, 1))
        # trasfer to tensor
        # self.p_imu = torch.from_numpy(self.p_imu).double().to(self.device)
        # self.v_imu = torch.from_numpy(self.v_imu).double().to(self.device)
        self.raw_acc_utc = self.raw_acc_enu[:,0] # 加速度的时间戳矩阵
        utc_init = self.datatime[self.current_step + (self.pos_num - 2)]
        self.idx_acc = np.argmin(np.abs(self.raw_acc_utc - utc_init)) # 初始 加速度索引

        self.innovation = np.ones([1,dim_R])
        self.kf_pred = np.ones([1, dim_Q])
        obs = self._next_observation()
        # must return in observation scale
        return obs  # self.tripIDnum#, obs#, {}

    def _normalize_pos(self, state):
        state[0] = state[0] / 10000
        state[1] = state[1] / 10000
        state[2] = state[2] / 100
        return state

    def _normalize_noise(self, obs_noise_R, obs_noise_Q):
        obs_noise_R = obs_noise_R / 100
        obs_noise_Q = obs_noise_Q / 10
        return obs_noise_R, obs_noise_Q

    def _normalize_los(self, gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        ## max normalize
        gnss[:, 1] = (gnss[:, 1]) / max(res_max, np.abs(res_min))
        gnss[:, 2] = (gnss[:, 2]) / max(losx_max, np.abs(losx_min))
        gnss[:, 3] = (gnss[:, 3]) / max(losy_max, np.abs(losy_min))
        gnss[:, 4] = (gnss[:, 4]) / max(losz_max, np.abs(losz_min))
        gnss[:, 5] = (gnss[:, 5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:, 6] = (gnss[:, 6]) / max(PRU_max, np.abs(PRU_min))
        gnss[:, 7] = (gnss[:, 7]) / max(AA_max, np.abs(AA_min))
        gnss[:, 8] = (gnss[:, 7]) / max(EA_max, np.abs(EA_min))
        # zero-score normalize
        # for i in range(gnss.shape[1]): # zero-score normalize
        #     if (i==0) or (i==gnss.shape[1]-1):
        #         continue
        #     mean,std = np.mean(gnss[:,i]),np.std(gnss[:,i])
        #     gnss[:,i] = (gnss[:,i]-mean) / std
        return gnss

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'E_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'N_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'U_RLpredict'].values])

        if self.baseline_mod == 'bl':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_bl']]],axis=1)
        elif self.baseline_mod == 'wls':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_wls']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_wls']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_wls']]],axis=1)
        elif self.baseline_mod == 'kf':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_kf']]],axis=1)

        obs = self._normalize_pos(obs)

        # gnss feature process
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features'].copy()
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        obs_feature = np.zeros([(self.max_visible_sat), CN0PRUEA_num])
        feature_index_list = [1, 2, 3, 4, 5, 6, 8]
        GPSL1_index = feature_tmp[:, 0] < 100
        feature_tmp = feature_tmp[:,feature_index_list]  # only GPSL1
        feature_tmp = feature_tmp[GPSL1_index,:]  # only GPSL1
        obs_feature[0:len(feature_tmp), :] = feature_tmp

        # noise cov feature process
        obs_Vnoise_pre = self.covv
        obs_Vnoise_cur = sigma_v ** 1.250 * np.eye(dim_Q)

        # obs_Vnoise_cur = sigma_v ** 1.250 * np.eye(dim_Q)
        if self.conv_corr == 'conv_corr_2': # 使用之前的不断迭代修改
            obs_noise_Q = np.diag(obs_Vnoise_pre)
            obs_noise_R = np.diag(self.covx)
        elif self.conv_corr == 'conv_corr_1': # 每次都是固定的修改
            obs_noise_Q = np.diag(obs_Vnoise_cur)
            obs_noise_R = np.diag(self.covx)

        # 新增的kf的状态预测
        kf_pred = self.kf_pred


        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'),
                   'gnss': obs_feature.reshape(1, CN0PRUEA_num * self.max_visible_sat, order='C'),
                   'Q_noise': obs_noise_Q.reshape(1, dim_Q, order='C'), 'R_noise': obs_noise_R.reshape(1, dim_R, order='C'),
                   'innovation': self.innovation,'kf_pred':kf_pred.reshape(1,-1) }

        # obs = obs.reshape(-1, 1) # + (traj_len-1)  + (traj_len-1)
        # obs=np.array([self.baseline.loc[self.current_step, 'LatitudeDegrees_norm'],self.baseline.loc[self.current_step, 'LongitudeDegrees_norm']])
        # obs=obs.reshape(2,1)
        # TODO latDeg lngDeg ... latDeg lngDeg
        return obs_all
    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def step(self, action): # modified in 20250521
        done = (self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) * self.traj_type[-1] - (
            self.pos_num) - outlayer_in_end_ecef)
        timestep = self.baseline.loc[self.current_step + (self.pos_num - 1), 'UnixTimeMillis']
        action = self.scale_action(action)

        action = np.reshape(action, [1, dim_Q + dim_Q])
        predict_pe = action[0, 0] * self.continuous_action_scale
        predict_pn = action[0, 1] * self.continuous_action_scale
        predict_pu = action[0, 2] * self.continuous_action_scale * 2
        predict_ve = action[0, 3] * self.continuous_action_scale * 1e-1
        predict_vn = action[0, 4] * self.continuous_action_scale * 1e-1
        predict_vu = action[0, 5] * self.continuous_action_scale * 1e-2
        predict_pitch = action[0, 6] * self.continuous_action_scale
        predict_roll = action[0, 7] * self.continuous_action_scale
        predict_yaw = action[0, 8] * self.continuous_action_scale
        predict_ae = action[0, 9] * self.continuous_action_scale * 1e-2
        predict_an = action[0, 10] * self.continuous_action_scale * 1e-2
        predict_au = action[0, 11] * self.continuous_action_scale * 1e-2
        processnoise = np.array([predict_pe,predict_pn,predict_pu,predict_ve,predict_vn,predict_vu,
                                 predict_pitch,predict_roll,predict_yaw,predict_ae,predict_an,predict_au]).reshape(12,1)

        # 协方差调整
        diag = [action[0, 12],action[0, 13],action[0, 14],action[0, 15],action[0, 16],action[0, 17],
                action[0, 18],action[0, 19],action[0, 20],action[0, 21],action[0, 22],action[0, 23]]
        predict_Q_noise = self.process_noise_scale * np.diagflat(diag)

        if self.baseline_mod == 'bl':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_bl']
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_bl']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_bl']
        elif self.baseline_mod == 'wls':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_wls']  # WLS结果作为观测
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_wls']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_wls']
        elif self.baseline_mod == 'kf':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_kf']
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_kf']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_kf']

        eskf_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_eskf_hfv2']
        eskf_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_eskf_hfv2']
        eskf_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_eskf_hfv2']

        v_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VE_wls']
        v_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VN_wls']
        v_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VU_wls']
        gro_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_gt']
        gro_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_gt']
        gro_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_gt']

        # modified for RLKF
        p_wls = np.array([obs_e, obs_n, obs_u])
        v_wls = np.array([v_e, v_n, v_u])
        rl_e, rl_n, rl_u, kf_pred = self.RL4ESKF_IMU(p_wls, v_wls, processnoise, predict_Q_noise)
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['E_RLpredict']] = rl_e
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['N_RLpredict']] = rl_n
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['U_RLpredict']] = rl_u

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['E_RLpredict']] = rl_e
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['N_RLpredict']] = rl_n
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['U_RLpredict']] = rl_u
        # reward function
        if self.reward_setting == 'RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2)) * 1e0  # *1e5
        # TODO 水平和高程是否需要加比例因子
        elif self.reward_setting == 'RMSEadv':
            reward = np.sqrt(((obs_e - gro_e) ** 2 + (obs_n - gro_n) ** 2 + (obs_u - gro_u) ** 2)) - \
                     np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))
        elif self.reward_setting == 'RMSEadv_eskf':
            reward = np.sqrt(((eskf_e - gro_e) ** 2 + (eskf_n - gro_n) ** 2 + (eskf_u - gro_u) ** 2)) - \
                     np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))
        elif self.reward_setting == 'RMSEadv_re': # 高程误差优势除2
            reward = np.sqrt(((eskf_e - gro_e) ** 2 + (eskf_n - gro_n) ** 2)) - np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2)) \
                      + (np.sqrt((eskf_u - gro_u) ** 2) - np.sqrt((rl_u - gro_u) ** 2)) * 5e-1

        error = np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))

        if step_print:
            print(
                f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                f'RL dist: [{np.abs(rl_e - gro_e):.2f}, {np.abs(rl_n - gro_n):.2f}, {np.abs(rl_u - gro_u):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()

        info = {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,'baseline': self.baseline,
                                   'error':error, 'tripid':self.tripIDnum, 'terminal_observation': obs if done else None }
        # info = {'tripIDnum': self.tripIDnum, 'current_step': self.current_step, 'baseline': self.baseline,
                # 'error': error, 'tripid': self.tripIDnum}
        return obs, reward, done, info

    def RL4ESKF_IMU(self,zs, us, processnoise, predict_Q_noise, policy=True): # RL for KF modified in 0303
        # Parameters
        sigma_mahalanobis = 100  # Mahalanobis distance for rejecting innovation
        # position and velocity by imu
        acc_utc = self.raw_acc_utc[self.idx_acc]
        gnss_utc = self.datatime[self.current_step + (self.pos_num - 1)]  # current gnss utc
        gnss_utc_pre = self.datatime[self.current_step + (self.pos_num - 2)]  # previous gnss utc
        while abs(acc_utc - gnss_utc) > 11:
            acc_enu = self.raw_acc_enu[self.idx_acc, 1:4].reshape((3, 1))
            # acc_enu = acc_enu - np.array(self.dx[9:12].cpu()) # to tensor
            acc_enu = acc_enu - self.dx[9:12]
            # acc_enu_tensor = torch.from_numpy(acc_enu).double().to(self.device)  # to tensor
            self.v_imu = self.v_imu + acc_enu * (acc_utc - self.raw_acc_utc[self.idx_acc - 1]) * 1e-3  # 时间差 # to tensor
            self.p_imu = self.p_imu + self.v_imu * (acc_utc - self.raw_acc_utc[self.idx_acc - 1]) * 1e-3
            self.idx_acc = self.idx_acc + 1
            acc_utc = self.raw_acc_utc[self.idx_acc]

        if self.interrupt_test:
            if self.interrupt_start_utc < gnss_utc:
                self.cnt_interrupt += 1
                if self.cnt_interrupt <= self.interrupt_time:
                    return self.p_imu[0, 0], self.p_imu[1, 0], self.p_imu[2, 0]
                elif self.cnt_interrupt == self.interrupt_time+1:
                    # 中断后，重新用gnss位置速度复位
                    print(f'interrupt end in {gnss_utc}')
                    self.p_imu = self.baseline.loc[self.current_step + (self.pos_num - 1), ['E_RLpredict', 'N_RLpredict','U_RLpredict']].values.reshape((3, 1))
                    self.v_imu = self.baseline.loc[self.current_step + (self.pos_num - 1), ['VE_wls', 'VN_wls', 'VU_wls']].values.reshape((3, 1))
                    return self.p_imu[0, 0], self.p_imu[1, 0], self.p_imu[2, 0]

        # 观测信息提取
        z = zs.reshape((3, 1))  # 位置
        u = us.reshape((3, 1))  # 速度
        # TODO 可以直接把矩阵导进来，减少计算量
        # update RM RN
        ecef_wls = np.array(pm.enu2ecef(z[0], z[1], z[2], self.state_llh[0], self.state_llh[1], self.state_llh[2]))
        llh_wls = np.array(pm.ecef2geodetic(ecef_wls[0], ecef_wls[1], ecef_wls[2]))
        RM = calculate_RM(llh_wls[0])  # 计算曲率半径
        RN = calculate_RN(llh_wls[0])

        # 计算状态矩阵
        Fv = np.array([[0, acc_enu[2, 0], -acc_enu[1, 0]],
                       [-acc_enu[2, 0], 0, acc_enu[0, 0]],
                       [acc_enu[1, 0], -acc_enu[0, 0], 0]])
        lat_rad = math.radians(z[0, 0])  # 将纬度转为弧度
        tan_lat = math.tan(lat_rad)
        Fd = np.array([[0, 1 / (RM + z[2, 0]), 0],
                       [-1 / (RN + z[2, 0]), 0, 0],
                       [-tan_lat / (RN + z[2, 0]), 0, 0]])

        tol = (gnss_utc - gnss_utc_pre) * 1e-3  # 时间间隔
        F = np.block([[IM, IM * tol, ZM, ZM],
                      [ZM, IM, Fv * tol, IM * tol],
                      [ZM, Fd * tol, IM, ZM],
                      [ZM, ZM, ZM, IM]])

        if self.conv_corr == 'conv_corr_1':  # Estimated WLS position covariance
            Q = sigma_v ** 1.250 * np.eye(dim_R)
            Q = Q + predict_Q_noise
        elif self.conv_corr == 'conv_corr_2':
            Q = self.covv + predict_Q_noise
        Q = np.where(Q < 0, 0.01, Q)
        self.covv = Q
        # F = torch.from_numpy(F).to(self.device) # trasfer to tensor
        self.dx = F @ self.dx + processnoise
        self.P = (F @ self.P) @ F.T + self.covv

        # 新添加上对于卡尔曼滤波的预测状态
        kf_pred = F @ self.dx

        # 新增，添加的
        def _stabilize_covariance(P):
            # 方法1：对称化+最小特征值约束
            P = 0.5 * (P + P.T)
            min_eigval = 1e-8
            eigvals = np.linalg.eigvals(P)
            if np.any(eigvals < min_eigval):
                P += np.diag(np.full(P.shape[0], min_eigval - np.min(eigvals)))
            return P

        self.P = _stabilize_covariance(self.P)

        # 新增的
        def _safe_mahalanobis(dz, H, P):
            try:
                # 方法1：正则化求逆
                inv_P = np.linalg.inv(P + 1e-6 * np.eye(P.shape[0]))
                return distance.mahalanobis(dz.flatten(), H, inv_P)
            except:
                # 退化情况：欧式距离
                return np.linalg.norm(dz - H @ self.dx)


        # self.P = (F @ self.P) @ F.T + torch.from_numpy(self.covv).to(self.device) # to tensor
        dz = np.vstack((self.p_imu - z, self.v_imu - u))  # imu和gnss计算的位置和速度误差
        # dz = np.vstack((np.array(self.p_imu.cpu()) - z, np.array(self.v_imu.cpu()) - u))  # imu和gnss计算的位置和速度误差 # to tensor
        # dz = torch.from_numpy(dz).to(self.device)
        ### 原本的
        # d = distance.mahalanobis(dz[:, 0], self.H @ self.dx, np.linalg.inv(self.P[:6, :6]))

        d = _safe_mahalanobis(dz[:, 0], self.H, self.P[:6, :6])
        # KF: Update step
        if d < sigma_mahalanobis:
            # R = torch.from_numpy(R).double().to(self.device) # to tensor
            y = dz - self.H @ self.dx
            S = (self.H @ self.P) @ self.H.T + self.covx
            K = (self.P @ self.H.T) @ np.linalg.inv(S)
            # S_inv = torch.from_numpy(np.linalg.inv(np.array(S.cpu()))).to(self.device)
            # K = (self.P @ self.H.T) @ S_inv # to tensor
            self.dx = self.dx + K @ y
            self.P = (self.I - (K @ self.H)) @ self.P
        else:
            # If observation update is not available, increase covariance
            self.P += 10 ** 2 * self.covv
            y = dz - self.H @ self.dx

        self.p_imu = self.p_imu - self.dx[0:3]
        self.v_imu = self.v_imu - self.dx[3:6]
        self.innovation = y.reshape((1, dim_R)) / 10  # normalization
        self.kf_pred = kf_pred
        return self.p_imu[0, 0], self.p_imu[1, 0], self.p_imu[2, 0], kf_pred


    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')

class DDPG_GPSPosition_continuous_lospos_convQR_onlyQNallcorrect(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self,trajdata_range, traj_type, triptype, continuous_action_scale, continuous_actionspace, reward_setting, trajdata_sort, baseline_mod, traj_len, noise_scale_dic,
                 conv_corr, interrupt_dic=None, allcorrect=True):
    # def __init__(self,trajdata_range, action_scale, discrete_actionspace, reward_setting, trajdata_sort, baseline_mod):
        super(DDPG_GPSPosition_continuous_lospos_convQR_onlyQNallcorrect, self).__init__()
        self.max_visible_sat=13
        self.pos_num = traj_len
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(self.max_visible_sat, 4), dtype=np.float)#shape=(2, 1)
        self.observation_space = spaces.Dict(
            {'gnss': spaces.Box(low=-1, high=1, shape=(1, self.max_visible_sat * CN0PRUEA_num)),
             'pos': spaces.Box(low=0, high=1, shape=(1, 3 * self.pos_num), dtype=np.float32),
             'Q_noise': spaces.Box(low=0, high=1, shape=(1, dim_Q), dtype=np.float32),
             'R_noise': spaces.Box(low=0, high=1, shape=(1, dim_R), dtype=np.float32)
             })

        self.allcorrect = allcorrect  # correct in the end or in the obs
        if triptype == 'highway':
            self.tripIDlist = traj_highway
        elif triptype == 'urban':
            self.tripIDlist = traj_urban
        elif triptype == 'xiaomi':
            self.tripIDlist = traj_xiaomi
        elif triptype == 'xiaomiurban':
            self.tripIDlist = traj_xiaomiurban
        elif triptype == 'pixel567urban':
            self.tripIDlist = traj_pixel567urban
        elif triptype == 'smurban':
            self.tripIDlist = traj_smurban

        self.triptype = triptype
        self.traj_type = traj_type
        # continuous action
        if trajdata_range=='full':
            self.trajdata_range = [0, len(self.tripIDlist)-1]
        else:
            self.trajdata_range = trajdata_range

        self.continuous_actionspace = continuous_actionspace
        self.process_noise_scale = noise_scale_dic['process']
        self.measurement_noise_scale = noise_scale_dic['measurement']
        self.continuous_action_scale = continuous_action_scale
        self.action_space = spaces.Box(low=continuous_actionspace[0], high=continuous_actionspace[1], shape=(1, dim_Q+dim_Q), dtype=np.float32) # modified for RLKF
        self.total_reward = 0
        self.reward_setting=reward_setting
        self.trajdata_sort=trajdata_sort
        self.baseline_mod=baseline_mod
        if self.trajdata_sort == 'sorted':
            self.tripIDnum = self.trajdata_range[0]
        elif self.trajdata_sort == 'randint':
            sublist = self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]]
            random.shuffle(sublist)
            self.tripIDlist[self.trajdata_range[0]:self.trajdata_range[1]] = sublist
            self.tripIDnum = self.trajdata_range[0]
            # continuous action
        # self.action_space = spaces.Box(low=-1, high=1, dtype=np.float)
        # noise cov correction parameter
        self.conv_corr = conv_corr
        if interrupt_dic is not None:
            self.cnt_interrupt = 0
            self.interrupt_test = True
            self.start_ratio = interrupt_dic['start_ratio']
            self.interrupt_time = interrupt_dic['time']
        else:
            self.interrupt_test = False


    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        if self.trajdata_sort == 'randint':
            # self.tripIDnum=random.randint(0,len(self.tripIDlist)-1)
            self.tripIDnum = random.randint(self.trajdata_range[0], self.trajdata_range[1])
        elif self.trajdata_sort == 'sorted':
            self.tripIDnum = self.tripIDnum + 1
            if self.tripIDnum > self.trajdata_range[1]:
                self.tripIDnum = self.trajdata_range[0]
        # self.tripIDnum=tripIDnum
        # self.info['tripIDnum']=self.tripIDnum
        self.baseline = data_truth_dic[self.tripIDlist[self.tripIDnum]].copy()
        self.losfeature = losfeature_all[self.tripIDlist[self.tripIDnum]].copy()
        self.raw_acc_enu = data_raw_acc_enu_dic[self.tripIDlist[self.tripIDnum]].values  ## 此处为IMU中对应的数据，包含加速度，每一个维度均为4，但是此文章需要对IMU中的数据进行机械编排
        self.RM_HF_dic = data_RM_HF_dic[self.tripIDlist[self.tripIDnum]]  ### 在GNSS失效时提供速度观测

        self.datatime = self.baseline['UnixTimeMillis']
        self.timeend = self.baseline.loc[len(self.baseline.loc[:, 'UnixTimeMillis'].values) - 1, 'UnixTimeMillis']
        # normalize baseline
        # self.baseline['LatitudeDegrees_norm'] = (self.baseline['LatitudeDegrees']-lat_min)/(lat_max-lat_min)
        # self.baseline['LongitudeDegrees_norm'] = (self.baseline['LongitudeDegrees']-lon_min)/(lon_max-lon_min)
        # gen pred
        if self.baseline_mod == 'bl':
            self.baseline['E_RLpredict'] = self.baseline['E_bl']
            self.baseline['N_RLpredict'] = self.baseline['N_bl']
            self.baseline['U_RLpredict'] = self.baseline['U_bl']
        elif self.baseline_mod == 'wls':
            self.baseline['E_RLpredict'] = self.baseline['E_wls']
            self.baseline['N_RLpredict'] = self.baseline['N_wls']
            self.baseline['U_RLpredict'] = self.baseline['U_wls']
        elif self.baseline_mod == 'kf':
            self.baseline['E_RLpredict'] = self.baseline['E_kf']
            self.baseline['N_RLpredict'] = self.baseline['N_kf']
            self.baseline['U_RLpredict'] = self.baseline['U_kf']

        if gnss_trig:
            self.gnss = gnss_dic[self.tripIDlist[self.tripIDnum]]
        # Set the current step to a random point within the data frame
        # revise 1017: need the specific percent of the traj
        self.current_step = np.ceil(len(self.baseline) * self.traj_type[0])  # self.current_step = 0
        # setting for interrupt time
        if self.interrupt_test:
            self.interrupt_start_step = np.ceil(
                len(self.baseline) * (self.traj_type[0] + self.start_ratio * (self.traj_type[-1] - self.traj_type[0])))
            self.interrupt_start_utc = self.datatime[self.interrupt_start_step + (self.pos_num - 1)]

        if self.traj_type[0] > 0:  # 只要剩下部分轨迹的定位结果
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['E_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['N_RLpredict']] = None
            data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[0:self.current_step + self.pos_num - 1, ['U_RLpredict']] = None

        # 协方差矩阵初始化
        self.covv = sigma_v ** 1.250 * np.eye(dim_Q)  ## 此为过程噪声协方差矩阵Q，也是要校正的对象,设置成12是合理的，就是速度，位置和姿态还有其他
        self.covx = sigma_x ** 1.250 * np.eye(dim_R)  ## 测量噪声R矩阵
        self.covx[2, 2] = self.covx[2, 2] * 2

        # initial kf state
        self.state_llh = self.baseline.loc[0,['LatitudeDegrees','LongitudeDegrees','AltitudeMeters']].values # 卡尔曼滤波的初始点经纬度，转坐标系用，但是这个初始点是否要更改呀？因为好像初始点不只是这些数据
        self.P = 5.0 ** 2 * np.eye(dim_Q)  # initial State covariance ，协方差矩阵
        self.I = np.eye(dim_Q)
        self.H = np.hstack((np.eye(dim_R),np.zeros((dim_R,dim_Q-dim_R))))  # 观测矩阵
        self.dx = np.zeros([dim_Q, 1]) # initial erro state
        # trasfer to tensor
        # self.P = 5.0 ** 2 * torch.eye(dim_Q).double().to(self.device)  # initial State covariance
        # self.I = torch.eye(dim_Q).double().to(self.device)
        # self.H = torch.from_numpy(np.hstack((np.eye(6),np.zeros((6,dim_Q-6))))).to(self.device)
        # self.dx = torch.zeros([dim_Q, 1]).double().to(self.device) # initial erro state

        self.p_imu = self.baseline.loc[self.current_step+(self.pos_num - 2),['E_RLpredict','N_RLpredict','U_RLpredict']].values.reshape((3, 1))
        self.v_imu = self.baseline.loc[self.current_step+(self.pos_num - 2),['VE_wls','VN_wls','VU_wls']].values.reshape((3, 1))
        # trasfer to tensor
        # self.p_imu = torch.from_numpy(self.p_imu).double().to(self.device)
        # self.v_imu = torch.from_numpy(self.v_imu).double().to(self.device)
        self.raw_acc_utc = self.raw_acc_enu[:,0] # 加速度的时间戳矩阵,来进行时间对齐
        utc_init = self.datatime[self.current_step + (self.pos_num - 2)]
        self.idx_acc = np.argmin(np.abs(self.raw_acc_utc - utc_init)) # 初始 加速度索引

        # self.innovation = np.ones([1,dim_R])
        obs = self._next_observation()
        # must return in observation scale
        return obs  # self.tripIDnum#, obs#, {}

    def _normalize_pos(self, state):
        state[0] = state[0] / 10000
        state[1] = state[1] / 10000
        state[2] = state[2] / 100
        return state

    def _normalize_noise(self, obs_noise_R, obs_noise_Q):
        obs_noise_R = obs_noise_R / 100
        obs_noise_Q = obs_noise_Q / 10
        return obs_noise_R, obs_noise_Q

    def _normalize_los(self, gnss):
        # gnss[:,0]=(gnss[:,0]-res_min) / (res_max - res_min)*2-1
        # gnss[:,1]=(gnss[:,1]-losx_min) / (losx_max - losx_min)*2-1
        # gnss[:,2]=(gnss[:,2]-losy_min) / (losy_max - losy_min)*2-1
        # gnss[:,3]=(gnss[:,3]-losz_min) / (losz_max - losz_min)*2-1
        ## max normalize
        gnss[:, 1] = (gnss[:, 1]) / max(res_max, np.abs(res_min))
        gnss[:, 2] = (gnss[:, 2]) / max(losx_max, np.abs(losx_min))
        gnss[:, 3] = (gnss[:, 3]) / max(losy_max, np.abs(losy_min))
        gnss[:, 4] = (gnss[:, 4]) / max(losz_max, np.abs(losz_min))
        gnss[:, 5] = (gnss[:, 5]) / max(CN0_max, np.abs(CN0_min))
        gnss[:, 6] = (gnss[:, 6]) / max(PRU_max, np.abs(PRU_min))
        gnss[:, 7] = (gnss[:, 7]) / max(AA_max, np.abs(AA_min))
        gnss[:, 8] = (gnss[:, 7]) / max(EA_max, np.abs(EA_min))
        # zero-score normalize
        # for i in range(gnss.shape[1]): # zero-score normalize
        #     if (i==0) or (i==gnss.shape[1]-1):
        #         continue
        #     mean,std = np.mean(gnss[:,i]),np.std(gnss[:,i])
        #     gnss[:,i] = (gnss[:,i]-mean) / std
        return gnss

    def _next_observation(self):
        obs = np.array([
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'E_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'N_RLpredict'].values,
            self.baseline.loc[self.current_step: self.current_step + (self.pos_num - 2), 'U_RLpredict'].values])

        if self.baseline_mod == 'bl':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_bl']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_bl']]],axis=1)
        elif self.baseline_mod == 'wls':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_wls']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_wls']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_wls']]],axis=1)
        elif self.baseline_mod == 'kf':
            obs = np.append(obs, [[self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_kf']],
                                  [self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_kf']]],axis=1)

        obs = self._normalize_pos(obs)

        # gnss feature process---GNSS的特征处理
        feature_tmp=self.losfeature[self.datatime[self.current_step + (self.pos_num-1)]]['features'].copy()
        # obs_feature = np.zeros([len(self.visible_sat), 4])
        feature_tmp = self._normalize_los(feature_tmp)
        obs_feature = np.zeros([(self.max_visible_sat), CN0PRUEA_num])
        feature_index_list = [1, 2, 3, 4, 5, 6, 8]
        GPSL1_index = feature_tmp[:, 0] < 100
        feature_tmp = feature_tmp[:,feature_index_list]  # only GPSL1
        feature_tmp = feature_tmp[GPSL1_index,:]  # only GPSL1
        obs_feature[0:len(feature_tmp), :] = feature_tmp

        # noise cov feature process
        obs_Vnoise_pre = self.covv
        obs_Vnoise_cur = sigma_v ** 1.250 * np.eye(dim_Q)

        # obs_Vnoise_cur = sigma_v ** 1.250 * np.eye(dim_Q)
        if self.conv_corr == 'conv_corr_2': # 使用之前的不断迭代修改
            obs_noise_Q = np.diag(obs_Vnoise_pre)
            obs_noise_R = np.diag(self.covx)
        elif self.conv_corr == 'conv_corr_1': # 每次都是固定的修改
            obs_noise_Q = np.diag(obs_Vnoise_cur)
            obs_noise_R = np.diag(self.covx)

        obs_all = {'pos': obs.reshape(1, 3 * self.pos_num, order='F'),
                   'gnss': obs_feature.reshape(1, CN0PRUEA_num * self.max_visible_sat, order='C'),
                   'Q_noise': obs_noise_Q.reshape(1, dim_Q, order='C'), 'R_noise': obs_noise_R.reshape(1, dim_R, order='C')
                   }

        # obs = obs.reshape(-1, 1) # + (traj_len-1)  + (traj_len-1)
        # obs=np.array([self.baseline.loc[self.current_step, 'LatitudeDegrees_norm'],self.baseline.loc[self.current_step, 'LongitudeDegrees_norm']])
        # obs=obs.reshape(2,1)
        # TODO latDeg lngDeg ... latDeg lngDeg
        return obs_all

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def step(self, action): # modified in 20250521
        done = (self.current_step >= len(self.baseline.loc[:, 'UnixTimeMillis'].values) * self.traj_type[-1] - (
            self.pos_num) - outlayer_in_end_ecef)
        timestep = self.baseline.loc[self.current_step + (self.pos_num - 1), 'UnixTimeMillis']
        # action for new prediction
        action = self.scale_action(action)
        action = np.reshape(action, [1, dim_Q + dim_Q])
        predict_pe = action[0, 0] * self.continuous_action_scale
        predict_pn = action[0, 1] * self.continuous_action_scale
        predict_pu = action[0, 2] * self.continuous_action_scale * 2
        predict_ve = action[0, 3] * self.continuous_action_scale * 1e-1
        predict_vn = action[0, 4] * self.continuous_action_scale * 1e-1
        predict_vu = action[0, 5] * self.continuous_action_scale * 1e-2
        predict_pitch = action[0, 6] * self.continuous_action_scale
        predict_roll = action[0, 7] * self.continuous_action_scale
        predict_yaw = action[0, 8] * self.continuous_action_scale
        predict_ae = action[0, 9] * self.continuous_action_scale * 1e-2
        predict_an = action[0, 10] * self.continuous_action_scale * 1e-2
        predict_au = action[0, 11] * self.continuous_action_scale * 1e-2
        processnoise = np.array([predict_pe,predict_pn,predict_pu,predict_ve,predict_vn,predict_vu,
                                 predict_pitch,predict_roll,predict_yaw,predict_ae,predict_an,predict_au]).reshape(12,1)

        # 协方差调整
        diag = [action[0, 12],action[0, 13],action[0, 14],action[0, 15],action[0, 16],action[0, 17],
                action[0, 18],action[0, 19],action[0, 20],action[0, 21],action[0, 22],action[0, 23]]
        predict_Q_noise = self.process_noise_scale * np.diagflat(diag)

        if self.baseline_mod == 'bl':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_bl']
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_bl']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_bl']
        elif self.baseline_mod == 'wls':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_wls']  # WLS结果作为观测
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_wls']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_wls']
        elif self.baseline_mod == 'kf':
            obs_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_kf']
            obs_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_kf']
            obs_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_kf']

        eskf_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_eskf_hfv2']
        eskf_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_eskf_hfv2']
        eskf_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_eskf_hfv2']

        v_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VE_wls']
        v_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VN_wls']
        v_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'VU_wls']
        gro_e = self.baseline.loc[self.current_step + (self.pos_num - 1), 'E_gt']
        gro_n = self.baseline.loc[self.current_step + (self.pos_num - 1), 'N_gt']
        gro_u = self.baseline.loc[self.current_step + (self.pos_num - 1), 'U_gt']

        # modified for RLKF
        p_wls = np.array([obs_e, obs_n, obs_u])
        v_wls = np.array([v_e, v_n, v_u])
        rl_e, rl_n, rl_u = self.RL4ESKF_IMU(p_wls, v_wls, processnoise, predict_Q_noise)
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['E_RLpredict']] = rl_e
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['N_RLpredict']] = rl_n
        self.baseline.loc[self.current_step + (self.pos_num - 1), ['U_RLpredict']] = rl_u

        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['E_RLpredict']] = rl_e
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['N_RLpredict']] = rl_n
        data_truth_dic[self.tripIDlist[self.tripIDnum]].loc[self.current_step + (self.pos_num - 1), ['U_RLpredict']] = rl_u
        # reward function
        if self.reward_setting == 'RMSE':
            # reward = np.mean(-((rl_lat - gro_lat) ** 2 + (rl_lng - gro_lng) ** 2))
            reward = -np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2)) * 1e0  # *1e5
        # TODO 水平和高程是否需要加比例因子
        elif self.reward_setting == 'RMSEadv':
            reward = np.sqrt(((obs_e - gro_e) ** 2 + (obs_n - gro_n) ** 2 + (obs_u - gro_u) ** 2)) - \
                     np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))
        elif self.reward_setting == 'RMSEadv_eskf':
            reward = np.sqrt(((eskf_e - gro_e) ** 2 + (eskf_n - gro_n) ** 2 + (eskf_u - gro_u) ** 2)) - \
                     np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))
        elif self.reward_setting == 'RMSEadv_re': # 高程误差优势除2
            reward = np.sqrt(((eskf_e - gro_e) ** 2 + (eskf_n - gro_n) ** 2)) - np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2)) \
                      + (np.sqrt((eskf_u - gro_u) ** 2) - np.sqrt((rl_u - gro_u) ** 2)) * 5e-1

        error = np.sqrt(((rl_e - gro_e) ** 2 + (rl_n - gro_n) ** 2 + (rl_u - gro_u) ** 2))

        if step_print:
            print(
                f'{self.tripIDlist[self.tripIDnum]}, Time {timestep}/{self.timeend} Baseline dist: [{np.abs(obs_x - gro_x):.2f}, {np.abs(obs_y - gro_y):.2f}, {np.abs(obs_z - gro_z):.2f}] m, '
                f'RL dist: [{np.abs(rl_e - gro_e):.2f}, {np.abs(rl_n - gro_n):.2f}, {np.abs(rl_u - gro_u):.2f}] m, RMSEadv: {reward:0.2e} m.')
        self.total_reward += reward
        # Execute one time step within the environment
        self.current_step += 1
        if done:
            obs = []
        else:
            obs = self._next_observation()

        info = {'tripIDnum': self.tripIDnum, 'current_step': self.current_step,'baseline': self.baseline,
                                   'error':error, 'tripid':self.tripIDnum, 'terminal_observation': obs if done else None }
        return obs, reward, done, info

    def RL4ESKF_IMU(self,zs, us, processnoise, predict_Q_noise, policy=True): # RL for KF modified in 0303
        # Parameters
        sigma_mahalanobis = 100  # Mahalanobis distance for rejecting innovation
        # position and velocity by imu
        acc_utc = self.raw_acc_utc[self.idx_acc]
        gnss_utc = self.datatime[self.current_step + (self.pos_num - 1)]  # current gnss utc
        gnss_utc_pre = self.datatime[self.current_step + (self.pos_num - 2)]  # previous gnss utc
        while abs(acc_utc - gnss_utc) > 11:
            acc_enu = self.raw_acc_enu[self.idx_acc, 1:4].reshape((3, 1))
            # acc_enu = acc_enu - np.array(self.dx[9:12].cpu()) # to tensor
            acc_enu = acc_enu - self.dx[9:12]
            # acc_enu_tensor = torch.from_numpy(acc_enu).double().to(self.device)  # to tensor
            self.v_imu = self.v_imu + acc_enu * (acc_utc - self.raw_acc_utc[self.idx_acc - 1]) * 1e-3  # 时间差 # to tensor
            self.p_imu = self.p_imu + self.v_imu * (acc_utc - self.raw_acc_utc[self.idx_acc - 1]) * 1e-3
            self.idx_acc = self.idx_acc + 1
            acc_utc = self.raw_acc_utc[self.idx_acc]

        # don't process eskf if interrupt
        if self.interrupt_test:
            if self.interrupt_start_utc < gnss_utc:
                self.cnt_interrupt += 1
                if self.cnt_interrupt <= self.interrupt_time:
                    return self.p_imu[0, 0], self.p_imu[1, 0], self.p_imu[2, 0]
                elif self.cnt_interrupt == self.interrupt_time + 1:
                    # 中断后，重新用gnss位置速度复位
                    print(f'interrupt end in {gnss_utc}')
                    self.p_imu = self.baseline.loc[
                        self.current_step + (self.pos_num - 1), ['E_kf', 'N_kf', 'U_kf']].values.reshape((3, 1))
                    self.v_imu = self.baseline.loc[
                        self.current_step + (self.pos_num - 1), ['VE_wls', 'VN_wls', 'VU_wls']].values.reshape(
                        (3, 1))
                    return self.p_imu[0, 0], self.p_imu[1, 0], self.p_imu[2, 0]

        # 观测信息提取
        z = zs.reshape((3, 1))  # 位置
        u = us.reshape((3, 1))  # 速度
        # TODO 可以直接把矩阵导进来，减少计算量
        # update RM RN
        ecef_wls = np.array(pm.enu2ecef(z[0], z[1], z[2], self.state_llh[0], self.state_llh[1], self.state_llh[2]))
        llh_wls = np.array(pm.ecef2geodetic(ecef_wls[0], ecef_wls[1], ecef_wls[2]))
        RM = calculate_RM(llh_wls[0])  # 计算曲率半径
        RN = calculate_RN(llh_wls[0])

        # 计算状态矩阵
        Fv = np.array([[0, acc_enu[2, 0], -acc_enu[1, 0]],
                       [-acc_enu[2, 0], 0, acc_enu[0, 0]],
                       [acc_enu[1, 0], -acc_enu[0, 0], 0]])
        lat_rad = math.radians(z[0, 0])  # 将纬度转为弧度
        tan_lat = math.tan(lat_rad)
        Fd = np.array([[0, 1 / (RM + z[2, 0]), 0],
                       [-1 / (RN + z[2, 0]), 0, 0],
                       [-tan_lat / (RN + z[2, 0]), 0, 0]])

        tol = (gnss_utc - gnss_utc_pre) * 1e-3  # 时间间隔
        F = np.block([[IM, IM * tol, ZM, ZM],
                      [ZM, IM, Fv * tol, IM * tol],
                      [ZM, Fd * tol, IM, ZM],
                      [ZM, ZM, ZM, IM]])

        if self.conv_corr == 'conv_corr_1':  # Estimated WLS position covariance
            Q = sigma_v ** 1.250 * np.eye(dim_R)
            Q = Q + predict_Q_noise
        elif self.conv_corr == 'conv_corr_2':
            Q = self.covv + predict_Q_noise
        Q = np.where(Q < 0, 0.01, Q)
        self.covv = Q
        # F = torch.from_numpy(F).to(self.device) # trasfer to tensor
        self.dx = F @ self.dx + processnoise
        self.P = (F @ self.P) @ F.T + self.covv
        # self.P = (F @ self.P) @ F.T + torch.from_numpy(self.covv).to(self.device) # to tensor

        dz = np.vstack((self.p_imu - z, self.v_imu - u))  # imu和gnss计算的位置和速度误差
        # dz = np.vstack((np.array(self.p_imu.cpu()) - z, np.array(self.v_imu.cpu()) - u))  # imu和gnss计算的位置和速度误差 # to tensor
        # dz = torch.from_numpy(dz).to(self.device)
        d = distance.mahalanobis(dz[:, 0], self.H @ self.dx, np.linalg.inv(self.P[:6, :6]))
        # KF: Update step
        if d < sigma_mahalanobis:
            # R = torch.from_numpy(R).double().to(self.device) # to tensor
            y = dz - self.H @ self.dx
            S = (self.H @ self.P) @ self.H.T + self.covx
            K = (self.P @ self.H.T) @ np.linalg.inv(S)
            # S_inv = torch.from_numpy(np.linalg.inv(np.array(S.cpu()))).to(self.device)
            # K = (self.P @ self.H.T) @ S_inv # to tensor
            self.dx = self.dx + K @ y
            self.P = (self.I - (K @ self.H)) @ self.P
        else:
            # If observation update is not available, increase covariance
            self.P += 10 ** 2 * self.covv
            y = dz - self.H @ self.dx

        self.p_imu = self.p_imu - self.dx[0:3]
        self.v_imu = self.v_imu - self.dx[3:6]
        self.innovation = y.reshape((1, dim_R)) / 10  # normalization
        return self.p_imu[0, 0], self.p_imu[1, 0], self.p_imu[2, 0]

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        #  print(f'reward: {self.reward}')
        print(f'total_reward: {self.total_reward}')


def calculate_RM(lat_deg):
    lat_rad = math.radians(lat_deg)
    sin2 = (math.sin(lat_rad)) ** 2
    RM = RE_WGS84 / math.sqrt(1 - E_2 * sin2)
    return RM

def calculate_RN(lat_deg):
    lat_rad = math.radians(lat_deg)  # 将纬度转为弧度
    sin_lat = math.sin(lat_rad)  # 计算 sin(φ)
    denominator = (1 - E_2 * sin_lat ** 2) ** (3 / 2)  # 分母项 (1 - e²sin²φ)^(3/2)
    RN = a * (1 - E_2) / denominator  # 最终公式
    return RN


import gym
from gym import spaces
import random
import numpy as np
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import glob as gl
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import simdkalman
import sys
sys.path.append("..")
from envRLKF.env_funcs_multic_AKF_IMU import LOSPRRprocess
import src.gnss_lib.coordinates as coord

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#导入数据
# 10.23.18.26
dir_path = '/mnt/sdb/home/tangjh/GSDC2023/'
data_path = '/mnt/sdb/home/tangjh/GSDC2023/sdc2023/'

data_truth_dic={}
gnss_dic={}
losfeature={}
raw_imu_all_dic = {}
Rotation_Matrix_HF_dic = {}

try:
    with open(dir_path+'env/raw_gnss_multic.pkl', "rb") as file:
        gnss_dic = pickle.load(file)
    file.close()
    # with open(dir_path+'env/raw_baseline_multic.pkl', "rb") as file:
    #     data_truth_dic = pickle.load(file)
    # file.close()
except:
    filenonexist=1
tripIDlist=[]
# Ground truth: LatitudeDegrees_truth
# KF: LatitudeDegrees
# Baseline(WLS): LatitudeDegrees_bl
# Robust WLS: LatitudeDegrees_wls
train_columns = ['UnixTimeMillis','LatitudeDegrees','LongitudeDegrees','AltitudeMeters',
                 'ecefX','ecefY','ecefZ']
merge_columns = ['UnixTimeMillis']

"""
# check minmax value
data_truth_list =[]
for i, dirname in enumerate(tqdm(sorted(gl.glob(f'{data_path}/train/*/*/')))):
    drive, phone = dirname.split('/')[-3:-1]
    tripID = f'{drive}/{phone}'
    if (not tripID in tripIDskiplist):
        # Read data
        truth_df = pd.read_csv(f'{dirname}/ground_truth.csv')
        baseline_df=pd.read_csv(f'{dirname}/baseline_KF.csv')
        pd_train = truth_df[train_columns].merge(baseline_df, on=merge_columns, suffixes=("_truth", ""))
        data_truth_list.append(pd_train)
data_train_truth = pd.concat(data_truth_list, ignore_index=True)
ground_truth_colums = ['LatitudeDegrees_truth','LongitudeDegrees_truth']
minmax_=[]
for col in ground_truth_colums:
    max_ = np.max(data_train_truth[col])
    min_ = np.min(data_train_truth[col])
    minmax_.append([max_,min_])
"""
# Loop for each trip
truth_min_X=[]
truth_min_Y=[]
truth_min_Z=[]
truth_max_X=[]
truth_max_Y=[]
truth_max_Z=[]
kf_min_e=[]
kf_min_n=[]
kf_min_u=[]
kf_max_e=[]
kf_max_n=[]
kf_max_u=[]
truth_min_lat=[]
truth_min_lon=[]
truth_min_alt=[]
truth_max_lat=[]
truth_max_lon=[]
truth_max_alt=[]

biastrig=1
pd_sat_info = pd.DataFrame(['stepnum','satnum_min','satnum_max','GPS_L1_min','GPS_L1_max','GLO_G1_min','GLO_G1_max',
                            'BDS_B1I_min','BDS_B1I_max','BDS_B1C_min','BDS_B1C_max','BDS_B2A_min','BDS_B2A_max',
                            'GAL_E1_min','GAL_E1_max','GPS_L5_min','GPS_L5_max',
                            'GAL_E5A_min','GAL_E5A_max','QZS_J1_min','QZS_J1_max','QZS_J5_min','QZS_J5_max',
                            'miss obs','less_4_satnum'], columns=['tripID'])

noobstraj = ['2020-08-04-00-20-us-ca-sb-mtv-101','2020-08-13-21-42-us-ca-mtv-sf-280/pixel5'] # 缺较多卫星观测
err_pd = pd.read_csv(f'{dir_path}/sdc2023/record_consecutive 50_ID.csv')
errortrajlist = err_pd['errorID'].values.tolist() # 缺较多IMU测量
result_pd = pd.read_csv(f'{dir_path}/sdc2023/records_train_kf_cal_True_g_bias_True_all_2023.csv')
tripIDskiplist = result_pd[result_pd['2Derror_imueskf_hfv3_avg']>20]['tripId'].tolist() # 误差太大

id_call=True
processing_gnss = False # 是否需要处理gnss特征
starting_flag = False
starting_traj = None #'2020-07-17-23-13-us-ca-sf-mtv-280/pixel4' # 从某条轨迹开始处理
baseline = 'wls_igst' # kf_realtime wls_igst

for i, dirname in enumerate(tqdm(sorted(gl.glob(f'{data_path}/train/*/*/')))):
    drive, phone = dirname.split('/')[-3:-1]
    tripID = f'{drive}/{phone}'
    # if '2021' in drive:
    # ignoring trajs without altitude
    if starting_traj != None:
        if tripID == starting_traj:
            starting_flag = True
    else:
        starting_flag = True
    if starting_flag:
        if tripID not in errortrajlist:
            if tripID not in tripIDskiplist:
                print(tripID)
                # Read data
                truth_df = pd.read_csv(f'{dirname}/ground_truth.csv')
                baseline_df=pd.read_csv(f'{dirname}/baseline_eskf_kf_cal_True_g_bias_True.csv')
                gnss_df_raw = pd.read_csv(f'{dirname}/device_gnss.csv')
                raw_imu_all = pd.read_csv(f'{dirname}/Raw_measurement_imu_ALL.csv')
                with open(dirname + '/Rotation_Matrix_HF.pkl', "rb") as file:
                    Rotation_Matrix_HF = pickle.load(file)
                file.close()

                LOSPRR=LOSPRRprocess(gnss_df_raw,truth_df,tripID,dir_path,baseline_df) # gnss processing
                gnss_df,sat_info = LOSPRR.LOSPRRprocesses()
                pd_sat_info.loc[:,tripID] = pd.DataFrame({tripID:sat_info})
                gnss_dic[tripID]=gnss_df
                try:
                    pd_train = gnss_df[['utcTimeMillis','LatitudeDegrees','LongitudeDegrees','AltitudeMeters',
                                        'ecefX','ecefY','ecefZ','VE_wls', 'VN_wls','VU_wls','Acc_e',
                                        'Acc_n','Acc_u','E_gt','N_gt','U_gt','E_bl','N_bl','U_bl',
                                        'E_wls','N_wls','U_wls','E_kf','N_kf','U_kf','E_eskf_hfv2',
                                        'N_eskf_hfv2','U_eskf_hfv2','E_eskf_hfall',
                                        'N_eskf_hfall','U_eskf_hfall']].copy()
                except:
                    pass
                pd_train.drop_duplicates(inplace=True)
                pd_train.rename(columns={'utcTimeMillis': 'UnixTimeMillis'}, inplace=True)
                pd_train_index = pd_train.reset_index(drop=True)
                # pd_train = truth_df[train_columns].merge(baseline_df, on=merge_columns, suffixes=("_truth", ""))
                # ecefxyz_kf_igst = pd_train[
                #     ['XEcefMeters_kf_igst', 'YEcefMeters_kf_igst', 'ZEcefMeters_kf_igst']].to_numpy()
                # lla_kf_igst = coord.ecef2geodetic(ecefxyz_kf_igst)
                # pd_train.loc[:, 'AltitudeMeters_kf_igst'] = lla_kf_igst[:,2]
                data_truth_dic[tripID] = pd_train_index
                raw_imu_all_dic[tripID] = raw_imu_all
                Rotation_Matrix_HF_dic[tripID] = Rotation_Matrix_HF

                if processing_gnss:
                    if baseline == 'wls_igst':
                        featureall, sat_summary_multicCN0, Complementaryfeatureall =LOSPRR.getitemECEFCN0AA_RLKF(1,biastrig,gnss_df,id_call)
                    elif baseline == 'kf_realtime':
                        featureall, sat_summary_multicCN0 = LOSPRR.getitemECEFCN0AA_KF_realtime(1, biastrig, gnss_df, id_call)
                    try:
                        sat_summary_multicCN0AA_all.loc[:, tripID] = pd.DataFrame({tripID:sat_summary_multicCN0['Nums']})
                        # sat_summary_multicCN0AA_all = pd.concat([sat_summary_multicCN0AA_all, pd.DataFrame({tripID:sat_summary_multicCN0['Nums']})], sort=False)
                    except:
                        sat_summary_multicCN0AA_all=sat_summary_multicCN0.rename(columns={'Nums':tripID})
                    losfeature[tripID]=featureall

                tripIDlist.append(tripID)
                # truth XYZ min max baselin XYZ min max
                truth_min_X.append(np.min(pd_train['ecefX'].to_numpy()))
                truth_min_Y.append(np.min(pd_train['ecefY'].to_numpy()))
                truth_min_Z.append(np.min(pd_train['ecefZ'].to_numpy()))
                truth_max_X.append(np.max(pd_train['ecefX'].to_numpy()))
                truth_max_Y.append(np.max(pd_train['ecefY'].to_numpy()))
                truth_max_Z.append(np.max(pd_train['ecefZ'].to_numpy()))

                kf_min_e.append(np.min(baseline_df['E_kf'].to_numpy()))
                kf_min_n.append(np.min(baseline_df['N_kf'].to_numpy()))
                kf_min_u.append(np.min(baseline_df['U_kf'].to_numpy()))
                kf_max_e.append(np.max(baseline_df['E_kf'].to_numpy()))
                kf_max_n.append(np.max(baseline_df['N_kf'].to_numpy()))
                kf_max_u.append(np.max(baseline_df['U_kf'].to_numpy()))

                truth_min_lat.append(np.min(pd_train['LatitudeDegrees'].to_numpy()))
                truth_min_lon.append(np.min(pd_train['LongitudeDegrees'].to_numpy()))
                truth_min_alt.append(np.min(pd_train['AltitudeMeters'].to_numpy()))
                truth_max_lat.append(np.max(pd_train['LatitudeDegrees'].to_numpy()))
                truth_max_lon.append(np.max(pd_train['LongitudeDegrees'].to_numpy()))
                truth_max_alt.append(np.max(pd_train['AltitudeMeters'].to_numpy()))

#%% raw data saving
with open(dir_path + 'envRLKF/raw_baseline_IMU_multic_enu_ALL.pkl', 'wb') as value_file:
    pickle.dump(data_truth_dic, value_file, True)
value_file.close()
# with open(dir_path + 'envRLKF/raw_gnss_multic_lla.pkl', 'wb') as value_file:
#     pickle.dump(gnss_dic, value_file, True)
# value_file.close()

with open(dir_path + 'envRLKF/raw_tripID_IMU_ALL.pkl', 'wb') as value_file:
    pickle.dump(tripIDlist, value_file, True)
value_file.close()
### record for AKF-RL
# record conv velocity and position
with open(dir_path + 'envRLKF/Raw_measurement_imu_ALL.pkl', 'wb') as value_file:
    pickle.dump(raw_imu_all_dic, value_file, True)
value_file.close()
with open(dir_path + 'envRLKF/raw_rotation_matrix_dic_ALL.pkl', 'wb') as value_file:
    pickle.dump(Rotation_Matrix_HF_dic, value_file, True)
value_file.close()

tripID_df=pd.DataFrame(tripIDlist, columns=['tripId'])

tripID_df['ecefX_min']=truth_min_X
tripID_df['ecefY_min']=truth_min_Y
tripID_df['ecefZ_min']=truth_min_Z
tripID_df['ecefX_max']=truth_max_X
tripID_df['ecefY_max']=truth_max_Y
tripID_df['ecefZ_max']=truth_max_Z

tripID_df['e_min_kf']=kf_min_e
tripID_df['n_min_kf']=kf_min_n
tripID_df['u_min_kf']=kf_min_u
tripID_df['e_max_kf']=kf_max_e
tripID_df['n_max_kf']=kf_max_n
tripID_df['u_max_kf']=kf_max_u

tripID_df['lat_min']=truth_min_lat
tripID_df['lon_min']=truth_min_lon
tripID_df['alt_min']=truth_min_alt
tripID_df['lat_max']=truth_max_lat
tripID_df['lon_max']=truth_max_lon
tripID_df['alt_max']=truth_max_alt

# pd_sat_info.to_csv(dir_path + 'envRLKF/raw_sat_information.csv', index=True)
tripID_df.to_csv(dir_path + 'envRLKF/raw_tripID_IMU_enu_ALL.csv', index=True)
if processing_gnss:
    sat_summary_multicCN0AA_all.to_csv(dir_path + 'envRLKF/raw_satnum_multicCN0AA_lla_50.csv', index=True)
    with open(dir_path + f'envRLKF/processed_features_{baseline}_multicCN0AA_lla_cid.pkl', 'wb') as value_file:
        pickle.dump(losfeature, value_file, True)
    value_file.close()
    with open(dir_path + 'envRLKF/Complementaryfeature_multic_RLAKF.pkl', 'wb') as value_file:
        pickle.dump(Complementaryfeatureall, value_file, True)  # 补充观测量
    value_file.close()

# for i, dirname in enumerate(tqdm(sorted(gl.glob(f'{data_path}/train/*/*/')))):
#     drive, phone = dirname.split('/')[-3:-1]
#     tripID = f'{drive}/{phone}'
#     if (not tripID in tripIDskiplist):
#         # Read data
#         truth_df = pd.read_csv(f'{dirname}/ground_truth.csv')
#         baseline_df=pd.read_csv(f'{dirname}/baseline_KF.csv')
#         pd_train = truth_df[train_columns].merge(data_train, on=merge_columns, suffixes=("_truth", ""))

print('finish')
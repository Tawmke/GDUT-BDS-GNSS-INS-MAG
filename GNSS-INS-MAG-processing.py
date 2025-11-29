import numpy as np
import pandas as pd
import sys
import pymap3d as pm
import pymap3d.vincenty as pmv
import matplotlib.pyplot as plt
import glob as gl
import scipy.optimize
from tqdm.auto import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial import distance
import os
from funcs.funcs_KF import *
from pathlib import Path
from scipy import signal
from scipy.signal import butter, filtfilt, buttord
from scipy.signal import medfilt
import src.gnss_lib.coordinates as coord
import warnings
import pickle
warnings.filterwarnings("ignore")  # 忽略所有警告
testing = False
processing = True
# Satellite selection using carrier frequency error, elevation angle, and C/N0

######################  Train data  ########################
# Target course/phone
"""
2023-09-07-19-33-us-ca/pixel6pro
"""
if testing:
    path = '/mnt/sdb/home/tangjh/GSDC2023/sdc2023/train/2023-09-07-19-33-us-ca/pixel6pro'
    drive, phone = path.split('/')[-2:]

    # Read data
    gnss_df = pd.read_csv(f'{path}/device_gnss.csv')  # GNSS data
    gt_df = pd.read_csv(f'{path}/ground_truth.csv')  # ground truth
    imu = pd.read_csv(f'{path}/device_imu.csv')
    mag_bias = pd.read_csv(f'/mnt/sdb/home/tangjh/GSDC2023/sdc2023/result/mag_bias/mag_bias.csv')
    acc_bias = pd.read_csv(f'/mnt/sdb/home/tangjh/GSDC2023/sdc2023/result/acc_bias/acc_bias.csv')
    acc = imu.query(f"MessageType=='UncalAccel'").reset_index(drop = True)
    gyr = imu.query(f"MessageType=='UncalGyro'").reset_index(drop = True)
    mag = imu.query(f"MessageType=='UncalMag'").reset_index(drop = True)

    # -------------------------零偏校正--------------------------
    cal = 1  # 是否校正磁力计零偏
    if cal:
        diy = 1  # 是否使用自己估计的零偏值进行校正
        if diy:
            # 磁力计恒偏校正
            mag['MeasurementX'] = mag['MeasurementX'] + mag_bias.loc[mag_bias['tripID'] == f'{drive}/{phone}', 'bias_x'].values
            mag['MeasurementY'] = mag['MeasurementY'] - mag['BiasY']
            mag['MeasurementZ'] = mag['MeasurementZ']
            mag['Y'] = mag['MeasurementY']
            mag['MeasurementY'] = -mag['MeasurementZ'] + mag_bias.loc[mag_bias['tripID'] == f'{drive}/{phone}', 'bias_y'].values - 100
            mag['MeasurementZ'] = mag['Y']
            # 加速度计恒偏校正
            acc['MeasurementX'] = acc['MeasurementX'] + acc_bias.loc[acc_bias['tripID'] == f'{drive}/{phone}', 'bias_x'].values
            acc['MeasurementY'] = acc['MeasurementY']
            acc['MeasurementZ'] = acc['MeasurementZ']
            acc['Y'] = acc['MeasurementY']
            acc['MeasurementY'] = -acc['MeasurementZ'] - 2 + acc_bias.loc[acc_bias['tripID'] == f'{drive}/{phone}', 'bias_y'].values
            acc['MeasurementZ'] = acc['Y'] - 9.81 + acc_bias.loc[acc_bias['tripID'] == f'{drive}/{phone}', 'bias_z'].values
        if not diy:
            mag['MeasurementX'] = mag['MeasurementX'] - mag['BiasX']
            mag['MeasurementY'] = mag['MeasurementY'] - mag['BiasY']
            mag['MeasurementZ'] = mag['MeasurementZ'] - mag['BiasZ']
            mag['Y'] = mag['MeasurementY']
            mag['MeasurementY'] = -mag['MeasurementZ']
            mag['MeasurementZ'] = mag['Y']
    if not cal:
        mag['Y'] = mag['MeasurementY']
        mag['MeasurementY'] = -mag['MeasurementZ']
        mag['MeasurementZ'] = mag['Y']

    # -------------------------检查数据--------------------------
    pd.set_option('display.max_columns', None)  # 显示所有行
    acc_ = acc[['utcTimeMillis', 'MeasurementX', 'MeasurementY', 'MeasurementZ']]
    acc_['acc_index'] = acc_.index
    print(f'acc length: {len(acc_)}')
    gyr_ = gyr[['utcTimeMillis', 'MeasurementX', 'MeasurementY', 'MeasurementZ']]
    gyr_['gyr_index'] = gyr_.index
    print(f'gyr length: {len(gyr_)}')
    mag_ = mag[['utcTimeMillis', 'MeasurementX', 'MeasurementY', 'MeasurementZ']]
    mag_['mag_index'] = mag_.index
    print(f'mag length: {len(mag_)}')
    pos = gnss_df[['utcTimeMillis', 'WlsPositionXEcefMeters',
                      'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].drop_duplicates('utcTimeMillis', keep='first')

    # -------------------------数据拼接--------------------------
    acc_.rename(columns={'MeasurementX': 'MeasurementX_acc', 'MeasurementY': 'MeasurementY_acc',
                         'MeasurementZ': 'MeasurementZ_acc'}, inplace=True)
    gyr_.rename(columns={'MeasurementX': 'MeasurementX_gyr', 'MeasurementY': 'MeasurementY_gyr',
                         'MeasurementZ': 'MeasurementZ_gyr'}, inplace=True)
    mag_.rename(columns={'MeasurementX': 'MeasurementX_mag', 'MeasurementY': 'MeasurementY_mag',
                         'MeasurementZ': 'MeasurementZ_mag'}, inplace=True)
    df_temp = pd.merge_asof(
        pos,
        acc_,
        left_on=['utcTimeMillis'],
        right_on=['utcTimeMillis'],
        direction='nearest',
        tolerance=20)

    print(f'mag length: {len(df_temp)}')

    df_temp = pd.merge_asof(
        df_temp,
        gyr_,
        left_on=['utcTimeMillis'],
        right_on=['utcTimeMillis'],
        direction='nearest',
        tolerance=10)

    df_temp = pd.merge_asof(
        df_temp,
        mag_,
        left_on=['utcTimeMillis'],
        right_on=['utcTimeMillis'],
        direction='nearest',
        tolerance=10)

    df_temp = pd.merge_asof(
        gt_df,
        df_temp,
        left_on=['UnixTimeMillis'],
        right_on=['utcTimeMillis'],
        direction='nearest',
        tolerance=5)

    columns_to_drop = ['SpeedAccuracyMps', 'BearingAccuracyDegrees', 'elapsedRealtimeNanos', 'VerticalAccuracyMeters']
    df_temp = df_temp.drop(columns=columns_to_drop)

    # 检查数据有无空缺值
    nan_acc = df_temp['MeasurementX_acc'].isnull().sum()
    nan_gyr = df_temp['MeasurementX_gyr'].isnull().sum()
    nan_mag = df_temp['MeasurementX_mag'].isnull().sum()
    print(f'acc nan number:{nan_acc},gyr nan number:{nan_gyr},mag nan number:{nan_mag}')
    # 做了合并之后填充nan值
    df_temp = compensate(df_temp,['MeasurementX_acc','MeasurementY_acc','MeasurementZ_acc'],max_consecutive)
    df_temp = compensate(df_temp,['MeasurementX_mag','MeasurementY_mag','MeasurementZ_mag'],max_consecutive)

    # 使用 np.where() 函数找到空缺值的位置
    # rows, cols = np.where(pd.isnull(df_temp))
    # # 打印出空缺值的位置
    # for row, col in zip(rows, cols):
    #     print(f'空缺值在第 {row + 1} 行，第 {col + 1} 列')
    #     if row > 0:
    #         df_temp.iat[row, col] = df_temp.iat[row - 1, col]

    # -------------------------计算偏航角--------------------------
    mag_x = df_temp['MeasurementX_mag'].to_numpy()
    mag_y = df_temp['MeasurementY_mag'].to_numpy()
    mag_z = df_temp['MeasurementZ_mag'].to_numpy()
    # 计算航向角（以度为单位），以北方向为0度，顺时针数，范围是0~359.99
    headings = np.arctan2(mag_y, mag_x) * (180 / np.pi) - 90

    # -------------------------构建观测--------------------------
    # 将gnss_dr数据集长度与ground_truth数据集长度对齐
    utcTimeMillis_gnss = gnss_df['utcTimeMillis'].unique()
    utcTimeMillis_gt = gt_df['UnixTimeMillis'].unique()
    index = np.where(np.isin(utcTimeMillis_gnss, utcTimeMillis_gt))

    # Point positioning（WLS）
    utc, x_wls, v_wls, cov_x, cov_v = point_positioning(gnss_df)

    # Exclude velocity outliers
    x_wls, v_wls, cov_x, cov_v = exclude_interpolate_outlier(x_wls, v_wls, cov_x, cov_v)
    v_wls = v_wls[index]
    x_wls = x_wls[index]
    # Convert to latitude and longitude
    llh_wls = np.array(pm.ecef2geodetic(x_wls[:, 0], x_wls[:, 1], x_wls[:, 2])).T
    # predict with kf
    x_kf, _, _ = Kalman_smoothing(x_wls, v_wls, cov_x, cov_v, phone)
    llh_kf = np.array(pm.ecef2geodetic(x_kf[:, 0], x_kf[:, 1], x_kf[:, 2])).T
    enu_kf = np.array(pm.ecef2enu(x_kf[:, 0], x_kf[:, 1],x_kf[:, 2],df_temp.loc[0,'LatitudeDegrees'],df_temp.loc[0,'LongitudeDegrees'],df_temp.loc[0,'AltitudeMeters'])).T

    ######### ecef/geo to enu (edited by tang) #############
    enu_wls = pm.ecef2enu(x_wls[:, 0], x_wls[:, 1],x_wls[:, 2],df_temp.loc[0,'LatitudeDegrees'],df_temp.loc[0,'LongitudeDegrees'],df_temp.loc[0,'AltitudeMeters'])
    v_enu_wls = pm.ecef2enuv(v_wls[:, 0], v_wls[:, 1], v_wls[:, 2], df_temp.loc[0,'LatitudeDegrees'],df_temp.loc[0,'LongitudeDegrees'])
    llh_gt = df_temp[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy()
    ecef_gt = coord.geodetic2ecef(llh_gt)
    df_temp['ecefX'] = ecef_gt[:, 0]
    df_temp['ecefY'] = ecef_gt[:, 1]
    df_temp['ecefZ'] = ecef_gt[:, 2]
    enu_gt = pm.ecef2enu(ecef_gt[:, 0], ecef_gt[:, 1],ecef_gt[:, 2],df_temp.loc[0,'LatitudeDegrees'],df_temp.loc[0,'LongitudeDegrees'],df_temp.loc[0,'AltitudeMeters'])
    df_temp['e_gt'] = enu_gt[0]
    df_temp['n_gt'] = enu_gt[1]
    df_temp['u_gt'] = enu_gt[2]
    df_temp['e_wls'] = enu_wls[0]
    df_temp['n_wls'] = enu_wls[1]
    df_temp['u_wls'] = enu_wls[2]
    df_temp['v_e_wls'] = v_enu_wls[0]
    df_temp['v_n_wls'] = v_enu_wls[1]
    df_temp['v_u_wls'] = v_enu_wls[2]
    # Baseline
    ecef_bl = gnss_df.groupby('TimeNanos')[['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].mean().to_numpy()
    ecef_bl = ecef_bl[index]
    llh_bl = np.array(pm.ecef2geodetic(ecef_bl[:, 0], ecef_bl[:, 1], ecef_bl[:, 2])).T
    enu_bl = pm.ecef2enu(ecef_bl[:, 0], ecef_bl[:, 1],ecef_bl[:, 2],df_temp.loc[0,'LatitudeDegrees'],df_temp.loc[0,'LongitudeDegrees'],df_temp.loc[0,'AltitudeMeters'])
    df_temp['e_bl'] = enu_bl[0]
    df_temp['n_bl'] = enu_bl[1]
    df_temp['u_bl'] = enu_bl[2]

    # 创建 DataFrame
    df = pd.DataFrame(df_temp)
    df['heading'] = headings
    df['heading_truth'] = df['BearingDegrees']
    df['end_lat'] = None
    df['end_lon'] = None
    pos_bl = df[['WlsPositionXEcefMeters', 'WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].to_numpy()
    llh_pos_bl = np.array(pm.ecef2geodetic(pos_bl[:, 0], pos_bl[:, 1], pos_bl[:, 2])).T
    df_pos_bl = pd.DataFrame(llh_pos_bl, columns=['Lat', 'Lon', 'A'])
    df['lat_bl'] = df_pos_bl['Lat']
    df['lon_bl'] = df_pos_bl['Lon']

    df_pos_wls = pd.DataFrame(llh_wls, columns=['lat_wls', 'lon_wls', 'A_wls'])
    df['lat_wls'] = df_pos_wls['lat_wls']
    df['lon_wls'] = df_pos_wls['lon_wls']

    # 先初始化第一行
    df.loc[0, 'lat_imu'] = df.loc[0, 'LatitudeDegrees'] # LatitudeDegrees是gt的列名
    df.loc[0, 'lon_imu'] = df.loc[0, 'LongitudeDegrees']
    # -----------------------------------------------------------------------------------------------------
    # for i in range(1, len(df)):
    #     df['end_lat'][i] = df['end_lat'][i - 1] + np.sqrt(
    #         (df['LatitudeDegrees'][i] - df['LatitudeDegrees'][i - 1]) ** 2 + (
    #                     df['LongitudeDegrees'][i] - df['LongitudeDegrees'][i - 1]) ** 2) * np.cos(
    #         np.radians(df['heading_truth'][i - 1]))
    #     df['end_lon'][i] = df['end_lon'][i - 1] + np.sqrt(
    #         (df['LatitudeDegrees'][i] - df['LatitudeDegrees'][i - 1]) ** 2 + (
    #                     df['LongitudeDegrees'][i] - df['LongitudeDegrees'][i - 1]) ** 2) * np.sin(
    #         np.radians(df['heading_truth'][i - 1]))
    #     # 检查缺失值
    # print(df.isnull().sum())
    # -----------------------------------------------------------------------------------------------------
    lla_gt = df[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy()
    # # x_gt = geodetic2ecef(lla_gt)
    # #
    # # v_gt = np.zeros((len(x_gt), 3))
    # #
    # # for i in range(len(x_gt) - 1):
    # #     v_gt[i, 0] = x_gt[i + 1, 0] - x_gt[i, 0]
    # #     v_gt[i, 1] = x_gt[i + 1, 1] - x_gt[i, 1]
    # #     v_gt[i, 2] = x_gt[i + 1, 2] - x_gt[i, 2]
    # #
    # # v_gt[len(x_gt) - 1, 0] = 0
    # # v_gt[len(x_gt) - 1, 1] = 0
    # # v_gt[len(x_gt) - 1, 2] = 0
    #
    # for i in range(1, len(df)):
    #     arrow_length = np.sqrt(
    #         (df['LatitudeDegrees'][i] - df['LatitudeDegrees'][i - 1]) ** 2 + (
    #                     df['LongitudeDegrees'][i] - df['LongitudeDegrees'][i - 1]) ** 2) + 0.000002
    #     df['end_lat'][i] = df['LatitudeDegrees'][i - 1] + arrow_length * np.cos(np.radians(df['heading'][i - 1]))
    #     df['end_lon'][i] = df['LongitudeDegrees'][i - 1] + arrow_length * np.sin(np.radians(df['heading'][i - 1]))
    # -----------------------------------------------------------------------------------------------------
    # for i in range(1, len(df)):
    #     arrow_length = np.sqrt(
    #         (df['lat_pos'][i] - df['lat_pos'][i - 1]) ** 2 + (
    #                     df['lon_pos'][i] - df['lon_pos'][i - 1]) ** 2) + 0.000002
    #     df['end_lat'][i] = df['lat_pos'][i - 1] + arrow_length * np.cos(np.radians(df['heading'][i - 1]))
    #     df['end_lon'][i] = df['lon_pos'][i - 1] + arrow_length * np.sin(np.radians(df['heading'][i - 1]))
    #
    # df['end_lat'] = pd.to_numeric(df['end_lat'], errors='coerce')
    # df['end_lon'] = pd.to_numeric(df['end_lon'], errors='coerce')
    # -----------------------------------------------------------------------------------------------------
    ####### calculate ll_imu with heading #############
    for i in range(1, len(df)):
        arrow_length = np.sqrt(
            (df['lat_wls'].iloc[i] - df['lat_wls'].iloc[i - 1]) ** 2 +
            (df['lon_wls'].iloc[i] - df['lon_wls'].iloc[i - 1]) ** 2) + 0.000002
        # 使用 .loc 安全地赋值
        df.loc[i, 'lat_imu'] = df.loc[i - 1, 'lat_wls'] + arrow_length * np.cos(np.radians(df.loc[i - 1, 'heading'])) # 磁力计计算的heading
        df.loc[i, 'lon_imu'] = df.loc[i - 1, 'lon_wls'] + arrow_length * np.sin(np.radians(df.loc[i - 1, 'heading']))
    # 将 lat_imu 和 lon_imu 转换为数值类型，遇到错误时置为 NaN
    df['lat_imu'] = pd.to_numeric(df['lat_imu'], errors='coerce')
    df['lon_imu'] = pd.to_numeric(df['lon_imu'], errors='coerce')
    rows, cols = np.where(pd.isnull(df))
    # 打印出空缺值的位置
    # for row, col in zip(rows, cols):
    #     print(f'空缺值在第 {row + 1} 行，第 {col + 1} 列')
    # 假设 geodetic2ecef 是你定义的函数，将经纬度转换为 ECEF
    llh_imu = df[['lat_imu', 'lon_imu', 'AltitudeMeters']].to_numpy()
    pos_imu = geodetic2ecef(llh_imu)

    v_imu = np.zeros((len(pos_imu), 3))  # 计算imu速度
    for i in range(len(pos_imu) - 1):
        v_imu[i, 0] = pos_imu[i + 1, 0] - pos_imu[i, 0]
        v_imu[i, 1] = pos_imu[i + 1, 1] - pos_imu[i, 1]
        v_imu[i, 2] = pos_imu[i + 1, 2] - pos_imu[i, 2]
    v_imu[len(pos_imu) - 1, 0] = 0
    v_imu[len(pos_imu) - 1, 1] = 0
    v_imu[len(pos_imu) - 1, 2] = 0

    ########## ESKF(ENU) by tang #########
    roll, pitch = get_roll_pitch_from_accelerometer( # 通过加速度计推roll和pitch
        df_temp['MeasurementX_acc'].values,
        df_temp['MeasurementY_acc'].values,
        df_temp['MeasurementZ_acc'].values)
    RM = euler_to_rotation_matrix(roll, pitch, headings) # 得到旋转矩阵
    RM_HF = get_rotation_matrix_high_freq(acc_,mag_) # 计算高频状态下的旋转矩阵

    v_enu_wls = np.array(v_enu_wls).T
    enu_wls = np.array(enu_wls).T
    enu_gt = np.array(enu_gt).T
    enu_bl = np.array(enu_bl).T

    enu_eskf_hf = Kalman_smoothing_enuESKF_HF(enu_kf, v_enu_wls, RM_HF, df, acc_, cov_x, cov_v, phone)
    ecef_eskf_imu_hf = np.array(pm.enu2ecef(enu_eskf_hf[:, 0], enu_eskf_hf[:, 1],enu_eskf_hf[:, 2],df_temp.loc[0,'LatitudeDegrees'],df_temp.loc[0,'LongitudeDegrees'],df_temp.loc[0,'AltitudeMeters'])).T
    llh_eskf_imu_hf = np.array(pm.ecef2geodetic(ecef_eskf_imu_hf[:, 0], ecef_eskf_imu_hf[:, 1], ecef_eskf_imu_hf[:, 2])).T

    enu_eskf_tang = Kalman_smoothing_enuESKF(enu_kf, v_enu_wls, RM, df, cov_x, cov_v, phone)
    ecef_eskf_imu_tang = np.array(pm.enu2ecef(enu_eskf_tang[:, 0], enu_eskf_tang[:, 1],enu_eskf_tang[:, 2],df_temp.loc[0,'LatitudeDegrees'],df_temp.loc[0,'LongitudeDegrees'],df_temp.loc[0,'AltitudeMeters'])).T
    llh_eskf_imu_tang = np.array(pm.ecef2geodetic(ecef_eskf_imu_tang[:, 0], ecef_eskf_imu_tang[:, 1], ecef_eskf_imu_tang[:, 2])).T

    ############ ESKF (by Xiaofan)，计算磁力计位置和wls误差，预测磁力计位置误差
    pos_er = pos_imu - x_wls
    vel_er = v_imu - v_wls
    num = 0
    # ESKF
    pos_er_kf, _, _ = Kalman_smoothing(pos_er, vel_er, cov_x, cov_v, phone)
    # for i in range(0, len(pos_er_kf)):
    #     if np.any(pos_er[i] > 10):
    #         num = num + 1
    #         pos_er_kf[i] = pos_er[i]
    # print(num)
    x_imu_kf = pos_imu - pos_er_kf
    # # 均值平滑
    # def moving_average(data, window_size):
    #     if window_size < 1:
    #         raise ValueError("window_size must be at least 1.")
    #     if len(data) < window_size:
    #         raise ValueError("data length must be greater than or equal to window_size.")
    #         # 使用 pandas 的滚动窗口功能计算移动平均
    #     smoothed_data = pd.Series(data).rolling(window=window_size, min_periods=1).mean().to_numpy()
    #
    #     return smoothed_data
    #
    # 示例数据
    # window_size = 2
    #


    # def rts_smoothing(data, noise_variance=1.0):
    #     """
    #     对一维数据应用 RTS 平滑
    #
    #     :param data: 一维输入数据，numpy数组
    #     :param noise_variance: 噪声方差
    #     :return: 平滑后的数据
    #     """
    #     n = len(data)
    #     smoothed_data = np.zeros(n)
    #
    #     # 初始化状态
    #     smoothed_data[-1] = data[-1]
    #
    #     # 后向递推
    #     for i in range(n - 2, -1, -1):
    #         # 计算平滑值
    #         smoothed_data[i] = data[i] + (noise_variance / (noise_variance + 1)) * (smoothed_data[i + 1] - data[i])
    #
    #     return smoothed_data
    #
    # # 计算均值平滑
    # x = pos_imu[:, 0]  # 提取第 0 列
    # y = pos_imu[:, 1]  # 提取第 1 列
    # z = pos_imu[:, 2]  # 提取第 2 列
    # ewm_x = rts_smoothing(x, )
    # ewm_y = rts_smoothing(y, )
    # ewm_z = rts_smoothing(z, )

    # # 将平滑后的数据组合
    # ewm_x_imu_kf = np.column_stack((ewm_x, ewm_y, ewm_z))
    enu_eskf_imu = np.array(pm.ecef2enu(x_imu_kf[:, 0], x_imu_kf[:, 1],x_imu_kf[:, 2],df_temp.loc[0,'LatitudeDegrees'],df_temp.loc[0,'LongitudeDegrees'],df_temp.loc[0,'AltitudeMeters'])).T
    llh_eskf_imu = np.array(pm.ecef2geodetic(x_imu_kf[:, 0], x_imu_kf[:, 1], x_imu_kf[:, 2])).T
    #
    # la = llh_eskf_imu[:, 0]  # 提取第 0 列
    # lo = llh_eskf_imu[:, 1]  # 提取第 1 列
    # ewm_x = moving_average(la, window_size)
    # ewm_y = moving_average(lo, window_size)
    # # 将平滑后的数据组合
    # llh_eskf_imu = np.column_stack((ewm_x, ewm_y))

    vd_bl = vincenty_distance(llh_bl, llh_gt)
    vd_wls = vincenty_distance(llh_wls, llh_gt)
    vd_kf = vincenty_distance(llh_kf, llh_gt)
    vd_imu = vincenty_distance(llh_imu, llh_gt)
    # vd_kf_imu = vincenty_distance(llh_kf_imu, llh_gt)
    vd_eskf = vincenty_distance(llh_eskf_imu, llh_gt)
    vd_eskf_tang = vincenty_distance(llh_eskf_imu_tang, llh_gt)
    vd_eskf_hf = vincenty_distance(llh_eskf_imu_hf, llh_gt)

    # Score
    score_bl = calc_score(llh_bl, llh_gt)
    score_wls = calc_score(llh_wls, llh_gt)
    score_kf = calc_score(llh_kf[:-1, :], llh_gt[:-1, :])
    score_imu = calc_score(llh_imu, llh_gt)
    # score_kf_imu = calc_score(llh_kf_imu[:-1, :], llh_gt[:-1, :])
    score_eskf = calc_score(llh_eskf_imu[:-1, :], llh_gt[:-1, :])
    score_eskf_tang = calc_score(llh_eskf_imu_tang[:-1, :], llh_gt[:-1, :])
    score_eskf_imu_hf = calc_score(llh_eskf_imu_hf[:-1, :], llh_gt[:-1, :])

    # 三维误差计算
    derr_bl = np.sqrt(np.sum((enu_gt - enu_bl) ** 2, axis=1))
    derr_wls = np.sqrt(np.sum((enu_gt - enu_wls) ** 2, axis=1))
    derr_kf = np.sqrt(np.sum((enu_gt - enu_kf) ** 2, axis=1))
    derr_eskf_imu = np.sqrt(np.sum((enu_gt - enu_eskf_imu) ** 2, axis=1))
    derr_eskf_imu_tang = np.sqrt(np.sum((enu_gt - enu_eskf_tang) ** 2, axis=1))
    derr_eskf_imu_hf = np.sqrt(np.sum((enu_gt - enu_eskf_hf) ** 2, axis=1))

    print(f'Baseline   : 3D error: {derr_bl.mean():.3f}+{derr_bl.std():.3f} [m], 2D error: {vd_bl.mean():.3f}+{vd_bl.std():.3f} [m], Score {score_bl:.4f} [m]')
    print(f'Robust WLS : 3D error: {derr_wls.mean():.3f}+{derr_wls.std():.3f} [m], 2D error: {vd_wls.mean():.3f}+{vd_wls.std():.3f} [m], Score {score_wls:.4f} [m]')
    print(f'WLS + KF   : 3D error: {derr_kf.mean():.3f}+{derr_kf.std():.3f} [m], 2D error: {vd_kf.mean():.3f}+{vd_kf.std():.3f} [m], Score {score_kf:.4f} [m]')
    # print(f'IMU        : 2D error: {vd_imu.mean():.3f}+{vd_imu.std():.3f} [m], Score {score_imu:.4f} [m]')
    # print(f'KF_IMU     : Distance from ground truth avg {vd_kf_imu.mean():.3f}+{vd_kf_imu.std():.3f} [m], Score {score_kf_imu:.4f} [m]')
    print(f'ESKF_IMU   : 3D error: {derr_eskf_imu.mean():.3f}+{derr_eskf_imu.std():.3f} [m], 2D error: {vd_eskf.mean():.3f}+{vd_eskf.std():.3f} [m], Score {score_eskf:.4f} [m]')
    print(f'ESKF_IMU_Tang : 3D error: {derr_eskf_imu_tang.mean():.3f}+{derr_eskf_imu_tang.std():.3f} [m], 2D error: {vd_eskf_tang.mean():.3f}+{vd_eskf_tang.std():.3f} [m], Score {score_eskf_tang:.4f} [m]')
    print(f'ESKF_IMU (high frequency) : 3D error: {derr_eskf_imu_hf.mean():.3f}+{derr_eskf_imu_hf.std():.3f} [m], 2D error: {vd_eskf_hf.mean():.3f}+{vd_eskf_hf.std():.3f} [m], Score {score_eskf_imu_hf:.4f} [m]')

    # Plot distance error
    plt.figure()
    plt.title(f'{drive}/{phone}')
    plt.ylabel('Distance error [m]')
    plt.plot(vd_bl, label=f'Baseline, Score: {score_bl:.4f} m')
    plt.plot(vd_wls, label=f'Robust WLS, Score: {score_wls:.4f} m')
    plt.plot(vd_kf, label=f'Robust WLS + KF, Score: {score_kf:.4f} m')
    # plt.plot(vd_imu, label=f'IMU, Score: {score_imu:.4f} m')
    # plt.plot(vd_kf_imu, label=f'KF_IMU, Score: {score_kf_imu:.4f} m')
    plt.plot(vd_eskf, label=f'ESKF_IMU, Score: {score_eskf:.4f} m')
    plt.plot(vd_eskf_tang, label=f'ESKF_IMU_Tang, Score: {score_eskf_tang:.4f} m')
    plt.plot(vd_eskf_hf, label=f'ESKF_IMU (high frequency), Score: {score_eskf_imu_hf:.4f} m')
    plt.legend()
    plt.grid()
    # plt.ylim([0, 50])
    # plt.show()

############  Test data and Submission¶
#  This part is based on @saitodevel01 's code. Thank you!
if processing:
    #path = '/mnt/sdb/home/tangjh/smartphone-decimeter-2022'
    # savepath = Path('/mnt/sdb/home/tangjh/smartphone-decimeter-2022')
    path = '/mnt/sdb/home/tangjh/GSDC2023/sdc2023'
    savepath = Path(r'/mnt/sdb/home/tangjh/GSDC2023/sdc2023/')
    tripIDskiplist = []

    # Read data of imu bias
    mag_bias = pd.read_csv(f'/mnt/sdb/home/tangjh/GSDC2023/sdc2023/result/mag_bias/mag_bias.csv')
    acc_bias = pd.read_csv(f'/mnt/sdb/home/tangjh/GSDC2023/sdc2023/result/acc_bias/acc_bias.csv')

    test_dfs = []
    truth_dfs = []
    record_dfs = []
    record_noalt_ID = []
    record_consecutive_ID = []
    obsmode = 'kf' # 选择用于eskf的位置观测，可选 kf wls

    # Loop for each trip
    for i, dirname in enumerate(tqdm(sorted(gl.glob(f'{path}/train/*/*/')))):
        # 2023-09-07-18-59-us-ca/pixel7pro/ 2023-09-07-19-33-us-ca/pixel6pro/ 2023-09-07-18-59-us-ca/pixel5/ 2023-05-24-20-26-us-ca-sjc-ge2/pixel7pro 2023-05-16-19-54-us-ca-mtv-xe1/pixel5
        dirname = path + '/train/2023-05-24-20-26-us-ca-sjc-ge2/pixel7pro/' #可以选择特定轨迹来debug
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        drive, phone = dirname.split('/')[-3:-1]
        tripID = f'{drive}/{phone}'
        print(f'Processing: {tripID}')
        try:
            if (not os.path.exists(f'{dirname}/baseline_imu_eskf.csv')) and (not tripID in tripIDskiplist):
                # Read data
                gnss_df = pd.read_csv(f'{dirname}/device_gnss.csv')  # GNSS data
                gt_df = pd.read_csv(f'{dirname}/ground_truth.csv')  # ground truth
                imu = pd.read_csv(f'{dirname}/device_imu.csv')
                acc = imu.query(f"MessageType=='UncalAccel'").reset_index(drop=True)
                gyr = imu.query(f"MessageType=='UncalGyro'").reset_index(drop=True)
                mag = imu.query(f"MessageType=='UncalMag'").reset_index(drop=True)
                # gt数据和gnss数据对齐
                utclist = gt_df['UnixTimeMillis'].tolist()
                gnss_df = gnss_df[gnss_df['utcTimeMillis'].isin(utclist)]

                # -------------------------零偏校正（一般不用动）--------------------------
                cal = True  # 是否校正磁力计零偏 True
                diy = False  # 是否使用自己计算的零偏值进行校正 False
                g_bias = True # 是否修正重力
                if cal:
                    if diy:
                        # 磁力计恒偏校正
                        mag['MeasurementX'] = mag['MeasurementX'] + mag_bias.loc[ mag_bias['tripID'] == f'{drive}/{phone}', 'bias_x'].values
                        mag['MeasurementY'] = mag['MeasurementY'] - mag['BiasY']
                        mag['MeasurementZ'] = mag['MeasurementZ']
                        mag['Y'] = mag['MeasurementY']
                        mag['MeasurementY'] = -mag['MeasurementZ']  + mag_bias.loc[mag_bias['tripID'] == f'{drive}/{phone}', 'bias_y'].values - 100
                        mag['MeasurementZ'] = mag['Y']
                        # 加速度计恒偏校正
                        acc['MeasurementX'] = acc['MeasurementX'] + acc_bias.loc[acc_bias['tripID'] == f'{drive}/{phone}', 'bias_x'].values
                        acc['MeasurementY'] = acc['MeasurementY']
                        acc['MeasurementZ'] = acc['MeasurementZ']
                        acc['Y'] = acc['MeasurementY']
                        acc['MeasurementY'] = -acc['MeasurementZ'] - 2 + acc_bias.loc[acc_bias['tripID'] == f'{drive}/{phone}', 'bias_y'].values
                        acc['MeasurementZ'] = acc['Y'] + acc_bias.loc[acc_bias['tripID'] == f'{drive}/{phone}', 'bias_z'].values
                        if g_bias:
                            acc['MeasurementZ'] = acc['MeasurementZ'] - g_acc # TODO 一部分数据需要减去重力

                    else:
                        mag['MeasurementX'] = mag['MeasurementX'] - mag['BiasX']
                        mag['MeasurementY'] = mag['MeasurementY'] - mag['BiasY']
                        mag['MeasurementZ'] = mag['MeasurementZ'] - mag['BiasZ']
                        mag['Y'] = mag['MeasurementY']
                        mag['MeasurementY'] = -mag['MeasurementZ']
                        mag['MeasurementZ'] = mag['Y']
                        acc['Y'] = acc['MeasurementY']
                        acc['MeasurementY'] = -acc['MeasurementZ']  # - 2 + acc_bias.loc[acc_bias['tripID'] == f'{drive}/{phone}', 'bias_y'].values
                        if g_bias:
                            acc['MeasurementZ'] = acc['Y'] - g_acc # TODO 一部分数据需要减去重力
                        else:
                            acc['MeasurementZ'] = acc['Y']
                        gyr['Y'] = gyr['MeasurementY']
                        gyr['MeasurementY'] = gyr['MeasurementZ'] # 是否应该加负号
                        gyr['MeasurementZ'] = gyr['Y']
                if not cal:
                    mag['Y'] = mag['MeasurementY']
                    mag['MeasurementY'] = -mag['MeasurementZ']
                    mag['MeasurementZ'] = mag['Y']
                    acc['Y'] = acc['MeasurementY']
                    acc['MeasurementY'] = -acc['MeasurementZ']  # - 2 + acc_bias.loc[acc_bias['tripID'] == f'{drive}/{phone}', 'bias_y'].values
                    if g_bias:
                        acc['MeasurementZ'] = acc['Y'] - g_acc  # TODO 一部分数据需要减去重力
                    else:
                        acc['MeasurementZ'] = acc['Y']

                    # -------------------------检查数据--------------------------
                pd.set_option('display.max_columns', None)  # 显示所有行
                acc_ = acc[['utcTimeMillis', 'MeasurementX', 'MeasurementY', 'MeasurementZ']]
                acc_['acc_index'] = acc_.index
                print(f'acc length: {len(acc_)}')
                gyr_ = gyr[['utcTimeMillis', 'MeasurementX', 'MeasurementY', 'MeasurementZ']]
                gyr_['gyr_index'] = gyr_.index
                print(f'gyr length: {len(gyr_)}')
                mag_ = mag[['utcTimeMillis', 'MeasurementX', 'MeasurementY', 'MeasurementZ']]
                mag_['mag_index'] = mag_.index
                print(f'mag length: {len(mag_)}')
                pos = gnss_df[['utcTimeMillis', 'WlsPositionXEcefMeters','WlsPositionYEcefMeters', 'WlsPositionZEcefMeters']].drop_duplicates('utcTimeMillis',
                              keep='first')

                # -------------------------数据拼接--------------------------
                acc_.rename(columns={'MeasurementX': 'MeasurementX_acc', 'MeasurementY': 'MeasurementY_acc',
                                     'MeasurementZ': 'MeasurementZ_acc'}, inplace=True)
                gyr_.rename(columns={'MeasurementX': 'MeasurementX_gyr', 'MeasurementY': 'MeasurementY_gyr',
                                     'MeasurementZ': 'MeasurementZ_gyr'}, inplace=True)
                mag_.rename(columns={'MeasurementX': 'MeasurementX_mag', 'MeasurementY': 'MeasurementY_mag',
                                     'MeasurementZ': 'MeasurementZ_mag'}, inplace=True)
                df_temp = pd.merge_asof(
                    pos,
                    acc_,
                    left_on=['utcTimeMillis'],
                    right_on=['utcTimeMillis'],
                    direction='nearest',
                    tolerance=20)

                df_temp = pd.merge_asof(
                    df_temp,
                    gyr_,
                    left_on=['utcTimeMillis'],
                    right_on=['utcTimeMillis'],
                    direction='nearest',
                    tolerance=20)

                df_temp = pd.merge_asof(
                    df_temp,
                    mag_,
                    left_on=['utcTimeMillis'],
                    right_on=['utcTimeMillis'],
                    direction='nearest',
                    tolerance=10)

                df_temp = pd.merge_asof(
                    gt_df,
                    df_temp,
                    left_on=['UnixTimeMillis'],
                    right_on=['utcTimeMillis'],
                    direction='nearest',
                    tolerance=5)

                columns_to_drop = ['SpeedAccuracyMps', 'BearingAccuracyDegrees', 'elapsedRealtimeNanos',
                                   'VerticalAccuracyMeters']
                df_temp = df_temp.drop(columns=columns_to_drop)

                # 检查数据有无空缺值
                nan_acc = df_temp['MeasurementX_acc'].isnull().sum()
                nan_gyr = df_temp['MeasurementX_gyr'].isnull().sum()
                nan_mag = df_temp['MeasurementX_mag'].isnull().sum()
                print(f'acc nan number:{nan_acc},mag nan number:{nan_mag},gyr nan number:{nan_gyr}')
                # 做了合并之后填充nan值
                df_temp = compensate(df_temp, ['MeasurementX_acc', 'MeasurementY_acc', 'MeasurementZ_acc'], max_consecutive)
                df_temp = compensate(df_temp, ['MeasurementX_mag', 'MeasurementY_mag', 'MeasurementZ_mag'], max_consecutive)
                df_temp = compensate(df_temp, ['MeasurementX_gyr', 'MeasurementY_gyr', 'MeasurementZ_gyr'], max_consecutive)

                # -------------------------计算偏航角--------------------------
                mag_x = df_temp['MeasurementX_mag'].to_numpy()
                mag_y = df_temp['MeasurementY_mag'].to_numpy()
                mag_z = df_temp['MeasurementZ_mag'].to_numpy()
                # 计算航向角（以度为单位），以北方向为0度，顺时针数，范围是0~359.99
                headings = np.arctan2(mag_y, mag_x) * (180 / np.pi) - 90

                # ----------------------构建观测计算wls和kf位置--------------------------
                # 将gnss_dr数据集长度与ground_truth数据集长度对齐
                # utcTimeMillis_gnss = gnss_df['utcTimeMillis'].unique()
                # utcTimeMillis_gt = gt_df['UnixTimeMillis'].unique()
                # index = np.where(np.isin(utcTimeMillis_gnss, utcTimeMillis_gt))

                # Point positioning（WLS）
                utc, x_wls, v_wls, cov_x, cov_v = point_positioning(gnss_df)

                # Exclude velocity outliers
                x_wls, v_wls, cov_x, cov_v = exclude_interpolate_outlier(x_wls, v_wls, cov_x, cov_v)

                x_wls = remove_nans(x_wls)
                v_wls = remove_nans(v_wls)
                # v_wls = v_wls[index]
                # x_wls = x_wls[index]
                # Convert to latitude and longitude
                llh_wls = np.array(pm.ecef2geodetic(x_wls[:, 0], x_wls[:, 1], x_wls[:, 2])).T
                # predict with kf
                x_kf, _, _ = Kalman_smoothing(x_wls, v_wls, cov_x, cov_v, phone)
                assert np.all(~np.isnan(x_kf))
                llh_kf = np.array(pm.ecef2geodetic(x_kf[:, 0], x_kf[:, 1], x_kf[:, 2])).T
                enu_kf = np.array(pm.ecef2enu(x_kf[:, 0], x_kf[:, 1], x_kf[:, 2], df_temp.loc[0, 'LatitudeDegrees'],
                                              df_temp.loc[0, 'LongitudeDegrees'], df_temp.loc[0, 'AltitudeMeters'])).T

                ######### ecef/geo to enu (edited by tang) #############
                enu_wls = pm.ecef2enu(x_wls[:, 0], x_wls[:, 1], x_wls[:, 2], df_temp.loc[0, 'LatitudeDegrees'],
                                      df_temp.loc[0, 'LongitudeDegrees'], df_temp.loc[0, 'AltitudeMeters'])
                v_enu_wls = pm.ecef2enuv(v_wls[:, 0], v_wls[:, 1], v_wls[:, 2], df_temp.loc[0, 'LatitudeDegrees'],
                                         df_temp.loc[0, 'LongitudeDegrees'])
                try:
                    llh_gt = df_temp[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy()
                except:
                    record_noalt_ID.append(tripID)
                    print(f'no altitude ID:{tripID}')
                    continue
                ecef_gt = coord.geodetic2ecef(llh_gt)
                enu_gt = pm.ecef2enu(ecef_gt[:, 0], ecef_gt[:, 1], ecef_gt[:, 2], df_temp.loc[0, 'LatitudeDegrees'],
                                     df_temp.loc[0, 'LongitudeDegrees'], df_temp.loc[0, 'AltitudeMeters'])

                # Baseline
                ecef_bl = pos.to_numpy()
                ecef_bl = ecef_bl[:,1:]
                # Interpolation processing
                x_df = pd.DataFrame({'x': ecef_bl[:, 0], 'y': ecef_bl[:, 1], 'z': ecef_bl[:, 2]})
                x_df = x_df.interpolate(method='spline', order=3, limit_direction='both') # 填充边界外的缺失值，但不填充内部缺失值。
                ecef_bl = x_df.to_numpy()
                # ecef_bl = ecef_bl[index]
                llh_bl = np.array(pm.ecef2geodetic(ecef_bl[:, 0], ecef_bl[:, 1], ecef_bl[:, 2])).T
                enu_bl = pm.ecef2enu(ecef_bl[:, 0], ecef_bl[:, 1], ecef_bl[:, 2], df_temp.loc[0, 'LatitudeDegrees'],
                                     df_temp.loc[0, 'LongitudeDegrees'], df_temp.loc[0, 'AltitudeMeters'])

                ########## ESKF(ENU) by tang #########
                roll, pitch = get_roll_pitch_from_accelerometer(  # 通过加速度计推roll和pitch
                    df_temp['MeasurementX_acc'].values,
                    df_temp['MeasurementY_acc'].values,
                    df_temp['MeasurementZ_acc'].values)
                RM = euler_to_rotation_matrix(roll, pitch, headings)  # 得到旋转矩阵
                RM_HF = get_rotation_matrix_high_freq(acc_, mag_)  # 计算高频状态下的旋转矩阵
                v_enu_wls = np.array(v_enu_wls).T
                enu_wls = np.array(enu_wls).T
                enu_gt = np.array(enu_gt).T
                enu_bl = np.array(enu_bl).T

                ############ eskf for gnss+imu by tang ############
                if obsmode == 'kf':
                    obs_enu = enu_kf
                elif obsmode == 'wls':
                    obs_enu = enu_wls

                # ------- (enu eskf) gyr+acc+mag（100hz）
                enu_eskf_hfv3, df_imuALL = Kalman_smoothing_enuESKF_ALL_HF(obs_enu, v_enu_wls, df_temp, acc_.copy(), mag_.copy(), gyr_.copy(), RM_HF, phone)
                ecef_eskf_imu_hfv3 = np.array(
                    pm.enu2ecef(enu_eskf_hfv3[:, 0], enu_eskf_hfv3[:, 1], enu_eskf_hfv3[:, 2], df_temp.loc[0, 'LatitudeDegrees'],
                                df_temp.loc[0, 'LongitudeDegrees'], df_temp.loc[0, 'AltitudeMeters'])).T
                llh_eskf_imu_hfv3 = np.array(
                    pm.ecef2geodetic(ecef_eskf_imu_hfv3[:, 0], ecef_eskf_imu_hfv3[:, 1], ecef_eskf_imu_hfv3[:, 2])).T

                # ------- enu加上姿态误差估计（100hz）
                enu_eskf_hfv2, raw_acc_v2_dfv2 = Kalman_smoothing_enuESKF_v2_HF(obs_enu, v_enu_wls, df_temp, acc_.copy(), mag_.copy(), RM_HF, phone)
                ecef_eskf_imu_hfv2 = np.array(
                    pm.enu2ecef(enu_eskf_hfv2[:, 0], enu_eskf_hfv2[:, 1], enu_eskf_hfv2[:, 2], df_temp.loc[0, 'LatitudeDegrees'],
                                df_temp.loc[0, 'LongitudeDegrees'], df_temp.loc[0, 'AltitudeMeters'])).T
                llh_eskf_imu_hfv2 = np.array(
                    pm.ecef2geodetic(ecef_eskf_imu_hfv2[:, 0], ecef_eskf_imu_hfv2[:, 1], ecef_eskf_imu_hfv2[:, 2])).T

                #------ enu利用imu高频率推算位置和速度（100hz）
                enu_eskf_hf, raw_acc_df = Kalman_smoothing_enuESKF_HF(obs_enu, v_enu_wls, RM_HF, df_temp, acc_.copy(), cov_x, cov_v, phone)
                ecef_eskf_imu_hf = np.array(
                    pm.enu2ecef(enu_eskf_hf[:, 0], enu_eskf_hf[:, 1], enu_eskf_hf[:, 2], df_temp.loc[0, 'LatitudeDegrees'],
                                df_temp.loc[0, 'LongitudeDegrees'], df_temp.loc[0, 'AltitudeMeters'])).T
                llh_eskf_imu_hf = np.array(
                    pm.ecef2geodetic(ecef_eskf_imu_hf[:, 0], ecef_eskf_imu_hf[:, 1], ecef_eskf_imu_hf[:, 2])).T

                #------ enu使用与卫导定位最接近时刻的惯导推算位置速度（1hz）
                enu_eskf_tang, acc_enu_all = Kalman_smoothing_enuESKF(obs_enu, v_enu_wls, RM, df_temp, cov_x, cov_v, phone)
                ecef_eskf_imu_tang = np.array(pm.enu2ecef(enu_eskf_tang[:, 0], enu_eskf_tang[:, 1], enu_eskf_tang[:, 2], df_temp.loc[0, 'LatitudeDegrees'],
                                                          df_temp.loc[0, 'LongitudeDegrees'], df_temp.loc[0, 'AltitudeMeters'])).T
                llh_eskf_imu_tang = np.array(
                    pm.ecef2geodetic(ecef_eskf_imu_tang[:, 0], ecef_eskf_imu_tang[:, 1], ecef_eskf_imu_tang[:, 2])).T

                # 水平误差计算
                vd_bl = vincenty_distance(llh_bl, llh_gt)
                vd_wls = vincenty_distance(llh_wls, llh_gt)
                vd_kf = vincenty_distance(llh_kf, llh_gt)
                vd_eskf_tang = vincenty_distance(llh_eskf_imu_tang, llh_gt)
                vd_eskf_hf = vincenty_distance(llh_eskf_imu_hf, llh_gt)
                vd_eskf_hfv2 = vincenty_distance(llh_eskf_imu_hfv2, llh_gt)
                vd_eskf_hfv3 = vincenty_distance(llh_eskf_imu_hfv3, llh_gt)

                # Score
                score_bl = calc_score(llh_bl, llh_gt)
                score_wls = calc_score(llh_wls, llh_gt)
                score_kf = calc_score(llh_kf[:-1, :], llh_gt[:-1, :])
                score_eskf_tang = calc_score(llh_eskf_imu_tang[:-1, :], llh_gt[:-1, :])
                score_eskf_imu_hf = calc_score(llh_eskf_imu_hf[:-1, :], llh_gt[:-1, :])
                score_eskf_imu_hfv2 = calc_score(llh_eskf_imu_hfv2[:-1, :], llh_gt[:-1, :])
                score_eskf_imu_hfv3 = calc_score(llh_eskf_imu_hfv3[:-1, :], llh_gt[:-1, :])

                # 三维误差计算
                derr_bl = np.sqrt(np.sum((enu_gt - enu_bl) ** 2, axis=1))
                derr_wls = np.sqrt(np.sum((enu_gt - enu_wls) ** 2, axis=1))
                derr_kf = np.sqrt(np.sum((enu_gt - enu_kf) ** 2, axis=1))
                derr_eskf_imu_tang = np.sqrt(np.sum((enu_gt - enu_eskf_tang) ** 2, axis=1))
                derr_eskf_imu_hf = np.sqrt(np.sum((enu_gt - enu_eskf_hf) ** 2, axis=1))
                derr_eskf_imu_hfv2 = np.sqrt(np.sum((enu_gt - enu_eskf_hfv2) ** 2, axis=1))
                derr_eskf_imu_hfv3 = np.sqrt(np.sum((enu_gt - enu_eskf_hfv3) ** 2, axis=1))

                print(f'Result of {tripID}')
                print(
                    f'Baseline   : 3D error: {derr_bl.mean():.3f}+{derr_bl.std():.3f} [m], 2D error: {vd_bl.mean():.3f}+{vd_bl.std():.3f} [m], Score {score_bl:.4f} [m]')
                print(
                    f'Robust WLS : 3D error: {derr_wls.mean():.3f}+{derr_wls.std():.3f} [m], 2D error: {vd_wls.mean():.3f}+{vd_wls.std():.3f} [m], Score {score_wls:.4f} [m]')
                print(
                    f'WLS + KF   : 3D error: {derr_kf.mean():.3f}+{derr_kf.std():.3f} [m], 2D error: {vd_kf.mean():.3f}+{vd_kf.std():.3f} [m], Score {score_kf:.4f} [m]')
                print(
                    f'ESKF_IMU_Tang : 3D error: {derr_eskf_imu_tang.mean():.3f}+{derr_eskf_imu_tang.std():.3f} [m], 2D error: {vd_eskf_tang.mean():.3f}+{vd_eskf_tang.std():.3f} [m], Score {score_eskf_tang:.4f} [m]')
                print(
                    f'ESKF_IMU (high frequency) : 3D error: {derr_eskf_imu_hf.mean():.3f}+{derr_eskf_imu_hf.std():.3f} [m], 2D error: {vd_eskf_hf.mean():.3f}+{vd_eskf_hf.std():.3f} [m], Score {score_eskf_imu_hf:.4f} [m]')
                print(
                    f'ESKF_IMU (high frequency V2) : 3D error: {derr_eskf_imu_hfv2.mean():.3f}+{derr_eskf_imu_hfv2.std():.3f} [m], 2D error: {vd_eskf_hfv2.mean():.3f}+{vd_eskf_hfv2.std():.3f} [m], Score {score_eskf_imu_hfv2:.4f} [m]')
                print(
                    f'ESKF_IMU (acc+gyr_mag) : 3D error: {derr_eskf_imu_hfv3.mean():.3f}+{derr_eskf_imu_hfv3.std():.3f} [m], 2D error: {vd_eskf_hfv3.mean():.3f}+{vd_eskf_hfv3.std():.3f} [m], Score {score_eskf_imu_hfv3:.4f} [m]')
                # Plot distance error
                # plt.figure()
                # plt.title(f'{drive}/{phone}')
                # plt.ylabel('Distance error [m]')
                # plt.plot(vd_bl, label=f'Baseline, Score: {score_bl:.4f} m')
                # plt.plot(vd_wls, label=f'Robust WLS, Score: {score_wls:.4f} m')
                # plt.plot(vd_kf, label=f'Robust WLS + KF, Score: {score_kf:.4f} m')
                # plt.plot(vd_eskf_tang, label=f'ESKF_IMU_Tang, Score: {score_eskf_tang:.4f} m')
                # plt.plot(vd_eskf_hf, label=f'ESKF_IMU (high frequency), Score: {score_eskf_imu_hf:.4f} m')
                # plt.plot(vd_eskf_hfv2, label=f'ESKF_IMU (high frequencyV2), Score: {score_eskf_imu_hfv2:.4f} m')
                # plt.legend()
                # plt.grid()
                # plt.ylim([0, 50])
                # plt.show()

                # Interpolation for submission
                UnixTimeMillis = gt_df['UnixTimeMillis'].to_numpy()
                e_kf = InterpolatedUnivariateSpline(utc, enu_kf[:, 0], ext=3)(UnixTimeMillis)
                n_kf = InterpolatedUnivariateSpline(utc, enu_kf[:, 1], ext=3)(UnixTimeMillis)
                u_kf = InterpolatedUnivariateSpline(utc, enu_kf[:, 2], ext=3)(UnixTimeMillis)
                e_bl = InterpolatedUnivariateSpline(utc, enu_bl[:, 0], ext=3)(UnixTimeMillis)
                n_bl = InterpolatedUnivariateSpline(utc, enu_bl[:, 1], ext=3)(UnixTimeMillis)
                u_bl = InterpolatedUnivariateSpline(utc, enu_bl[:, 2], ext=3)(UnixTimeMillis)
                e_wls = InterpolatedUnivariateSpline(utc, enu_wls[:, 0], ext=3)(UnixTimeMillis)
                n_wls = InterpolatedUnivariateSpline(utc, enu_wls[:, 1], ext=3)(UnixTimeMillis)
                u_wls = InterpolatedUnivariateSpline(utc, enu_wls[:, 2], ext=3)(UnixTimeMillis)
                e_eskf_tang = InterpolatedUnivariateSpline(utc, enu_eskf_tang[:, 0], ext=3)(UnixTimeMillis)
                n_eskf_tang = InterpolatedUnivariateSpline(utc, enu_eskf_tang[:, 1], ext=3)(UnixTimeMillis)
                u_eskf_tang = InterpolatedUnivariateSpline(utc, enu_eskf_tang[:, 2], ext=3)(UnixTimeMillis)
                e_eskf_hf = InterpolatedUnivariateSpline(utc, enu_eskf_hf[:, 0], ext=3)(UnixTimeMillis)
                n_eskf_hf = InterpolatedUnivariateSpline(utc, enu_eskf_hf[:, 1], ext=3)(UnixTimeMillis)
                u_eskf_hf = InterpolatedUnivariateSpline(utc, enu_eskf_hf[:, 2], ext=3)(UnixTimeMillis)
                e_eskf_hfv2 = InterpolatedUnivariateSpline(utc, enu_eskf_hfv2[:, 0], ext=3)(UnixTimeMillis)
                n_eskf_hfv2 = InterpolatedUnivariateSpline(utc, enu_eskf_hfv2[:, 1], ext=3)(UnixTimeMillis)
                u_eskf_hfv2 = InterpolatedUnivariateSpline(utc, enu_eskf_hfv2[:, 2], ext=3)(UnixTimeMillis)
                e_eskf_hfall = InterpolatedUnivariateSpline(utc, enu_eskf_hfv3[:, 0], ext=3)(UnixTimeMillis)
                n_eskf_hfall = InterpolatedUnivariateSpline(utc, enu_eskf_hfv3[:, 1], ext=3)(UnixTimeMillis)
                u_eskf_hfall = InterpolatedUnivariateSpline(utc, enu_eskf_hfv3[:, 2], ext=3)(UnixTimeMillis)
                ve_wls = InterpolatedUnivariateSpline(utc, v_enu_wls[:, 0], ext=3)(UnixTimeMillis)
                vn_wls = InterpolatedUnivariateSpline(utc, v_enu_wls[:, 1], ext=3)(UnixTimeMillis)
                vu_wls = InterpolatedUnivariateSpline(utc, v_enu_wls[:, 2], ext=3)(UnixTimeMillis)
                acc_e = InterpolatedUnivariateSpline(utc, acc_enu_all[:, 0], ext=3)(UnixTimeMillis)
                acc_n = InterpolatedUnivariateSpline(utc, acc_enu_all[:, 1], ext=3)(UnixTimeMillis)
                acc_u = InterpolatedUnivariateSpline(utc, acc_enu_all[:, 2], ext=3)(UnixTimeMillis)

                # Save the velocity(enu) data
                trip_baseline_df = pd.DataFrame({
                    'tripId': tripID, 'UnixTimeMillis': UnixTimeMillis,
                    'VE_wls': ve_wls, 'VN_wls': vn_wls,'VU_wls': vu_wls,
                    'Acc_e': acc_e, 'Acc_n': acc_n, 'Acc_u': acc_u,
                    'E_gt': enu_gt[:,0], 'N_gt': enu_gt[:,1], 'U_gt': enu_gt[:,2],
                    'E_bl': e_bl, 'N_bl': n_bl, 'U_bl': u_bl,
                    'E_wls':e_wls,'N_wls':n_wls,'U_wls':u_wls,
                    'E_kf':e_kf,'N_kf':n_kf,'U_kf':u_kf,
                    'E_eskf':e_eskf_tang,'N_eskf':n_eskf_tang,'U_eskf':u_eskf_tang,
                    'E_eskf_hf': e_eskf_hf, 'N_eskf_hf': n_eskf_hf, 'U_eskf_hf': u_eskf_hf,
                    'E_eskf_hfv2': e_eskf_hfv2, 'N_eskf_hfv2': n_eskf_hfv2, 'U_eskf_hfv2': u_eskf_hfv2,
                    'E_eskf_hfall': e_eskf_hfall, 'N_eskf_hfall': n_eskf_hfall, 'U_eskf_hfall': u_eskf_hfall,
                })
                trip_baseline_df.to_csv(f'{dirname}/baseline_eskf_{obsmode}_cal_{cal}_g_bias_{g_bias}.csv', index=False)
                df_imuALL.to_csv(f'{dirname}/Raw_measurement_imu_ALL.csv', index=False)
                with open(dirname + '/Rotation_Matrix_HF.pkl','wb') as value_file:
                    pickle.dump(RM_HF, value_file, True)
                value_file.close()
                # test_dfs.append(trip_df)
                # # Write submission.csv
                # trip_df.to_csv(f'{dirname}/baseline_KF.csv', index=False)
                # test_dftmp = pd.concat(test_dfs)
                # test_dftmp.to_csv(savepath / 'baseline_locations_train_2023.csv', index=False)

                record_df = pd.DataFrame({
                    'tripId': tripID,
                    '2Derror_bl_avg': vd_bl.mean(), '3Derror_bl_avg': derr_bl.mean(),
                    '2Derror_wls_avg': vd_wls.mean(), '3Derror_wls_avg': derr_wls.mean(),
                    '2Derror_kf_avg': vd_kf.mean(), '3Derror_kf_avg': derr_kf.mean(),
                    '2Derror_imueskf_avg': vd_eskf_tang.mean(), '3Derror_imueskf_avg': derr_eskf_imu_tang.mean(),
                    '2Derror_imueskf_hf_avg': vd_eskf_hf.mean(), '3Derror_imueskf_hf_avg': derr_eskf_imu_hf.mean(),
                    '2Derror_imueskf_hfv2_avg': vd_eskf_hfv2.mean(), '3Derror_imueskf_hfv2_avg': derr_eskf_imu_hfv2.mean(),
                    '2Derror_imueskf_hfv3_avg': vd_eskf_hfv3.mean(), '3Derror_imueskf_hfv3_avg': derr_eskf_imu_hfv3.mean(),
                    '3Derror_kf-imueskf_hfv2': derr_kf.mean()-derr_eskf_imu_hfv2.mean(),
                    '3Derror_wls-imueskf_hfv2': derr_wls.mean() - derr_eskf_imu_hfv2.mean(),
                    '2Derror_kf-imueskf_hfv2': vd_kf.mean() - vd_eskf_hfv2.mean(),
                    '3Derror_wls-imueskf_hfv2': vd_wls.mean() - vd_eskf_hfv2.mean(),
                }, index=[f'{i}'])
                record_dfs.append(record_df)

                    # To save covatiance of process and measurement

        except Exception as e:
            record_consecutive_ID.append(tripID)
            print(f"{tripID} error: {e}")

# Write submission.csv
records_df = pd.concat(record_dfs)
records_df.to_csv(savepath / f'records_train_{obsmode}_cal_{cal}_g_bias_{g_bias}_all_2023.csv', index=False)
df_consecutive_ID = pd.DataFrame({'errorID':record_consecutive_ID})
df_noalttraj = pd.DataFrame({'noaltID':record_noalt_ID})
df_consecutive_ID.to_csv(f'{savepath}/record_consecutive {max_consecutive}_ID.csv', index=False)
df_noalttraj.to_csv(f'{savepath}/record_noalt_ID.csv', index=False)

import sys, os
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from numpy.random import default_rng
import time

from src.gnss_lib.utils import datetime_to_tow
import src.gnss_lib.coordinates as coord
# from env.env_processing_gpsl1 import *

class LOSPRRprocess():
    def __init__ (self,gnss_df,truth_df,tripID,out_root,baseline_df):
        self._path=tripID
        self.gnss_df=gnss_df
        self.truth_df=truth_df
        self.out_root=out_root
        self.baseline_df=baseline_df

    def LOSPRRprocesses(self):
        _path=self._path
        gt=self.truth_df
        lla = gt[['LatitudeDegrees', 'LongitudeDegrees', 'AltitudeMeters']].to_numpy()
        ecef = coord.geodetic2ecef(lla)
        gt['ecefX'] = ecef[:, 0]
        gt['ecefY'] = ecef[:, 1]
        gt['ecefZ'] = ecef[:, 2]
        gt['b'] = np.zeros_like(ecef[:, 0]) # 初始化一个与数组的特定列形状相同的全零数组
        self.baseline_df['bias_bl'] = np.zeros_like(ecef[:, 0])
        self.baseline_df['bias_wls_igst'] = np.zeros_like(ecef[:, 0])
        self.baseline_df['bias_KF_realtime'] = np.zeros_like(ecef[:, 0])

        data=self.gnss_df
        data["PrM"] = data["RawPseudorangeMeters"] \
                      + data["SvClockBiasMeters"] \
                      - data["IsrbMeters"] \
                      - data["IonosphericDelayMeters"] \
                      - data["TroposphericDelayMeters"] # 伪距钟差等修正
        data["PrrM"] = data["PseudorangeRateMetersPerSecond"] \
                      + data["SvClockDriftMetersPerSecond"] # 伪距率等修正

        data["SvName"] = data["SignalType"] + "_" + str(data["Svid"])
        data["SvName"] = data["ConstellationType"].replace(
            [1, 2, 3, 4, 5, 6], ["G", "S", "R", "Q", "B", "E"]) + data["Svid"].astype(str)
        derived_timestamps = data['utcTimeMillis'].unique()
        gt_timestamps = gt['UnixTimeMillis'].unique() # 所有唯一时间戳值
        gt["millisSinceGpsEpoch"]=gt['UnixTimeMillis']
        utc_gt = gt["millisSinceGpsEpoch"].values
        # The timestamps in derived are one epoch ahead. We need to map each epoch
        # in derived to the prior one (in Raw).
        # indexes = np.searchsorted(gt_timestamps, derived_timestamps) # 这个函数返回将元素插入数组后仍然保持数组有序的位置索引
        # # indexes = np.unique(indexes)[:-1]
        # # from_t_to_fix_derived = dict(zip(derived_timestamps, gt_timestamps[indexes - 1])) # -1 because GSDC2021 mismatch
        # from_t_to_fix_derived = dict(zip(derived_timestamps, gt_timestamps[indexes]))
        # data['millisSinceGpsEpoch_old'] = data['utcTimeMillis'].values
        # data['millisSinceGpsEpoch'] = np.array(
        #     list(map(lambda v: from_t_to_fix_derived[v], data['utcTimeMillis'])))
        utclist = utc_gt.tolist() # 排除gnss数据中没有真值的时间戳
        data = data[data['utcTimeMillis'].isin(utclist)].copy()
        data.loc[:,'millisSinceGpsEpoch_old'] = data['utcTimeMillis'].values
        data.loc[:,'millisSinceGpsEpoch'] = data['utcTimeMillis'].values

        count = 0
        chunk = []
        chunk_num = 0
        null_break=False
        min_sat_all=1e6
        max_sat_all=0
        min_sat_gpsl1=1e6
        max_sat_gpsl1=0
        min_sat_glog1=1e6
        max_sat_glog1=0
        min_sat_bdsb1i=1e6
        max_sat_bdsb1i=0
        min_sat_bdsb1c=1e6 # 新增频点
        max_sat_bdsb1c=0
        min_sat_bdsb2a=1e6 # 新增频点
        max_sat_bdsb2a=0
        min_sat_gale1=1e6
        max_sat_gale1=0
        min_sat_gpsl5=1e6
        max_sat_gpsl5=0
        min_sat_gale5a=1e6
        max_sat_gale5a=0
        min_sat_qzsj1=1e6
        max_sat_qzsj1=0
        min_sat_qzsj5=1e6
        max_sat_qzsj5=0
        noobscount = 0
        less4sat_count = 0 # 统计星数小于4次数
        noobstime_list =[]

        for idx, dframe in tqdm(data.groupby("millisSinceGpsEpoch")):
            # Drop duplicate data if it exists
            dframe.drop_duplicates(inplace=True) # 用于移除数据框（DataFrame）中重复行的方法
            # remove nans in dframe 20221025
            dframe = dframe[dframe['SignalType'].notna()]
            if len(dframe) < 1: # check if signal is miss, pad with previous gnss obs
                try:
                    dframe_multic_pre = dframe_multic.copy()
                    dframe_multic_pre['utcTimeMillis'] = idx
                    dframe_multic_pre['millisSinceGpsEpoch'] = idx
                    chunk.append(dframe_multic_pre)
                    count += 1
                    noobscount += 1
                    noobstime_list.append(idx)
                    continue
                except:
                    noobscount += 1
                    continue
            elif len(dframe) < 4:
                less4sat_count += 1
            # Remove all signals except GPS L1
            dframe_L1 = dframe.drop(index=dframe.loc[dframe["SignalType"] != "GPS_L1_CA"].index)
            dframe_L1.drop_duplicates(subset=['Svid'], inplace=True) # 使用'Svid'列的数值来判断是否为重复行，并移除这些重复行。
            dframe_L1.reset_index(drop=True, inplace=True)
            # remove  QZS_J1 and  QZS_J5
            dframe_multic = dframe.drop(index=dframe.loc[dframe["SignalType"] == "QZS_L1_CA"].index)
            dframe_multic = dframe_multic.drop(index=dframe.loc[dframe["SignalType"] == "QZS_L5_Q"].index)
            #  Svid rules
            #  GPS_L1_CA +0 , GPS_L5 +100
            #  BDS_B1I +200, BDS_B1C +300, BDS_B2AP +400,
            #  GLO_G1 +500
            #  GAL_E1 +600, GAL_E5A +700
            dframe_multic.loc[dframe["SignalType"] == "GPS_L1_CA", 'Svid'] += 0
            dframe_multic.loc[dframe["SignalType"] == "GPS_L5_Q", 'Svid']+= 100
            dframe_multic.loc[dframe["SignalType"] == "BDS_B1_I", 'Svid']+= 200
            dframe_multic.loc[dframe["SignalType"] == "BDS_B1_C", 'Svid']+= 300  # 新增了BDS的频点
            dframe_multic.loc[dframe["SignalType"] == "BDS_B2A_P", 'Svid']+= 400 # 新增了BDS的频点
            dframe_multic.loc[dframe["SignalType"] == "GLO_G1_CA", 'Svid']+=500
            dframe_multic.loc[dframe["SignalType"] == "GAL_E1_C_P", 'Svid']+=600
            dframe_multic.loc[dframe["SignalType"] == "GAL_E5A_Q", 'Svid']+=700
            dframe_multic.drop_duplicates(subset=['Svid'], inplace=True)
            dframe_multic.reset_index(drop=True, inplace=True) # 重置 DataFrame 索引的方法, drop=True 表示在重置索引的丢弃原始索引, inplace=True表示在原始DF上进行操作
            # Use satellite measurements to obtain receiver clock ground truth estimate
            gt_row = gt["millisSinceGpsEpoch"] == idx
            gt_slice = gt.loc[gt_row].copy() # 时间戳对齐
            bl_row = self.baseline_df['UnixTimeMillis']==idx
            baseline_slice=self.baseline_df.loc[bl_row].copy()

            # record sat nums: 每个卫星系统频段的最大最小卫星数统计
            if len(dframe)>max_sat_all:
                max_sat_all=len(dframe)
            if len(dframe)<min_sat_all:
                min_sat_all=len(dframe)
            if len(dframe_L1)>max_sat_gpsl1:
                max_sat_gpsl1=len(dframe_L1)
            if len(dframe_L1)<min_sat_gpsl1:
                min_sat_gpsl1=len(dframe_L1)
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "GLO_G1_CA"].index))
            if tmpnum>max_sat_glog1:
                max_sat_glog1=tmpnum
            if tmpnum<min_sat_glog1:
                min_sat_glog1=tmpnum
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "BDS_B1_I"].index))
            if tmpnum>max_sat_bdsb1i:
                max_sat_bdsb1i=tmpnum
            if tmpnum<min_sat_bdsb1i:
                min_sat_bdsb1i=tmpnum
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "BDS_B1_C"].index))
            if tmpnum>max_sat_bdsb1c:
                max_sat_bdsb1c=tmpnum
            if tmpnum<min_sat_bdsb1c:
                min_sat_bdsb1c=tmpnum
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "BDS_B2A_P"].index))
            if tmpnum>max_sat_bdsb2a:
                max_sat_bdsb2a=tmpnum
            if tmpnum<min_sat_bdsb2a:
                min_sat_bdsb2a=tmpnum
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "GAL_E1_C_P"].index))
            if tmpnum>max_sat_gale1:
                max_sat_gale1=tmpnum
            if tmpnum<min_sat_gale1:
                min_sat_gale1=tmpnum
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "GPS_L5_Q"].index))
            if tmpnum>max_sat_gpsl5:
                max_sat_gpsl5=tmpnum
            if tmpnum<min_sat_gpsl5:
                min_sat_gpsl5=tmpnum
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "GAL_E5A_Q"].index))
            if tmpnum>max_sat_gale5a:
                max_sat_gale5a=tmpnum
            if tmpnum<min_sat_gale5a:
                min_sat_gale5a=tmpnum
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "QZS_L1_CA"].index))
            if tmpnum>max_sat_qzsj1:
                max_sat_qzsj1=tmpnum
            if tmpnum<min_sat_qzsj1:
                min_sat_qzsj1=tmpnum
            tmpnum=len(dframe.drop(index=dframe.loc[dframe["SignalType"] != "QZS_L5_Q"].index))
            if tmpnum>max_sat_qzsj5:
                max_sat_qzsj5=tmpnum
            if tmpnum<min_sat_qzsj5:
                min_sat_qzsj5=tmpnum
            # if (idx>=1619481583000 and idx <=1619481600000) or (chunk_num>=130):
            #     checknow=1
            # In case gt_slice is empty, find closest ground truth
            if len(gt_slice) == 0:
                low_gt_half = gt.loc[gt['UnixTimeMillis'] <= idx]
                try:
                    low_diff = np.abs(low_gt_half.iloc[-1]['millisSinceGpsEpoch'] - idx)
                except IndexError:
                    low_diff = 10000
                high_gt_half = gt.loc[gt['millisSinceGpsEpoch'] >= idx]
                try:
                    high_diff = np.abs(high_gt_half.iloc[0]['millisSinceGpsEpoch'] - idx)
                except IndexError:
                    high_diff = 10000
                if low_diff < high_diff:
                    gt_slice = low_gt_half.iloc[[-1]]
                else:
                    gt_slice = high_gt_half.iloc[[0]]
            # gt_slice['b'] = solve_gt_b(dframe, gt_slice)
            gt_slice['b'] = solve_gt_b_GSDC2022(dframe, gt_slice) # ？
            gt.loc[gt_row, 'b'] = gt_slice['b']
            # gt_slice['Wls_bias']=solve_wls_b_GSDC2022(dframe, gt_slice)
            # self.baseline_df.loc[bl_row, 'bias_bl']=gt_slice['Wls_bias']
            # if len(dframe_L1)>0:
            #     gt_slice['Wls_bias']=solve_wls_b_GSDC2022(dframe_L1, gt_slice)
            # else:
            #     gt_slice['Wls_bias'] = 0.0
            if len(baseline_slice)>0:
                baseline_slice['bias_wls_igst']=solve_wls_b_GSDC2022(dframe, baseline_slice)
                baseline_slice['bias_KF_realtime']=solve_KF_realtime_b_GSDC2022(dframe, baseline_slice)
                self.baseline_df.loc[bl_row, 'bias_wls_igst']=baseline_slice['bias_wls_igst']
                self.baseline_df.loc[bl_row, 'bias_KF_realtime']=baseline_slice['bias_KF_realtime']
            else:
                bugt=1

            # Add ground truth to the measurement data frame
            shaped_ones = np.ones(len(dframe_multic))
            # .to_numpy() required because gt_slice is always a DataFrame (ensures receiver bias value is always a scalar)
            dframe_multic.loc[:, 'ecefX'] = gt_slice['ecefX'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'ecefY'] = gt_slice['ecefY'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'ecefZ'] = gt_slice['ecefZ'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'LatitudeDegrees'] = gt_slice['LatitudeDegrees'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'LongitudeDegrees'] = gt_slice['LongitudeDegrees'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'AltitudeMeters'] = gt_slice['AltitudeMeters'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'b'] = gt_slice['b'].to_numpy() * shaped_ones
            # dframe_multic.loc[:, 'Wls_bias'] = gt_slice['Wls_bias'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'XEcefMeters_kf_igst'] = baseline_slice['XEcefMeters_kf_igst'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'YEcefMeters_kf_igst'] = baseline_slice['YEcefMeters_kf_igst'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'ZEcefMeters_kf_igst'] = baseline_slice['ZEcefMeters_kf_igst'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'XEcefMeters_wls_igst'] = baseline_slice['XEcefMeters_wls_igst'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'YEcefMeters_wls_igst'] = baseline_slice['YEcefMeters_wls_igst'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'ZEcefMeters_wls_igst'] = baseline_slice['ZEcefMeters_wls_igst'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'XEcefMeters_KF_realtime'] = baseline_slice['XEcefMeters_KF_realtime'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'YEcefMeters_KF_realtime'] = baseline_slice['YEcefMeters_KF_realtime'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'ZEcefMeters_KF_realtime'] = baseline_slice['ZEcefMeters_KF_realtime'].to_numpy() * shaped_ones
            ecefxyz_kf_igst = baseline_slice[['XEcefMeters_kf_igst', 'YEcefMeters_kf_igst', 'ZEcefMeters_kf_igst']].to_numpy()
            ecefxyz_wls_igst = baseline_slice[['XEcefMeters_wls_igst', 'YEcefMeters_wls_igst', 'ZEcefMeters_wls_igst']].to_numpy()
            lla_wls_igst = coord.ecef2geodetic(ecefxyz_wls_igst).reshape(-1)
            lla_kf_igst = coord.ecef2geodetic(ecefxyz_kf_igst).reshape(-1)
            dframe_multic.loc[:, 'LatitudeDegrees_kf_igst'] = lla_kf_igst[0] * shaped_ones
            dframe_multic.loc[:, 'LongitudeDegrees_kf_igst'] = lla_kf_igst[1] * shaped_ones
            dframe_multic.loc[:, 'AltitudeMeters_kf_igst'] = lla_kf_igst[2] * shaped_ones
            dframe_multic.loc[:, 'LatitudeDegrees_wls_igst'] = lla_wls_igst[0] * shaped_ones
            dframe_multic.loc[:, 'LongitudeDegrees_wls_igst'] = lla_wls_igst[1] * shaped_ones
            dframe_multic.loc[:, 'AltitudeMeters_wls_igst'] = lla_wls_igst[2] * shaped_ones
            dframe_multic.loc[:, 'bias_wls_igst'] = baseline_slice['bias_wls_igst'].to_numpy() * shaped_ones
            dframe_multic.loc[:, 'bias_KF_realtime'] = baseline_slice['bias_KF_realtime'].to_numpy() * shaped_ones
            if dframe_multic.isnull().values.any():
                nan_record=pd.DataFrame({'col':dframe_multic.isnull().any()})
                nan_names=nan_record[nan_record.col==True].index.to_list()
                if 'ecefX' in nan_names or 'b' in nan_names or 'Wls_bias' in nan_names:
                    print(dframe_multic)
                    filer=open(self.out_root + '/env/nan_records_multic.txt','a+')
                    print(f'NaNs in DF columns {nan_names} at one epoch, sat num {len(dframe)}')
                    filer.write(f'{_path} NaNs in DF columns {nan_names} at one epoch, sat num {len(dframe)}, visible sat num {len(dframe_multic)}\n')
                    filer.close()
                    null_break=True
                    break
                    # raise ValueError('NaNs in DF at one epoch')
            chunk.append(dframe_multic)
            count += 1

        # recording sat nums
        filer = open(self.out_root + '/env/sat_num_records_multic.txt', 'a+')
        filer.write(f'{_path}, satnum, {min_sat_all}, {max_sat_all}, GPS_L1, {min_sat_gpsl1}, {max_sat_gpsl1}, '
                    f'GLO_G1, {min_sat_glog1}, {max_sat_glog1}, BDS_B1I, {min_sat_bdsb1i}, {max_sat_bdsb1i}, '
                    f'GAL_E1, {min_sat_gale1}, {max_sat_gale1}, GPS_L5, {min_sat_gpsl5}, {max_sat_gpsl5}, '
                    f'GAL_E5A, {min_sat_gale5a}, {max_sat_gale5a}, QZS_J1, {min_sat_qzsj1}, {max_sat_qzsj1}, '
                    f'QZS_J5, {min_sat_qzsj5}, {max_sat_qzsj5}, miss obs={noobscount}, less 4 satnum={less4sat_count}\n')
        filer.close()
        chunk_df = pd.concat(chunk)
        # record sat information
        sat_info = [count, min_sat_all,max_sat_all,min_sat_gpsl1,max_sat_gpsl1,min_sat_glog1,max_sat_glog1,min_sat_bdsb1i,max_sat_bdsb1i,
                    min_sat_bdsb1c,min_sat_bdsb1c,min_sat_bdsb2a,max_sat_bdsb2a,min_sat_gale1,max_sat_gale1,min_sat_gpsl5,max_sat_gpsl5,
                    min_sat_gale5a,max_sat_gale5a,min_sat_qzsj1,max_sat_qzsj1,min_sat_qzsj5,max_sat_qzsj5,noobscount,less4sat_count]
        return chunk_df,sat_info

    def getitemECEFCN0AA_RLKF(self, idx, biastrig, gnss_df, id_call):#, guess_XYZb=None
        # record sat num of all constellation, total sat num 53, in summary
        satnum=50
        gpsl1satnum=33
        #  GPS_L1_CA +0 , GPS_L5 +100
        #  BDS_B1I +200, BDS_B1C +300, BDS_B2AP +400,
        #  GLO_G1 +500
        #  GAL_E1 +600, GAL_E5A +700
        cons_name=['GPS_L1', 'GPS_L5', 'BDS_B1I', 'BDS_B1C', 'BDS_B2A', 'GLO_G1', 'GAL_E1', 'GAL_E5A']
        satname=[f'{cons_name[j]}_s{i}' for j in range(len(cons_name)) for i in range(1,satnum+1) ]
        sat_summary_multic=pd.DataFrame({'Svid':satname,
                                        'Nums':np.zeros(len(satname))})
        cons_map = [0,100,200,300,400,500,600,700]
        Satnum = [j+i for j in cons_map for i in range(1,satnum+1) ]
        sat_summary_multic['Satnum'] = pd.DataFrame({'Satnum':Satnum})

        pd_tmp=pd.DataFrame({'Svid':['Prrmax','Prrmin','LosXmax','LosXmin',
                                     'LosYmax','LosYmin','LosZmax','LosZmin',
                                     'Prrmean','Prrstd','LosXmean','LosXstd',
                                     'LosYmean','LosYstd','LosZmean','LosZstd',
                                     'Edgenummean','Edgenumstd','Edgenummax','Edgenummin',
                                     'CN0mean','CN0std','CN0max','CN0min',
                                     'PRUmean','PRUstd','PRUmax','PRUmin',
                                     'ClkBmean','ClkBstd','ClkBmax','ClkBmin',
                                     'PRratemean','PRratestd','PRratemax','PRratemin',
                                     'ADRmean','ADRstd','ADRmax','ADRmin',
                                     'ADRSmean','ADRSstd','ADRSmax','ADRSmin',
                                     'EAmean','EAstd','EAmax','EAmin',
                                     'AAmean','AAstd','AAmax','AAmin',
                                     'IonDmean','IonDstd','IonDmax','IonDmin',
                                     'TroDmean','TroDstd','TroDmax','TroDmin',
                                     ],
                                        'Nums':[-1e4,1e4,-1e4,1e4,-1e4,1e4,-1e4,1e4,
                                                -1e4,1e4,-1e4,1e4,-1e4,1e4,-1e4,1e4,
                                                0,0,1e-4,1e4,0,0,1e-4,1e4,0,0,1e-4,1e4,
                                                0,0,1e-4,1e4,0,0,1e-4,1e4,0,0,1e-4,1e4,
                                                0,0,1e-4,1e4,0,0,1e-4,1e4,0,0,1e-4,1e4,
                                                0,0,1e-4,1e4,0,0,1e-4,1e4,]})
        sat_summary_multic=sat_summary_multic.append(pd_tmp, ignore_index = True)

        # key, timestep = self.indices[idx]
        # key_file, times = self.get_files(key)
        # times=self.truth_df['UnixTimeMillis'].to_numpy()
        dftimes = gnss_df['utcTimeMillis'].copy().drop_duplicates(inplace=False)
        times = dftimes.to_numpy()
        featureall={}
        featureonlyall=[]
        Complementaryfeatureall = {}
        edgenumall=[]
        # 判断载波相位的周跳或初始状态
        gnss_df = carrier_smoothing(gnss_df)
        # 检查载波频率误差
        CarrierFrequencyHzRef = gnss_df.groupby(['Svid', 'SignalType'])['CarrierFrequencyHz'].median()
        gnss_df = gnss_df.merge(CarrierFrequencyHzRef, how='left', on=['Svid', 'SignalType'], suffixes=('', 'Ref'))
        gnss_df['CarrierErrorHz'] = np.abs((gnss_df['CarrierFrequencyHz'] - gnss_df['CarrierFrequencyHzRef']))
        for timestep in times:
            try:
                data = gnss_df[gnss_df['millisSinceGpsEpoch'] == timestep]
                data = satellite_selection(data, 'pr_smooth')
                if data.empty:
                    data = gnss_df[gnss_df['millisSinceGpsEpoch'] == timestep]
                    _data0 = data.iloc[0]
                else:
                    _data0 = data.iloc[0]
            except:
                raise ValueError('idx not found in dataset')

            data_svid=np.array(data['Svid'])#data['Svid'].to_numpy
            #  GPS_L1_CA +0 , GPS_L5 +100
            #  BDS_B1I +200, BDS_B1C +300, BDS_B2AP +400,
            #  GLO_G1 +500
            #  GAL_E1 +600, GAL_E5A +700
            for id in data_svid:
                if id<100:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'GPS_L1_s{id}', 'Nums'] += 1
                elif id<200:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'GPS_L5_s{id-100}', 'Nums'] += 1
                elif id<300:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'BDS_B1I_s{id-200}', 'Nums'] += 1
                elif id<400:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'BDS_B1C_s{id-300}', 'Nums'] += 1
                elif id<500:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'BDS_B2AP_s{id-400}', 'Nums'] += 1
                elif id<600:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'GLO_G1_s{id-500}', 'Nums'] += 1
                elif id<700:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'GAL_E1_s{id-600}', 'Nums'] += 1
                elif id<800:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'GAL_E5A_s{id-700}', 'Nums'] += 1

            # Select random initialization
            true_XYZb = np.array([_data0['ecefX'], _data0['ecefY'], _data0['ecefZ'], _data0['b']])
            # if guess_XYZb is None:
            #     guess_XYZb = self.add_guess_noise(true_XYZb)  # Generate guess by adding noise to groundtruth
            #         guess_XYZb = np.copy(true_XYZb)         # 0 noise for debugging
            # guess_XYZb= np.array([_data0['WlsPositionXEcefMeters'], _data0['WlsPositionYEcefMeters'],
            #                       _data0['WlsPositionZEcefMeters'], _data0['Wls_bias']])
            guess_XYZb= np.array([_data0['XEcefMeters_wls_igst'], _data0['YEcefMeters_wls_igst'],
                                  _data0['ZEcefMeters_wls_igst'], _data0['bias_wls_igst']])

            # Primary feature extraction
            # expected_pseudo, satXYZV = expected_measurements(data, guess_XYZb)
            if biastrig==1:
                expected_pseudo, satXYZV = expected_measurements_GSDC2022(data, guess_XYZb)
            elif biastrig==0:
                expected_pseudo, satXYZV = expected_measurements_GSDC2022_withoutbias(data, guess_XYZb)

            residuals = (data['PrM'] - expected_pseudo).to_numpy()
            los_vector = (satXYZV[['x', 'y', 'z']] - guess_XYZb[:3])
            los_vector = los_vector.div(np.sqrt(np.square(los_vector).sum(axis=1)), axis='rows').to_numpy()
            # los_vector = ref_local.ecef2nedv(los_vector)
            CN0=data['Cn0DbHz'].to_numpy()
            RawPseudorangeUncertaintyMeters=data['RawPseudorangeUncertaintyMeters'].to_numpy() # 伪距不确定性
            AA=data['SvAzimuthDegrees'].to_numpy() # 卫星方位角
            EA=data['SvElevationDegrees'].to_numpy() # 高度角
            PRrate=data['PseudorangeRateMetersPerSecond'].to_numpy() # 伪距速率,伪距变化率
            ADR=data['AccumulatedDeltaRangeMeters'].to_numpy() # 累积增量范围
            ClkB=data['SvClockBiasMeters'].to_numpy() # 卫星钟差
            IonD=data['IonosphericDelayMeters'].to_numpy() # 电离层延迟误差
            TroD=data['TroposphericDelayMeters'].to_numpy() # 大气层延迟误差
            ADRS=data['slip_state'].to_numpy() # 累积增量范围状态
            Prr=data['PrrM'].to_numpy() # 伪距速率
            clockbias = np.ones(len(data)) * true_XYZb[3] # 所有卫星系统统一钟差，可能会有误差
            satecef = satXYZV[['x', 'y', 'z']].to_numpy()
            mutipath=data['MultipathIndicator'].to_numpy()
            TDCP = data['TDCP'].to_numpy()

            features = np.concatenate((np.reshape(residuals, [-1, 1]), los_vector, np.reshape(CN0, [-1, 1]),
                                       np.reshape(RawPseudorangeUncertaintyMeters, [-1, 1]),
                                       np.reshape(AA, [-1, 1]), np.reshape(EA, [-1, 1])), axis=1)

            # 补充的一些特征：0载波测量，1TDCP, 2载波状态，3伪距速率，4预测钟差，5-7卫星位置
            Complementaryfeature = np.concatenate((np.reshape(ADR, [-1, 1]),np.reshape(TDCP, [-1, 1]),
                                                   np.reshape(ADRS, [-1, 1]),np.reshape(Prr, [-1, 1]),
                                                   np.reshape(clockbias, [-1, 1]),satecef), axis=1)

            featureonlyall.append(features)
            if id_call:
                features=np.insert(features, 0, values=np.array(data['Svid']), axis=1)
                Complementaryfeature=np.insert(Complementaryfeature, 0, values=np.array(data['Svid']), axis=1)
                Complementaryfeature=Complementaryfeature[Complementaryfeature[:,2]!=1] # 排除周跳状态卫星测量

            #multi c edge_thre=5
            edge_thre = 5
            max_visible_sat = 53
            ############
            values=torch.tensor(features[:,1]) #features[:,1] check on 20230421, with satid being 1
            diffs = np.abs(values[:, None] - values[None, :])
            diagvalues=torch.diag_embed(torch.tensor(np.ones([diffs.shape[0],])*100))
            diffs = diffs+diagvalues
            edge_index = torch.nonzero(diffs <= edge_thre, as_tuple=False).t()
            values_id=features[:,0]
            edges = []
            n=len(values_id)
            for i in range(n):
                tmp=values_id.copy()
                tmp[i]=1e4
                for j in range(n):
                    if values_id[i] < 100 and tmp[j] < 100:
                        edges.append([i, j])
                        edges.append([j, i])
                    elif 100< values_id[i] < 200 and 100< tmp[j] < 200:
                        edges.append((i, j))
                        edges.append((j, i))
                    elif 200< values_id[i] < 300 and 200< tmp[j] < 300:
                        edges.append((i, j))
                        edges.append((j, i))
                    elif 300< values_id[i] < 400 and 300< tmp[j] < 400:
                        edges.append((i, j))
                        edges.append((j, i))
                    elif 400< values_id[i] < 500 and 400< tmp[j] < 500:
                        edges.append((i, j))
                        edges.append((j, i))
                    elif 500< values_id[i] < 600 and 500< tmp[j] < 600:
                        edges.append((i, j))
                        edges.append((j, i))
                    elif 600< values_id[i] < 700 and 600< tmp[j] < 700:
                        edges.append((i, j))
                        edges.append((j, i))
                    elif 700< values_id[i] < 800 and 700< tmp[j] < 700:
                        edges.append((i, j))
                        edges.append((j, i))
            edge_index_cons = np.array(edges).T
            try:
                edge_index_all = np.concatenate((edge_index,edge_index_cons),axis=1)
            except:
                edge_index_all = edge_index
            edge_index_unique = np.unique(edge_index_all, axis=1, return_index=True)
            edge_index_m = edge_index_unique[0]
            edgenumall.append(edge_index_m.shape[-1])

            # edge_size=edge_index_m.shape[-1]
            # edge_index_shape = np.zeros([2, max_visible_sat*(max_visible_sat-1)])
            # edge_index_shape[:,:(edge_size)] = edge_index_m

            sample = {
                'features': features, #torch.Tensor(features)
                'true_correction': (true_XYZb - guess_XYZb)[:3],
                'guess': guess_XYZb,
                'edge_index': edge_index_m
            }
            featureall[timestep]=sample
            Complementaryfeatureall[timestep] = Complementaryfeature
        # if self.transform is not None:
        #     sample = self.transform(sample)
        # record
        featureonlyall=np.concatenate(featureonlyall)
        edgenumall=np.array(edgenumall)
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Prrmax', 'Nums']=max(featureonlyall[:,0])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Prrmin', 'Nums']=min(featureonlyall[:,0])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosXmax', 'Nums']=max(featureonlyall[:,1])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosXmin', 'Nums']=min(featureonlyall[:,1])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosYmax', 'Nums']=max(featureonlyall[:,2])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosYmin', 'Nums']=min(featureonlyall[:,2])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosZmax', 'Nums']=max(featureonlyall[:,3])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosZmin', 'Nums']=min(featureonlyall[:,3])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Prrmean', 'Nums']=np.mean(featureonlyall[:,0])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Prrstd', 'Nums']=np.std(featureonlyall[:,0])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosXmean', 'Nums']=np.mean(featureonlyall[:,1])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosXstd', 'Nums']=np.std(featureonlyall[:,1])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosYmean', 'Nums']=np.mean(featureonlyall[:,2])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosYstd', 'Nums']=np.std(featureonlyall[:,2])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosZmean', 'Nums']=np.mean(featureonlyall[:,3])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosZstd', 'Nums']=np.std(featureonlyall[:,3])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Edgenummean', 'Nums']=np.mean(edgenumall)
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Edgenumstd', 'Nums']=np.std(edgenumall)
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Edgenummax', 'Nums']=max(edgenumall)
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Edgenummin', 'Nums']=min(edgenumall)
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'CN0mean', 'Nums']=np.mean(featureonlyall[:,4])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'CN0std', 'Nums']=np.std(featureonlyall[:,4])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'CN0max', 'Nums']=max(featureonlyall[:,4])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'CN0min', 'Nums']=min(featureonlyall[:,4])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRUmean', 'Nums']=np.mean(featureonlyall[:,5])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRUstd', 'Nums']=np.std(featureonlyall[:,5])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRUmax', 'Nums']=max(featureonlyall[:,5])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRUmin', 'Nums']=min(featureonlyall[:,5])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'AAmean', 'Nums']=np.mean(featureonlyall[:,6])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'AAstd', 'Nums']=np.std(featureonlyall[:,6])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'AAmax', 'Nums']=max(featureonlyall[:,6])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'AAmin', 'Nums']=min(featureonlyall[:,6])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'EAmean', 'Nums']=np.mean(featureonlyall[:,7])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'EAstd', 'Nums']=np.std(featureonlyall[:,7])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'EAmax', 'Nums']=max(featureonlyall[:,7])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'EAmin', 'Nums']=min(featureonlyall[:,7])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRratemean', 'Nums']=np.mean(featureonlyall[:,8])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRratestd', 'Nums']=np.std(featureonlyall[:,8])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRratemax', 'Nums']=max(featureonlyall[:,8])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRratemin', 'Nums']=min(featureonlyall[:,8])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRmean', 'Nums']=np.mean(featureonlyall[:,9])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRstd', 'Nums']=np.std(featureonlyall[:,9])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRmax', 'Nums']=max(featureonlyall[:,9])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRmin', 'Nums']=min(featureonlyall[:,9])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ClkBmean', 'Nums']=np.mean(featureonlyall[:,10])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ClkBstd', 'Nums']=np.std(featureonlyall[:,10])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ClkBmax', 'Nums']=max(featureonlyall[:,10])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ClkBmin', 'Nums']=min(featureonlyall[:,10])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'IonDmean', 'Nums']=np.mean(featureonlyall[:,11])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'IonDstd', 'Nums']=np.std(featureonlyall[:,11])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'IonDmax', 'Nums']=max(featureonlyall[:,11])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'IonDmin', 'Nums']=min(featureonlyall[:,11])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'TroDmean', 'Nums']=np.mean(featureonlyall[:,12])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'TroDstd', 'Nums']=np.std(featureonlyall[:,12])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'TroDmax', 'Nums']=max(featureonlyall[:,12])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'TroDmin', 'Nums']=min(featureonlyall[:,12])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRSmean', 'Nums']=np.mean(featureonlyall[:,13])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRSstd', 'Nums']=np.std(featureonlyall[:,13])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRSmax', 'Nums']=max(featureonlyall[:,13])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRSmin', 'Nums']=min(featureonlyall[:,13])
        return featureall, sat_summary_multic, Complementaryfeatureall

    def getitemECEFCN0AA_KF_realtime(self, idx, biastrig, gnss_df, id_call):#, guess_XYZb=None
        # record sat num of all constellation, total sat num 53, in summary
        satnum=50
        gpsl1satnum=33
        #  GPS_L1_CA +0 , GPS_L5 +100
        #  BDS_B1I +200, BDS_B1C +300, BDS_B2AP +400,
        #  GLO_G1 +500
        #  GAL_E1 +600, GAL_E5A +700
        cons_name=['GPS_L1', 'GPS_L5', 'BDS_B1I', 'BDS_B1C', 'BDS_B2A', 'GLO_G1', 'GAL_E1', 'GAL_E5A']
        satname=[f'{cons_name[j]}_s{i}' for j in range(len(cons_name)) for i in range(1,satnum+1) ]
        sat_summary_multic=pd.DataFrame({'Svid':satname,
                                        'Nums':np.zeros(len(satname))})
        cons_map = [0,100,200,300,400,500,600,700]
        Satnum = [j+i for j in cons_map for i in range(1,satnum+1) ]
        sat_summary_multic['Satnum'] = pd.DataFrame({'Satnum':Satnum})

        pd_tmp=pd.DataFrame({'Svid':['Prrmax','Prrmin','LosXmax','LosXmin',
                                     'LosYmax','LosYmin','LosZmax','LosZmin',
                                     'Prrmean','Prrstd','LosXmean','LosXstd',
                                     'LosYmean','LosYstd','LosZmean','LosZstd',
                                     'Edgenummean','Edgenumstd','Edgenummax','Edgenummin',
                                     'CN0mean','CN0std','CN0max','CN0min',
                                     'PRUmean','PRUstd','PRUmax','PRUmin',
                                     'ClkBmean','ClkBstd','ClkBmax','ClkBmin',
                                     'PRratemean','PRratestd','PRratemax','PRratemin',
                                     'ADRmean','ADRstd','ADRmax','ADRmin',
                                     'ADRSmean','ADRSstd','ADRSmax','ADRSmin',
                                     'EAmean','EAstd','EAmax','EAmin',
                                     'AAmean','AAstd','AAmax','AAmin',
                                     'IonDmean','IonDstd','IonDmax','IonDmin',
                                     'TroDmean','TroDstd','TroDmax','TroDmin',
                                     ],
                                        'Nums':[-1e4,1e4,-1e4,1e4,-1e4,1e4,-1e4,1e4,
                                                -1e4,1e4,-1e4,1e4,-1e4,1e4,-1e4,1e4,
                                                0,0,1e-4,1e4,0,0,1e-4,1e4,0,0,1e-4,1e4,
                                                0,0,1e-4,1e4,0,0,1e-4,1e4,0,0,1e-4,1e4,
                                                0,0,1e-4,1e4,0,0,1e-4,1e4,0,0,1e-4,1e4,
                                                0,0,1e-4,1e4,0,0,1e-4,1e4,]})
        sat_summary_multic=sat_summary_multic.append(pd_tmp, ignore_index = True)

        # key, timestep = self.indices[idx]
        # key_file, times = self.get_files(key)
        # times=self.truth_df['UnixTimeMillis'].to_numpy()
        dftimes = gnss_df['utcTimeMillis'].copy().drop_duplicates(inplace=False)
        times = dftimes.to_numpy()
        featureall={}
        featureonlyall=[]
        edgenumall=[]
        for timestep in times:
            try:
                data = gnss_df[gnss_df['millisSinceGpsEpoch'] == timestep]
                _data0 = data.iloc[0]
            except:
                raise ValueError('idx not found in dataset')

            data_svid=np.array(data['Svid'])#data['Svid'].to_numpy
            #  GPS_L1_CA +0 , GPS_L5 +100
            #  BDS_B1I +200, BDS_B1C +300, BDS_B2AP +400,
            #  GLO_G1 +500
            #  GAL_E1 +600, GAL_E5A +700
            for id in data_svid:
                if id<100:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'GPS_L1_s{id}', 'Nums'] += 1
                elif id<200:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'GPS_L5_s{id-100}', 'Nums'] += 1
                elif id<300:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'BDS_B1I_s{id-200}', 'Nums'] += 1
                elif id<400:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'BDS_B1C_s{id-300}', 'Nums'] += 1
                elif id<500:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'BDS_B2AP_s{id-400}', 'Nums'] += 1
                elif id<600:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'GLO_G1_s{id-500}', 'Nums'] += 1
                elif id<700:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'GAL_E1_s{id-600}', 'Nums'] += 1
                elif id<800:
                    sat_summary_multic.loc[sat_summary_multic['Svid'] == f'GAL_E5A_s{id-700}', 'Nums'] += 1

            # Select random initialization
            true_XYZb = np.array([_data0['ecefX'], _data0['ecefY'], _data0['ecefZ'], _data0['b']])
            # if guess_XYZb is None:
            #     guess_XYZb = self.add_guess_noise(true_XYZb)  # Generate guess by adding noise to groundtruth
            #         guess_XYZb = np.copy(true_XYZb)         # 0 noise for debugging
            # guess_XYZb= np.array([_data0['WlsPositionXEcefMeters'], _data0['WlsPositionYEcefMeters'],
            #                       _data0['WlsPositionZEcefMeters'], _data0['Wls_bias']])
            guess_XYZb= np.array([_data0['XEcefMeters_KF_realtime'], _data0['YEcefMeters_KF_realtime'],
                                  _data0['ZEcefMeters_KF_realtime'], _data0['bias_KF_realtime']])

            # Primary feature extraction
            # expected_pseudo, satXYZV = expected_measurements(data, guess_XYZb)
            if biastrig==1:
                expected_pseudo, satXYZV = expected_measurements_GSDC2022(data, guess_XYZb)
            elif biastrig==0:
                expected_pseudo, satXYZV = expected_measurements_GSDC2022_withoutbias(data, guess_XYZb)

            residuals = (data['PrM'] - expected_pseudo).to_numpy()
            los_vector = (satXYZV[['x', 'y', 'z']] - guess_XYZb[:3])
            los_vector = los_vector.div(np.sqrt(np.square(los_vector).sum(axis=1)), axis='rows').to_numpy()
            # los_vector = ref_local.ecef2nedv(los_vector)
            CN0=data['Cn0DbHz'].to_numpy()
            RawPseudorangeUncertaintyMeters=data['RawPseudorangeUncertaintyMeters'].to_numpy() # 伪距不确定性
            AA=data['SvAzimuthDegrees'].to_numpy() # 卫星方位角
            EA=data['SvElevationDegrees'].to_numpy() # 高度角
            PRrate=data['PseudorangeRateMetersPerSecond'].to_numpy() # 伪距速率,伪距变化率
            ADR=data['AccumulatedDeltaRangeMeters'].to_numpy() # 累积增量范围
            ClkB=data['SvClockBiasMeters'].to_numpy() # 卫星钟差
            IonD=data['IonosphericDelayMeters'].to_numpy() # 电离层延迟误差
            TroD=data['TroposphericDelayMeters'].to_numpy() # 大气层延迟误差
            ADRS=data['AccumulatedDeltaRangeState'].to_numpy() # 累积增量范围状态

            features = np.concatenate((np.reshape(residuals, [-1, 1]), los_vector, np.reshape(CN0, [-1, 1]),
                                       np.reshape(RawPseudorangeUncertaintyMeters, [-1, 1]),
                                       np.reshape(AA, [-1, 1]), np.reshape(EA, [-1, 1])), axis=1)
            featureonlyall.append(features)
            if id_call:
                features=np.insert(features, 0, values=np.array(data['Svid']), axis=1)

            #multi c edge_thre=5
            edge_thre = 5
            max_visible_sat = 53
            ############
            values=torch.tensor(features[:,1]) #features[:,1] check on 20230421, with satid being 1
            diffs = np.abs(values[:, None] - values[None, :])
            diagvalues=torch.diag_embed(torch.tensor(np.ones([diffs.shape[0],])*100))
            diffs = diffs+diagvalues
            edge_index = torch.nonzero(diffs <= edge_thre, as_tuple=False).t()
            values_id=features[:,0]
            edges = []
            n=len(values_id)
            for i in range(n):
                tmp=values_id.copy()
                tmp[i]=1e4
                for j in range(n):
                    if values_id[i] < 100 and tmp[j] < 100:
                        edges.append([i, j])
                        edges.append([j, i])
                    elif 100< values_id[i] < 200 and 100< tmp[j] < 200:
                        edges.append((i, j))
                        edges.append((j, i))
                    elif 200< values_id[i] < 300 and 200< tmp[j] < 300:
                        edges.append((i, j))
                        edges.append((j, i))
                    elif 300< values_id[i] < 400 and 300< tmp[j] < 400:
                        edges.append((i, j))
                        edges.append((j, i))
                    elif 400< values_id[i] < 500 and 400< tmp[j] < 500:
                        edges.append((i, j))
                        edges.append((j, i))
                    elif 500< values_id[i] < 600 and 500< tmp[j] < 600:
                        edges.append((i, j))
                        edges.append((j, i))
                    elif 600< values_id[i] < 700 and 600< tmp[j] < 700:
                        edges.append((i, j))
                        edges.append((j, i))
                    elif 700< values_id[i] < 800 and 700< tmp[j] < 700:
                        edges.append((i, j))
                        edges.append((j, i))
            edge_index_cons = np.array(edges).T
            edge_index_all = np.concatenate((edge_index,edge_index_cons),axis=1)
            edge_index_unique = np.unique(edge_index_all, axis=1, return_index=True)
            edge_index_m = edge_index_unique[0]
            edgenumall.append(edge_index_m.shape[-1])

            # edge_size=edge_index_m.shape[-1]
            # edge_index_shape = np.zeros([2, max_visible_sat*(max_visible_sat-1)])
            # edge_index_shape[:,:(edge_size)] = edge_index_m

            sample = {
                'features': features, #torch.Tensor(features)
                'true_correction': (true_XYZb - guess_XYZb)[:3],
                'guess': guess_XYZb,
                'edge_index': edge_index_m
            }
            featureall[timestep]=sample

        # if self.transform is not None:
        #     sample = self.transform(sample)
        # record
        featureonlyall=np.concatenate(featureonlyall)
        edgenumall=np.array(edgenumall)
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Prrmax', 'Nums']=max(featureonlyall[:,0])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Prrmin', 'Nums']=min(featureonlyall[:,0])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosXmax', 'Nums']=max(featureonlyall[:,1])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosXmin', 'Nums']=min(featureonlyall[:,1])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosYmax', 'Nums']=max(featureonlyall[:,2])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosYmin', 'Nums']=min(featureonlyall[:,2])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosZmax', 'Nums']=max(featureonlyall[:,3])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosZmin', 'Nums']=min(featureonlyall[:,3])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Prrmean', 'Nums']=np.mean(featureonlyall[:,0])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Prrstd', 'Nums']=np.std(featureonlyall[:,0])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosXmean', 'Nums']=np.mean(featureonlyall[:,1])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosXstd', 'Nums']=np.std(featureonlyall[:,1])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosYmean', 'Nums']=np.mean(featureonlyall[:,2])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosYstd', 'Nums']=np.std(featureonlyall[:,2])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosZmean', 'Nums']=np.mean(featureonlyall[:,3])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'LosZstd', 'Nums']=np.std(featureonlyall[:,3])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Edgenummean', 'Nums']=np.mean(edgenumall)
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Edgenumstd', 'Nums']=np.std(edgenumall)
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Edgenummax', 'Nums']=max(edgenumall)
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'Edgenummin', 'Nums']=min(edgenumall)
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'CN0mean', 'Nums']=np.mean(featureonlyall[:,4])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'CN0std', 'Nums']=np.std(featureonlyall[:,4])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'CN0max', 'Nums']=max(featureonlyall[:,4])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'CN0min', 'Nums']=min(featureonlyall[:,4])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRUmean', 'Nums']=np.mean(featureonlyall[:,5])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRUstd', 'Nums']=np.std(featureonlyall[:,5])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRUmax', 'Nums']=max(featureonlyall[:,5])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRUmin', 'Nums']=min(featureonlyall[:,5])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'AAmean', 'Nums']=np.mean(featureonlyall[:,6])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'AAstd', 'Nums']=np.std(featureonlyall[:,6])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'AAmax', 'Nums']=max(featureonlyall[:,6])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'AAmin', 'Nums']=min(featureonlyall[:,6])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'EAmean', 'Nums']=np.mean(featureonlyall[:,7])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'EAstd', 'Nums']=np.std(featureonlyall[:,7])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'EAmax', 'Nums']=max(featureonlyall[:,7])
        sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'EAmin', 'Nums']=min(featureonlyall[:,7])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRratemean', 'Nums']=np.mean(featureonlyall[:,8])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRratestd', 'Nums']=np.std(featureonlyall[:,8])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRratemax', 'Nums']=max(featureonlyall[:,8])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'PRratemin', 'Nums']=min(featureonlyall[:,8])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRmean', 'Nums']=np.mean(featureonlyall[:,9])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRstd', 'Nums']=np.std(featureonlyall[:,9])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRmax', 'Nums']=max(featureonlyall[:,9])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRmin', 'Nums']=min(featureonlyall[:,9])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ClkBmean', 'Nums']=np.mean(featureonlyall[:,10])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ClkBstd', 'Nums']=np.std(featureonlyall[:,10])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ClkBmax', 'Nums']=max(featureonlyall[:,10])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ClkBmin', 'Nums']=min(featureonlyall[:,10])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'IonDmean', 'Nums']=np.mean(featureonlyall[:,11])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'IonDstd', 'Nums']=np.std(featureonlyall[:,11])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'IonDmax', 'Nums']=max(featureonlyall[:,11])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'IonDmin', 'Nums']=min(featureonlyall[:,11])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'TroDmean', 'Nums']=np.mean(featureonlyall[:,12])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'TroDstd', 'Nums']=np.std(featureonlyall[:,12])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'TroDmax', 'Nums']=max(featureonlyall[:,12])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'TroDmin', 'Nums']=min(featureonlyall[:,12])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRSmean', 'Nums']=np.mean(featureonlyall[:,13])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRSstd', 'Nums']=np.std(featureonlyall[:,13])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRSmax', 'Nums']=max(featureonlyall[:,13])
        # sat_summary_multic.loc[sat_summary_multic.loc[:, 'Svid'] == 'ADRSmin', 'Nums']=min(featureonlyall[:,13])

        return featureall, sat_summary_multic,

def solve_gt_b_GSDC2022(dframe, gt_slice, max_iter=20, tol=1e-3):
    gt_ecef = np.zeros([1, 3])
    gt_ecef[0, 0] = gt_slice["ecefX"]
    gt_ecef[0, 1] = gt_slice["ecefY"]
    gt_ecef[0, 2] = gt_slice["ecefZ"]

    sat_pos = np.hstack((dframe["SvPositionXEcefMeters"].to_numpy().reshape(-1, 1),
                         dframe["SvPositionYEcefMeters"].to_numpy().reshape(-1, 1),
                         dframe["SvPositionZEcefMeters"].to_numpy().reshape(-1, 1)))
    gt_pos = np.tile(gt_ecef, (sat_pos.shape[0], 1)) #沿着第行轴复制sat_pos.shape[0]次，沿着第1轴（列轴）复制1次

    gt_ranges = np.linalg.norm(gt_pos - sat_pos, axis=1) # 计算各个卫星的LOS

    G = np.ones((sat_pos.shape[0], 1))
    #     G[:,:3] = np.divide(gt_pos - sat_pos,gt_ranges.reshape(-1,1))

    W = np.diag(1. / dframe["RawPseudorangeUncertaintyMeters"] ** 2)
    rb = 0.
    for _ in range(max_iter):
        rho_diff = dframe["PrM"].to_numpy() - gt_ranges - rb

        rho_diff = rho_diff.reshape(-1, 1)

        delta = np.linalg.pinv(W.dot(G)).dot(W).dot(rho_diff)
        rb_new = rb + delta
        rb = rb_new
        if np.abs(delta) < tol:
            break
    return rb_new

# def solve_wls_b_GSDC2022(dframe, gt_slice, max_iter=20, tol=1e-3):
#     gt_ecef = np.zeros([1, 3])
#     gt_ecef[0, 0] = dframe["WlsPositionXEcefMeters"][dframe.index[0]]
#     gt_ecef[0, 1] = dframe["WlsPositionYEcefMeters"][dframe.index[0]]
#     gt_ecef[0, 2] = dframe["WlsPositionZEcefMeters"][dframe.index[0]]
#
#     sat_pos = np.hstack((dframe["SvPositionXEcefMeters"].to_numpy().reshape(-1, 1),
#                          dframe["SvPositionYEcefMeters"].to_numpy().reshape(-1, 1),
#                          dframe["SvPositionZEcefMeters"].to_numpy().reshape(-1, 1)))
#     gt_pos = np.tile(gt_ecef, (sat_pos.shape[0], 1))
#
#     gt_ranges = np.linalg.norm(gt_pos - sat_pos, axis=1)
#
#     G = np.ones((sat_pos.shape[0], 1))
#     #     G[:,:3] = np.divide(gt_pos - sat_pos,gt_ranges.reshape(-1,1))
#
#     W = np.diag(1. / dframe["RawPseudorangeUncertaintyMeters"] ** 2)
#     rb = 0.
#     for _ in range(max_iter):
#         rho_diff = dframe["PrM"].to_numpy() \
#                    - gt_ranges \
#                    - rb
#
#         rho_diff = rho_diff.reshape(-1, 1)
#
#         delta = np.linalg.pinv(W.dot(G)).dot(W).dot(rho_diff)
#         rb_new = rb + delta
#         rb = rb_new
#         if np.abs(delta) < tol:
#             break
#     return rb_new

def solve_wls_b_GSDC2022(dframe, baseline_df, max_iter=20, tol=1e-3):
    gt_ecef = np.zeros([1, 3])
    gt_ecef[0, 0] = baseline_df["XEcefMeters_wls_igst"]
    gt_ecef[0, 1] = baseline_df["YEcefMeters_wls_igst"]
    gt_ecef[0, 2] = baseline_df["ZEcefMeters_wls_igst"]

    sat_pos = np.hstack((dframe["SvPositionXEcefMeters"].to_numpy().reshape(-1, 1),
                         dframe["SvPositionYEcefMeters"].to_numpy().reshape(-1, 1),
                         dframe["SvPositionZEcefMeters"].to_numpy().reshape(-1, 1)))
    gt_pos = np.tile(gt_ecef, (sat_pos.shape[0], 1))

    gt_ranges = np.linalg.norm(gt_pos - sat_pos, axis=1)

    G = np.ones((sat_pos.shape[0], 1))
    #     G[:,:3] = np.divide(gt_pos - sat_pos,gt_ranges.reshape(-1,1))

    W = np.diag(1. / dframe["RawPseudorangeUncertaintyMeters"] ** 2)
    rb = 0.
    for _ in range(max_iter):
        rho_diff = dframe["PrM"].to_numpy() \
                   - gt_ranges \
                   - rb

        rho_diff = rho_diff.reshape(-1, 1)

        delta = np.linalg.pinv(W.dot(G)).dot(W).dot(rho_diff)
        rb_new = rb + delta
        rb = rb_new
        if np.abs(delta) < tol:
            break
    return rb_new

def solve_KF_realtime_b_GSDC2022(dframe, baseline_df, max_iter=20, tol=1e-3):
    gt_ecef = np.zeros([1, 3])
    gt_ecef[0, 0] = baseline_df["XEcefMeters_KF_realtime"]
    gt_ecef[0, 1] = baseline_df["YEcefMeters_KF_realtime"]
    gt_ecef[0, 2] = baseline_df["ZEcefMeters_KF_realtime"]

    sat_pos = np.hstack((dframe["SvPositionXEcefMeters"].to_numpy().reshape(-1, 1),
                         dframe["SvPositionYEcefMeters"].to_numpy().reshape(-1, 1),
                         dframe["SvPositionZEcefMeters"].to_numpy().reshape(-1, 1)))
    gt_pos = np.tile(gt_ecef, (sat_pos.shape[0], 1))

    gt_ranges = np.linalg.norm(gt_pos - sat_pos, axis=1)

    G = np.ones((sat_pos.shape[0], 1))
    #     G[:,:3] = np.divide(gt_pos - sat_pos,gt_ranges.reshape(-1,1))

    W = np.diag(1. / dframe["RawPseudorangeUncertaintyMeters"] ** 2)
    rb = 0.
    for _ in range(max_iter):
        rho_diff = dframe["PrM"].to_numpy() \
                   - gt_ranges \
                   - rb

        rho_diff = rho_diff.reshape(-1, 1)

        delta = np.linalg.pinv(W.dot(G)).dot(W).dot(rho_diff)
        rb_new = rb + delta
        rb = rb_new
        if np.abs(delta) < tol:
            break
    return rb_new

def expected_measurements_GSDC2022(dframe, guess_XYZb):
    satX = dframe.loc[:, "SvPositionXEcefMeters"].to_numpy()
    satY = dframe.loc[:, "SvPositionYEcefMeters"].to_numpy()
    satZ = dframe.loc[:, "SvPositionZEcefMeters"].to_numpy()
    satvX = dframe.loc[:, "SvVelocityXEcefMetersPerSecond"].to_numpy()
    satvY = dframe.loc[:, "SvVelocityYEcefMetersPerSecond"].to_numpy()
    satvZ = dframe.loc[:, "SvVelocityZEcefMetersPerSecond"].to_numpy()
    gt_ranges = np.sqrt((satX - guess_XYZb[0]) ** 2 \
                        + (satY - guess_XYZb[1]) ** 2 \
                        + (satZ - guess_XYZb[2]) ** 2)
    expected_rho = gt_ranges + guess_XYZb[3] ## calculate expected_pseudorange with calculated bias
    #     gt_ranges = gt_ranges.values
    satXYZV = pd.DataFrame()
    satXYZV['x'] = satX
    satXYZV['y'] = satY
    satXYZV['z'] = satZ
    satXYZV['vx'] = satvX
    satXYZV['vy'] = satvY
    satXYZV['vz'] = satvZ
    return expected_rho, satXYZV

def expected_measurements_GSDC2022_withoutbias(dframe, guess_XYZb):
    satX = dframe.loc[:, "SvPositionXEcefMeters"].to_numpy()
    satY = dframe.loc[:, "SvPositionYEcefMeters"].to_numpy()
    satZ = dframe.loc[:, "SvPositionZEcefMeters"].to_numpy()
    satvX = dframe.loc[:, "SvVelocityXEcefMetersPerSecond"].to_numpy()
    satvY = dframe.loc[:, "SvVelocityYEcefMetersPerSecond"].to_numpy()
    satvZ = dframe.loc[:, "SvVelocityZEcefMetersPerSecond"].to_numpy()
    gt_ranges = np.sqrt((satX - guess_XYZb[0]) ** 2 \
                        + (satY - guess_XYZb[1]) ** 2 \
                        + (satZ - guess_XYZb[2]) ** 2)
    #     gt_ranges = gt_ranges.values
    expected_rho = gt_ranges #+ guess_XYZb[3] ## calculate expected_pseudorange with calculated bias
    satXYZV = pd.DataFrame()
    satXYZV['x'] = satX
    satXYZV['y'] = satY
    satXYZV['z'] = satZ
    satXYZV['vx'] = satvX
    satXYZV['vy'] = satvY
    satXYZV['vz'] = satvZ
    return expected_rho, satXYZV

def carrier_smoothing(gnss_df):
    '''
    Computes pseudorange smoothing using carrier phase. See https://gssc.esa.int/navipedia/index.php/Carrier-smoothing_of_code_pseudoranges

            Parameters:
                    gnss_df (pandas.DataFrame): Pandas DataFrame containing pseudorange and carrier phase measurements
            Returns:
                    df (pandas.DataFrame): Pandas DataFrame with carrier-smoothing pseudorange 'pr_smooth'
    '''

    carr_th = 1.5  # carrier phase jump threshold [m] ** 2.0 -> 1.5 **
    pr_th = 20.0  # pseudorange jump threshold [m]

    prsmooth = np.full_like(gnss_df['RawPseudorangeMeters'], np.nan)
    slip_state = np.full_like(gnss_df['RawPseudorangeMeters'], np.nan)
    ADR_err = np.full_like(gnss_df['RawPseudorangeMeters'], np.nan)
    TDCP = np.full_like(gnss_df['RawPseudorangeMeters'], np.nan)
    # Loop for each signal
    for (i, (svid_sigtype, df)) in enumerate((gnss_df.groupby(['Svid', 'SignalType']))):
        df = df.replace(
            {'AccumulatedDeltaRangeMeters': {0: np.nan}})  # 0 to NaN

        # Compare time difference between pseudorange/carrier with Doppler
        drng1 = df['AccumulatedDeltaRangeMeters'].diff() - df['PseudorangeRateMetersPerSecond']
        drng2 = df['RawPseudorangeMeters'].diff() - df['PseudorangeRateMetersPerSecond']
        # 计算载波测量与前一行的差值
        tdcp_sat = df['AccumulatedDeltaRangeMeters'].diff()
        tdcp_sat_prrnan = tdcp_sat.fillna(df['PrrM'])

        # Check cycle-slip
        slip1 = (df['AccumulatedDeltaRangeState'].to_numpy()
                 & 2**1) != 0  # reset flag
        slip2 = (df['AccumulatedDeltaRangeState'].to_numpy()
                 & 2**2) != 0  # cycle-slip flag
        slip3 = np.fabs(drng1.to_numpy()) > carr_th  # Carrier phase jump
        slip4 = np.fabs(drng2.to_numpy()) > pr_th  # Pseudorange jump

        idx_slip = slip1 | slip2 | slip3 | slip4
        idx_slip[0] = True

        # groups with continuous carrier phase tracking
        df['group_slip'] = np.cumsum(idx_slip)

        # Psudorange - carrier phase
        df['dpc'] = df['RawPseudorangeMeters'] - \
            df['AccumulatedDeltaRangeMeters']

        # Absolute distance bias of carrier phase
        meandpc = df.groupby('group_slip')['dpc'].mean()
        df = df.merge(meandpc, on='group_slip', suffixes=('', '_Mean'))

        # Index of original gnss_df
        idx = (gnss_df['Svid'] == svid_sigtype[0]) & (
            gnss_df['SignalType'] == svid_sigtype[1])

        # Carrier phase + bias
        prsmooth[idx] = df['AccumulatedDeltaRangeMeters'] + df['dpc_Mean']
        slip_state[idx] = idx_slip
        ADR_err[idx] = np.abs(drng1)
        TDCP[idx] = tdcp_sat_prrnan

    # If carrier smoothing is not possible, use original pseudorange
    idx_nan = np.isnan(prsmooth)
    prsmooth[idx_nan] = gnss_df['RawPseudorangeMeters'][idx_nan]
    gnss_df['pr_smooth'] = prsmooth
    gnss_df['TDCP'] = TDCP
    idx_nan = np.isnan(slip_state)
    slip_state[idx_nan] = True
    gnss_df['slip_state'] = slip_state
    idx_nan = np.isnan(ADR_err)
    ADR_err[idx_nan] = 100
    # gnss_df['ADR_err'] = ADR_err
    return gnss_df

def satellite_selection(df, column):
    '''
    Returns a dataframe with a satellite selection

            Parameters:
                    df (pandas.DataFrame): Pandas dataframe
                    column (str): Pandas label (column) to filter

            Returns:
                    df (pandas.DataFrame): DataFrame with eliminated satellite signals
    '''

    idx = df[column].notnull()
    idx &= df['ReceivedSvTimeUncertaintyNanos'] < 500
    idx &= df['CarrierErrorHz'] < 2.0e6  # carrier frequency error (Hz)
    idx &= df['SvElevationDegrees'] > 20  # elevation angle (deg) default:20
    idx &= df['Cn0DbHz'] > 15  # C/N0 (dB-Hz) default: 15
    idx &= df['MultipathIndicator'] == 0  # Multipath flag

    return df[idx]
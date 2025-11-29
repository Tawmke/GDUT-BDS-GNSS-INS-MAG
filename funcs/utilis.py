# import library
import numpy as np
from math import radians, cos, sin, asin, sqrt
from haversine import haversine
import pickle
import pandas as pd

def exp_average(data, expFactor=0.1):
    expRawRewards = np.zeros(data.shape)
    for i in range(data.shape[0]):
        expRaw = 0.0
        J = 0.0
        for j in range(data.shape[1]):
            J *= (1.0-expFactor)
            J += (expFactor)
            rate = expFactor/J
            expRaw = (1-rate)*expRaw
            expRaw += rate*data[i][j]
            expRawRewards[i, j] = expRaw
    return expRawRewards

def exp_average_list(data, expFactor=0.1):
    expRawRewards = np.zeros(len(data))
    expRaw = 0.0
    J = 0.0
    for j in range(len(data)):
        J *= (1.0-expFactor)
        J += (expFactor)
        rate = expFactor/J
        expRaw = (1-rate)*expRaw
        expRaw += rate*data[j]
        expRawRewards[j] = expRaw
    return expRawRewards

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance

def cal_distance(row):
    """
    计算两个经纬度点之间的距离
    """
    long1 = row['LongitudeDegrees_truth']
    lat1 = row['LatitudeDegrees_truth']
    long2 = row['lngDeg_RLpredict']
    lat2 = row['latDeg_RLpredict']
    long3 = row['LongitudeDegrees']
    lat3 = row['LatitudeDegrees']
    g1 = (lat1,long1)
    g2 = (lat2,long2)
    g3 = (lat3,long3)
    # g1 = (long1, lat1)
    # g2 = (long2, lat2)
    # g3 = (long3, lat3)
    ret1 = haversine(g1, g2, unit='m')
    ret2 = haversine(g1, g3, unit='m')
    result1 = "%.7f" % ret1
    result2 = "%.7f" % ret2
    return result1, result2

def cal_distance_ecef(row,baseline_mod):
    """
    计算两个经纬度点之间的距离
    """
    y1 = row['ecefY']
    x1 = row['ecefX']
    z1 = row['ecefZ']
    y2 = row['Y_RLpredict']
    x2 = row['X_RLpredict']
    z2 = row['Z_RLpredict']
    if baseline_mod == 'bl':
        y3 = row['YEcefMeters_bl']
        x3 = row['XEcefMeters_bl']
        z3 = row['ZEcefMeters_bl']
    elif baseline_mod == 'wls':
        y3 = row['YEcefMeters_wls']
        x3 = row['XEcefMeters_wls']
        z3 = row['ZEcefMeters_wls']
    elif baseline_mod == 'wls_igst':
        y3 = row['YEcefMeters_wls_igst']
        x3 = row['XEcefMeters_wls_igst']
        z3 = row['ZEcefMeters_wls_igst']
    elif baseline_mod == 'afterRLAKF':
        y3 = row['YEcefMeters_KF_realtime']
        x3 = row['XEcefMeters_KF_realtime']
        z3 = row['ZEcefMeters_KF_realtime']
    elif baseline_mod == 'bds':
        y3 = row['YEcefMeters_bds']
        x3 = row['XEcefMeters_bds']
        z3 = row['ZEcefMeters_bds']
    elif baseline_mod == 'kf':
        y3 = row['YEcefMeters_kf']
        x3 = row['XEcefMeters_kf']
        z3 = row['ZEcefMeters_kf']
    elif baseline_mod == 'kf_igst':
        y3 = row['YEcefMeters_kf_igst']
        x3 = row['XEcefMeters_kf_igst']
        z3 = row['ZEcefMeters_kf_igst']
    #ret1 = haversine(g1, g2, unit='m')
    #ret2 = haversine(g1, g3, unit='m')
    result1 = np.sqrt(((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))
    result2 = np.sqrt(((x3 - x1) ** 2 + (y3 - y1) ** 2 + (z3 - z1) ** 2))
    if np.isnan(x2) or np.isnan(y2) or np.isnan(z2):
        xerr1 = np.nan
        yerr1 = np.nan
        zerr1 = np.nan
    else:
        xerr1 = np.sqrt(((x2 - x1) ** 2))
        yerr1 = np.sqrt(((y2 - y1) ** 2))
        zerr1 = np.sqrt(((z2 - z1) ** 2))
    xerr2 = np.sqrt(((x3 - x1) ** 2))
    yerr2 = np.sqrt(((y3 - y1) ** 2))
    zerr2 = np.sqrt(((z3 - z1) ** 2))

    return result1, result2, xerr1, yerr1, zerr1, xerr2, yerr2, zerr2

def cal_distance_ecef_RLKF(row,baseline_mod):
    """
    计算两个经纬度点之间的距离
    """
    y1 = row['ecefY']
    x1 = row['ecefX']
    z1 = row['ecefZ']
    y2 = row['Y_RLpredict']
    x2 = row['X_RLpredict']
    z2 = row['Z_RLpredict']
    if baseline_mod == 'bl':
        y3 = row['YEcefMeters_bl']
        x3 = row['XEcefMeters_bl']
        z3 = row['ZEcefMeters_bl']
    elif baseline_mod == 'wls':
        y3 = row['YEcefMeters_wls_igst']
        x3 = row['XEcefMeters_wls_igst']
        z3 = row['ZEcefMeters_wls_igst']
    elif baseline_mod == 'wls_igst':
        y3 = row['YEcefMeters_KF_realtime']
        x3 = row['XEcefMeters_KF_realtime']
        z3 = row['ZEcefMeters_KF_realtime']
    elif baseline_mod == 'afterRLAKF':
        y3 = row['YEcefMeters_KF_realtime']
        x3 = row['XEcefMeters_KF_realtime']
        z3 = row['ZEcefMeters_KF_realtime']
    elif baseline_mod == 'bds':
        y3 = row['YEcefMeters_bds']
        x3 = row['XEcefMeters_bds']
        z3 = row['ZEcefMeters_bds']
    elif baseline_mod == 'kf':
        y3 = row['YEcefMeters_kf']
        x3 = row['XEcefMeters_kf']
        z3 = row['ZEcefMeters_kf']
    elif baseline_mod == 'kf_igst':
        y3 = row['YEcefMeters_kf_igst']
        x3 = row['XEcefMeters_kf_igst']
        z3 = row['ZEcefMeters_kf_igst']
    #ret1 = haversine(g1, g2, unit='m')
    #ret2 = haversine(g1, g3, unit='m')
    result1 = np.sqrt(((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2))
    result2 = np.sqrt(((x3 - x1) ** 2 + (y3 - y1) ** 2 + (z3 - z1) ** 2))
    if np.isnan(x2) or np.isnan(y2) or np.isnan(z2):
        xerr1 = np.nan
        yerr1 = np.nan
        zerr1 = np.nan
    else:
        xerr1 = np.sqrt(((x2 - x1) ** 2))
        yerr1 = np.sqrt(((y2 - y1) ** 2))
        zerr1 = np.sqrt(((z2 - z1) ** 2))
    xerr2 = np.sqrt(((x3 - x1) ** 2))
    yerr2 = np.sqrt(((y3 - y1) ** 2))
    zerr2 = np.sqrt(((z3 - z1) ** 2))

    return result1, result2, xerr1, yerr1, zerr1, xerr2, yerr2, zerr2

def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    """
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    dist = 2 * RADIUS * np.arcsin(a**0.5)
    return dist

def percentile50(x):
    return np.percentile(x, 50)
def percentile95(x):
    return np.percentile(x, 95)

def get_train_score(df, gt):
    gt = gt.rename(columns={'latDeg':'latDeg_gt', 'lngDeg':'lngDeg_gt'})
    df = df.merge(gt, on=['collectionName', 'phoneName', 'millisSinceGpsEpoch'], how='inner')
    # calc_distance_error
    df['err'] = calc_haversine(df['latDeg_gt'], df['lngDeg_gt'], df['latDeg'], df['lngDeg'])
    # calc_evaluate_score
    df['phone'] = df['collectionName'] + '_' + df['phoneName']
    res = df.groupby('phone')['err'].agg([percentile50, percentile95])
    res['p50_p90_mean'] = (res['percentile50'] + res['percentile95']) / 2
    score = res['p50_p90_mean'].mean()
    return score

def recording_results(data_truth_dic,trajdata_range,tripIDlist,logdirname):
    error_mean_all = 0
    rl_distances_mean_all = 0
    or_distances_mean_all = 0
    error_std_all = 0
    rl_distances_std_all = 0
    or_distances_std_all = 0
    for train_tripIDnum in range(trajdata_range[0], trajdata_range[1] + 1):
        try:
            pd_train = data_truth_dic[tripIDlist[train_tripIDnum]]
            test = pd_train.loc[:, ['LongitudeDegrees_truth', 'LatitudeDegrees_truth',
                                    'lngDeg_RLpredict', 'latDeg_RLpredict', 'LongitudeDegrees', 'LatitudeDegrees']]
            test['rl_distance'] = test.apply(lambda test: cal_distance(test)[0], axis=1)
            test['or_distance'] = test.apply(lambda test: cal_distance(test)[1], axis=1)
            test['error'] = test['rl_distance'].astype(
                float) - test['or_distance'].astype(float)
            test['count_rl_distance'] = test['rl_distance'].astype(float)
            test['count_or_distance'] = test['or_distance'].astype(float)
            if train_tripIDnum > trajdata_range[0]:
                error_pd.insert(error_pd.shape[1], f'{train_tripIDnum}', test['error'].describe())
                rl_distance_pd.insert(rl_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_rl_distance'].describe())
                or_distance_pd.insert(or_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_or_distance'].describe())
            else:
                error_pd = pd.DataFrame(test['error'].describe())
                error_pd = error_pd.rename(columns={'error': f'{train_tripIDnum}'})
                error_pd.index.name = 'errors'
                rl_distance_pd = pd.DataFrame(test['count_rl_distance'].describe())
                rl_distance_pd = rl_distance_pd.rename(columns={'count_rl_distance': f'{train_tripIDnum}'})
                rl_distance_pd.index.name = 'rl_distances'
                or_distance_pd = pd.DataFrame(test['count_or_distance'].describe())
                or_distance_pd = or_distance_pd.rename(columns={'count_or_distance': f'{train_tripIDnum}'})
                or_distance_pd.index.name = 'or_distances'
            error_mean_all += test['error'].describe()['count'] * test['error'].describe()['mean']
            rl_distances_mean_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['mean']
            or_distances_mean_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['mean']
            error_std_all += test['error'].describe()['count'] * test['error'].describe()['std']
            rl_distances_std_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['std']
            or_distances_std_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['std']
        except:
            print(f'Trajectory {train_tripIDnum} error.')

    num_total_err = np.sum(error_pd.loc['count', :])
    num_total_rl = np.sum(rl_distance_pd.loc['count', :])
    num_total_or = np.sum(or_distance_pd.loc['count', :])
    error_min = np.min(error_pd.loc['min', :])
    error_max = np.max(error_pd.loc['max', :])
    error_pd.insert(error_pd.shape[1], 'Avg', [num_total_err, error_mean_all / num_total_err, error_std_all / num_total_err,
                                               error_min, 0, 0, 0, error_max])
    rl_distance_pd.insert(rl_distance_pd.shape[1], 'Avg',
                          [num_total_rl, rl_distances_mean_all / num_total_rl, rl_distances_std_all / num_total_rl,
                           np.min(rl_distance_pd.loc['min', :]), 0, 0, 0, np.max(rl_distance_pd.loc['max', :])])
    or_distance_pd.insert(or_distance_pd.shape[1], 'Avg',
                          [num_total_or, or_distances_mean_all / num_total_or, or_distances_std_all / num_total_or,
                           np.min(or_distance_pd.loc['min', :]), 0, 0, 0, np.max(or_distance_pd.loc['max', :])])
    error_pd.to_csv(logdirname + 'errors.csv', index=True)
    rl_distance_pd.to_csv(logdirname + 'rl_distances.csv', index=True)
    or_distance_pd.to_csv(logdirname + 'or_distances.csv', index=True)
    print(
        f'Perfermances: count {num_total_err:1.0f}, compared with baseline mean: {error_mean_all / num_total_err:4.3f}+{error_std_all / num_total_err:4.3f}m, '
        f'min: {error_min:4.3f}m, max: {error_max:4.3f}m.')

def recording_results_ecef(data_truth_dic,trajdata_range,tripIDlist,logdirname,baseline_mod,traj_record):
    error_mean_all = 0
    rl_distances_mean_all = 0
    or_distances_mean_all = 0
    error_std_all = 0
    rl_distances_std_all = 0
    or_distances_std_all = 0
    pd_gen=False
    for train_tripIDnum in range(trajdata_range[0], trajdata_range[1] + 1):
        try:
            pd_train = data_truth_dic[tripIDlist[train_tripIDnum]]
            pd_train = pd_train[pd_train['X_RLpredict'].notnull()]
            if traj_record:
                # record rl traj
                if baseline_mod == 'kf_igst':
                    record_columns=['UnixTimeMillis','ecefX', 'ecefY', 'ecefZ','X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                    'XEcefMeters_kf_igst', 'YEcefMeters_kf_igst', 'ZEcefMeters_kf_igst']
                elif baseline_mod == 'wls_igst':
                    record_columns = ['UnixTimeMillis', 'ecefX', 'ecefY', 'ecefZ', 'X_RLpredict', 'Y_RLpredict','Z_RLpredict',
                                      'XEcefMeters_wls_igst', 'YEcefMeters_wls_igst', 'ZEcefMeters_wls_igst']
                pd_record = pd_train[record_columns]
                # pd_record = pd_record[pd_record['X_RLpredict'].notnull()]
                pd_record.to_csv(logdirname + f'rl_traj_{tripIDlist[train_tripIDnum].replace("/","_")}.csv', index=True)
            if baseline_mod == 'bl':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_bl', 'YEcefMeters_bl', 'ZEcefMeters_bl']]
            elif baseline_mod == 'wls_igst':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_wls_igst', 'YEcefMeters_wls_igst', 'ZEcefMeters_wls_igst']]
            elif baseline_mod == 'bds':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_bds', 'YEcefMeters_bds', 'ZEcefMeters_bds']]
            elif baseline_mod == 'kf':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_kf', 'YEcefMeters_kf', 'ZEcefMeters_kf']]
            elif baseline_mod == 'kf_igst':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_kf_igst', 'YEcefMeters_kf_igst', 'ZEcefMeters_kf_igst']]
            test['rl_distance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[0], axis=1)
            test['or_distance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[1], axis=1)
            test['error'] = test['rl_distance'].astype(float) - test['or_distance'].astype(float)
            test['count_rl_distance'] = test['rl_distance'].astype(float)
            test['count_or_distance'] = test['or_distance'].astype(float)
            test['rl_xdistance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[2], axis=1)
            test['rl_ydistance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[3], axis=1)
            test['rl_zdistance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[4], axis=1)
            test['count_rl_xdistance'] = test['rl_xdistance'].astype(float)
            test['count_rl_ydistance'] = test['rl_ydistance'].astype(float)
            test['count_rl_zdistance'] = test['rl_zdistance'].astype(float)
            test['or_xdistance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[5], axis=1)
            test['or_ydistance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[6], axis=1)
            test['or_zdistance'] = test.apply(lambda test: cal_distance_ecef(test,baseline_mod)[7], axis=1)
            test['count_or_xdistance'] = test['or_xdistance'].astype(float)
            test['count_or_ydistance'] = test['or_ydistance'].astype(float)
            test['count_or_zdistance'] = test['or_zdistance'].astype(float)
            rl_xdistance_mean=np.mean(test['rl_xdistance'])
            if pd_gen:
                error_pd.insert(error_pd.shape[1], f'{train_tripIDnum}', test['error'].describe())
                rl_distance_pd.insert(rl_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_rl_distance'].describe())
                or_distance_pd.insert(or_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_or_distance'].describe())
                tmp_dic = {'tripID':tripIDlist[train_tripIDnum], 'rl_xdistance_mean': np.mean(test['rl_xdistance']), 'rl_ydistance_mean': np.mean(test['rl_ydistance']),'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                                                'rl_xdistance_std': np.std(test['rl_xdistance']),'rl_ydistance_std': np.std(test['rl_ydistance']), 'rl_zdistance_std': np.std(test['rl_zdistance']),
                                                'rl_xdistance_min': min(test['rl_xdistance']),'rl_ydistance_min': np.nanmin(test['rl_ydistance']), 'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                                                'rl_xdistance_max': np.nanmax(test['rl_xdistance']),'rl_ydistance_max': np.nanmax(test['rl_ydistance']), 'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                                                'or_xdistance_mean': np.mean(test['or_xdistance']), 'or_ydistance_mean': np.mean(test['or_ydistance']),'or_zdistance_mean': np.mean(test['or_zdistance']),
                                                'or_xdistance_std': np.std(test['or_xdistance']),'or_ydistance_std': np.std(test['or_ydistance']), 'or_zdistance_std': np.std(test['or_zdistance']),
                                                'or_xdistance_min': np.nanmin(test['or_xdistance']),'or_ydistance_min': np.nanmin(test['or_ydistance']), 'or_zdistance_min': np.nanmin(test['or_zdistance']),
                                                'or_xdistance_max': np.nanmax(test['or_xdistance']),'or_ydistance_max': np.nanmax(test['or_ydistance']), 'or_zdistance_max': np.nanmax(test['or_zdistance'])}

                xyz_distance_pd.insert(xyz_distance_pd.shape[1], f'{train_tripIDnum}',
                                       pd.DataFrame.from_dict(tmp_dic, orient='index').loc[:, 0])
            else:
                error_pd = pd.DataFrame(test['error'].describe())
                error_pd = error_pd.rename(columns={'error': f'{train_tripIDnum}'})
                error_pd.index.name = 'errors'
                rl_distance_pd = pd.DataFrame(test['count_rl_distance'].describe())
                rl_distance_pd = rl_distance_pd.rename(columns={'count_rl_distance': f'{train_tripIDnum}'})
                rl_distance_pd.index.name = 'rl_distances'
                or_distance_pd = pd.DataFrame(test['count_or_distance'].describe())
                or_distance_pd = or_distance_pd.rename(columns={'count_or_distance': f'{train_tripIDnum}'})
                or_distance_pd.index.name = 'or_distances'

                tmp_dic = {'tripID':tripIDlist[train_tripIDnum], 'rl_xdistance_mean': np.mean(test['rl_xdistance']), 'rl_ydistance_mean': np.mean(test['rl_ydistance']),'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                                                'rl_xdistance_std': np.std(test['rl_xdistance']),'rl_ydistance_std': np.std(test['rl_ydistance']), 'rl_zdistance_std': np.std(test['rl_zdistance']),
                                                'rl_xdistance_min': np.nanmin(test['rl_xdistance']),'rl_ydistance_min': np.nanmin(test['rl_ydistance']), 'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                                                'rl_xdistance_max': np.nanmax(test['rl_xdistance']),'rl_ydistance_max': np.nanmax(test['rl_ydistance']), 'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                                                'or_xdistance_mean': np.mean(test['or_xdistance']), 'or_ydistance_mean': np.mean(test['or_ydistance']),'or_zdistance_mean': np.mean(test['or_zdistance']),
                                                'or_xdistance_std': np.std(test['or_xdistance']),'or_ydistance_std': np.std(test['or_ydistance']), 'or_zdistance_std': np.std(test['or_zdistance']),
                                                'or_xdistance_min': np.nanmin(test['or_xdistance']),'or_ydistance_min': np.nanmin(test['or_ydistance']), 'or_zdistance_min': np.nanmin(test['or_zdistance']),
                                                'or_xdistance_max': np.nanmax(test['or_xdistance']),'or_ydistance_max': np.nanmax(test['or_ydistance']), 'or_zdistance_max': np.nanmax(test['or_zdistance'])}
                xyz_distance_pd=pd.DataFrame.from_dict(tmp_dic, orient='index')
                pd_gen=True
            error_mean_all += test['error'].describe()['count'] * test['error'].describe()['mean']
            rl_distances_mean_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['mean']
            or_distances_mean_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['mean']
            error_std_all += test['error'].describe()['count'] * test['error'].describe()['std']
            rl_distances_std_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['std']
            or_distances_std_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['std']
        except:
            print(f'Trajectory {train_tripIDnum} error.')

    num_total_err = np.sum(error_pd.loc['count', :])
    num_total_rl = np.sum(rl_distance_pd.loc['count', :])
    num_total_or = np.sum(or_distance_pd.loc['count', :])
    error_min = np.min(error_pd.loc['min', :])
    error_max = np.max(error_pd.loc['max', :])
    error_pd.insert(error_pd.shape[1], 'Avg', [num_total_err, error_mean_all / num_total_err, error_std_all / num_total_err,
                                               error_min, 0, 0, 0, error_max])
    rl_distance_pd.insert(rl_distance_pd.shape[1], 'Avg',
                          [num_total_rl, rl_distances_mean_all / num_total_rl, rl_distances_std_all / num_total_rl,
                           np.min(rl_distance_pd.loc['min', :]), 0, 0, 0, np.max(rl_distance_pd.loc['max', :])])
    or_distance_pd.insert(or_distance_pd.shape[1], 'Avg',
                          [num_total_or, or_distances_mean_all / num_total_or, or_distances_std_all / num_total_or,
                           np.min(or_distance_pd.loc['min', :]), 0, 0, 0, np.max(or_distance_pd.loc['max', :])])
    error_pd.to_csv(logdirname + 'errors.csv', index=True)
    rl_distance_pd.to_csv(logdirname + 'rl_distances.csv', index=True)
    or_distance_pd.to_csv(logdirname + 'or_distances.csv', index=True)
    xyz_distance_pd.to_csv(logdirname + 'xyz_distances.csv', index=True)
    print(
        f'Perfermances: count {num_total_err:1.0f}, compared with baseline mean: {error_mean_all / num_total_err:4.3f}+{error_std_all / num_total_err:4.3f}m, '
        f'min: {error_min:4.3f}m, max: {error_max:4.3f}m.')

def recording_results_ecef_RL4KF(dir_path, data_truth_dic,trajdata_range,tripIDlist,logdirname,baseline_mod,traj_record):
    error_mean_all = 0
    rl_distances_mean_all = 0
    or_distances_mean_all = 0
    error_std_all = 0
    rl_distances_std_all = 0
    or_distances_std_all = 0
    pd_gen=False
    with open(dir_path + 'envRLKF/raw_kf_igst_realtime.pkl', "rb") as file:
        data_kf_realtime = pickle.load(file) # load kf realtime
    file.close()
    for train_tripIDnum in range(trajdata_range[0], trajdata_range[1] + 1):
        try:
            pd_train = data_truth_dic[tripIDlist[train_tripIDnum]]
            pd_kf_realtime = data_kf_realtime[tripIDlist[train_tripIDnum]]
            drop_num = len(pd_kf_realtime) - len(pd_train)
            pd_kf_realtime = pd_kf_realtime.drop(pd_kf_realtime.index[:drop_num]).reset_index(drop=True)
            index_nonull = pd_train['X_RLpredict'].notnull()
            pd_train = pd_train[index_nonull]
            pd_kf_realtime = pd_kf_realtime[index_nonull]
            if traj_record:
                # record rl traj
                if baseline_mod == 'kf_igst':
                    record_columns=['UnixTimeMillis','ecefX', 'ecefY', 'ecefZ','X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                    'XEcefMeters_kf_igst', 'YEcefMeters_kf_igst', 'ZEcefMeters_kf_igst']
                elif baseline_mod == 'wls_igst':
                    record_columns = ['UnixTimeMillis', 'ecefX', 'ecefY', 'ecefZ', 'X_RLpredict', 'Y_RLpredict','Z_RLpredict',
                                      'XEcefMeters_wls_igst', 'YEcefMeters_wls_igst', 'ZEcefMeters_wls_igst','satnum','CN0_mean','EA_mean','PR_mean']
                elif baseline_mod == 'wls':
                    record_columns = ['UnixTimeMillis', 'ecefX', 'ecefY', 'ecefZ', 'X_RLpredict', 'Y_RLpredict','Z_RLpredict',
                                      'XEcefMeters_wls_igst', 'YEcefMeters_wls_igst', 'ZEcefMeters_wls_igst']
                elif baseline_mod == 'afterRLAKF':
                    record_columns = ['UnixTimeMillis', 'ecefX', 'ecefY', 'ecefZ', 'X_RLpredict', 'Y_RLpredict','Z_RLpredict',
                                      'X_RLAKF', 'Y_RLAKF', 'Z_RLAKF']
                record_columns_kf_realtime = ['UnixTimeMillis', 'XEcefMeters_KF_realtime', 'YEcefMeters_KF_realtime', 'ZEcefMeters_KF_realtime']
                pd_record_kfrealtime = pd_kf_realtime[record_columns_kf_realtime]
                try:
                    pd_record = pd_train[record_columns]
                except:
                    pd_record = pd_train[record_columns[0:10]]
                pd_record = pd.merge(pd_record, pd_record_kfrealtime, on='UnixTimeMillis')
                pd_record.to_csv(logdirname + f'rl_traj_{tripIDlist[train_tripIDnum].replace("/","_")}.csv', index=True)
            if baseline_mod == 'bl':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_bl', 'YEcefMeters_bl', 'ZEcefMeters_bl']]
            elif baseline_mod == 'wls_igst' or baseline_mod == 'wls':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_wls_igst', 'YEcefMeters_wls_igst', 'ZEcefMeters_wls_igst']]
                test = pd_record
            elif baseline_mod == 'afterRLAKF':
                test = pd_record
            elif baseline_mod == 'bds':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_bds', 'YEcefMeters_bds', 'ZEcefMeters_bds']]
            elif baseline_mod == 'kf':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_kf', 'YEcefMeters_kf', 'ZEcefMeters_kf']]
            elif baseline_mod == 'kf_igst':
                test = pd_train.loc[:, ['ecefX', 'ecefY', 'ecefZ',
                                        'X_RLpredict', 'Y_RLpredict', 'Z_RLpredict',
                                        'XEcefMeters_kf_igst', 'YEcefMeters_kf_igst', 'ZEcefMeters_kf_igst']]
                test = pd_record
            test['rl_distance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod)[0], axis=1)
            test['or_distance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod)[1], axis=1)
            test['error'] = test['rl_distance'].astype(float) - test['or_distance'].astype(float)
            test['count_rl_distance'] = test['rl_distance'].astype(float)
            test['count_or_distance'] = test['or_distance'].astype(float)
            test['rl_xdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod)[2], axis=1)
            test['rl_ydistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod)[3], axis=1)
            test['rl_zdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod)[4], axis=1)
            test['count_rl_xdistance'] = test['rl_xdistance'].astype(float)
            test['count_rl_ydistance'] = test['rl_ydistance'].astype(float)
            test['count_rl_zdistance'] = test['rl_zdistance'].astype(float)
            test['or_xdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod)[5], axis=1)
            test['or_ydistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod)[6], axis=1)
            test['or_zdistance'] = test.apply(lambda test: cal_distance_ecef_RLKF(test,baseline_mod)[7], axis=1)
            test['count_or_xdistance'] = test['or_xdistance'].astype(float)
            test['count_or_ydistance'] = test['or_ydistance'].astype(float)
            test['count_or_zdistance'] = test['or_zdistance'].astype(float)
            rl_xdistance_mean=np.mean(test['rl_xdistance'])
            if pd_gen:
                error_pd.insert(error_pd.shape[1], f'{train_tripIDnum}', test['error'].describe())
                rl_distance_pd.insert(rl_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_rl_distance'].describe())
                or_distance_pd.insert(or_distance_pd.shape[1], f'{train_tripIDnum}',
                                      test['count_or_distance'].describe())
                tmp_dic = {'tripID':tripIDlist[train_tripIDnum], 'rl_xdistance_mean': np.mean(test['rl_xdistance']), 'rl_ydistance_mean': np.mean(test['rl_ydistance']),'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                                                'rl_xdistance_std': np.std(test['rl_xdistance']),'rl_ydistance_std': np.std(test['rl_ydistance']), 'rl_zdistance_std': np.std(test['rl_zdistance']),
                                                'rl_xdistance_min': min(test['rl_xdistance']),'rl_ydistance_min': np.nanmin(test['rl_ydistance']), 'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                                                'rl_xdistance_max': np.nanmax(test['rl_xdistance']),'rl_ydistance_max': np.nanmax(test['rl_ydistance']), 'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                                                'or_xdistance_mean': np.mean(test['or_xdistance']), 'or_ydistance_mean': np.mean(test['or_ydistance']),'or_zdistance_mean': np.mean(test['or_zdistance']),
                                                'or_xdistance_std': np.std(test['or_xdistance']),'or_ydistance_std': np.std(test['or_ydistance']), 'or_zdistance_std': np.std(test['or_zdistance']),
                                                'or_xdistance_min': np.nanmin(test['or_xdistance']),'or_ydistance_min': np.nanmin(test['or_ydistance']), 'or_zdistance_min': np.nanmin(test['or_zdistance']),
                                                'or_xdistance_max': np.nanmax(test['or_xdistance']),'or_ydistance_max': np.nanmax(test['or_ydistance']), 'or_zdistance_max': np.nanmax(test['or_zdistance'])}

                xyz_distance_pd.insert(xyz_distance_pd.shape[1], f'{train_tripIDnum}',
                                       pd.DataFrame.from_dict(tmp_dic, orient='index').loc[:, 0])
            else:
                error_pd = pd.DataFrame(test['error'].describe())
                error_pd = error_pd.rename(columns={'error': f'{train_tripIDnum}'})
                error_pd.index.name = 'errors'
                rl_distance_pd = pd.DataFrame(test['count_rl_distance'].describe())
                rl_distance_pd = rl_distance_pd.rename(columns={'count_rl_distance': f'{train_tripIDnum}'})
                rl_distance_pd.index.name = 'rl_distances'
                or_distance_pd = pd.DataFrame(test['count_or_distance'].describe())
                or_distance_pd = or_distance_pd.rename(columns={'count_or_distance': f'{train_tripIDnum}'})
                or_distance_pd.index.name = 'or_distances'

                tmp_dic = {'tripID':tripIDlist[train_tripIDnum], 'rl_xdistance_mean': np.mean(test['rl_xdistance']), 'rl_ydistance_mean': np.mean(test['rl_ydistance']),'rl_zdistance_mean': np.mean(test['rl_zdistance']),
                                                'rl_xdistance_std': np.std(test['rl_xdistance']),'rl_ydistance_std': np.std(test['rl_ydistance']), 'rl_zdistance_std': np.std(test['rl_zdistance']),
                                                'rl_xdistance_min': np.nanmin(test['rl_xdistance']),'rl_ydistance_min': np.nanmin(test['rl_ydistance']), 'rl_zdistance_min': np.nanmin(test['rl_zdistance']),
                                                'rl_xdistance_max': np.nanmax(test['rl_xdistance']),'rl_ydistance_max': np.nanmax(test['rl_ydistance']), 'rl_zdistance_max': np.nanmax(test['rl_zdistance']),
                                                'or_xdistance_mean': np.mean(test['or_xdistance']), 'or_ydistance_mean': np.mean(test['or_ydistance']),'or_zdistance_mean': np.mean(test['or_zdistance']),
                                                'or_xdistance_std': np.std(test['or_xdistance']),'or_ydistance_std': np.std(test['or_ydistance']), 'or_zdistance_std': np.std(test['or_zdistance']),
                                                'or_xdistance_min': np.nanmin(test['or_xdistance']),'or_ydistance_min': np.nanmin(test['or_ydistance']), 'or_zdistance_min': np.nanmin(test['or_zdistance']),
                                                'or_xdistance_max': np.nanmax(test['or_xdistance']),'or_ydistance_max': np.nanmax(test['or_ydistance']), 'or_zdistance_max': np.nanmax(test['or_zdistance'])}
                xyz_distance_pd=pd.DataFrame.from_dict(tmp_dic, orient='index')
                pd_gen=True
            error_mean_all += test['error'].describe()['count'] * test['error'].describe()['mean']
            rl_distances_mean_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['mean']
            or_distances_mean_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['mean']
            error_std_all += test['error'].describe()['count'] * test['error'].describe()['std']
            rl_distances_std_all += test['count_rl_distance'].describe()['count'] * test['count_rl_distance'].describe()['std']
            or_distances_std_all += test['count_or_distance'].describe()['count'] * test['count_or_distance'].describe()['std']
        except:
            print(f'Trajectory {train_tripIDnum} error.')

    num_total_err = np.sum(error_pd.loc['count', :])
    num_total_rl = np.sum(rl_distance_pd.loc['count', :])
    num_total_or = np.sum(or_distance_pd.loc['count', :])
    error_min = np.min(error_pd.loc['min', :])
    error_max = np.max(error_pd.loc['max', :])
    error_pd.insert(error_pd.shape[1], 'Avg', [num_total_err, error_mean_all / num_total_err, error_std_all / num_total_err,
                                               error_min, 0, 0, 0, error_max])
    rl_distance_pd.insert(rl_distance_pd.shape[1], 'Avg',
                          [num_total_rl, rl_distances_mean_all / num_total_rl, rl_distances_std_all / num_total_rl,
                           np.min(rl_distance_pd.loc['min', :]), 0, 0, 0, np.max(rl_distance_pd.loc['max', :])])
    or_distance_pd.insert(or_distance_pd.shape[1], 'Avg',
                          [num_total_or, or_distances_mean_all / num_total_or, or_distances_std_all / num_total_or,
                           np.min(or_distance_pd.loc['min', :]), 0, 0, 0, np.max(or_distance_pd.loc['max', :])])
    error_pd.to_csv(logdirname + 'errors.csv', index=True)
    rl_distance_pd.to_csv(logdirname + 'rl_distances.csv', index=True)
    or_distance_pd.to_csv(logdirname + 'or_distances.csv', index=True)
    xyz_distance_pd.to_csv(logdirname + 'xyz_distances.csv', index=True)
    print(
        f'Perfermances: count {num_total_err:1.0f}, compared with baseline mean: {error_mean_all / num_total_err:4.3f}+{error_std_all / num_total_err:4.3f}m, '
        f'min: {error_min:4.3f}m, max: {error_max:4.3f}m.')
import numpy as np
import pandas as pd
import pymap3d as pm
import pymap3d.vincenty as pmv
import matplotlib.pyplot as plt
import glob as gl
import math
import scipy.optimize
from tqdm.auto import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial import distance

# Constants
CLIGHT = 299_792_458   # speed of light (m/s)
RE_WGS84 = 6_378_137   # earth semimajor axis (WGS84) (m)
OMGE = 7.2921151467E-5  # earth angular velocity (IS-GPS) (rad/s)
g_acc = 9.80665 # Gravitational acceleration
E_2 = 0.00669437999014 # 第一偏心率

max_consecutive = 50
phone_imu_sigmav_dic = {'pixel5a':1,'pixel6pro':0.3,'sm-s908b':1,'pixel7pro':5,'pixel5':1}

# Satellite selection using carrier frequency error, elevation angle, and C/N0
def satellite_selection(df, column):
    """
    Args:
        df : DataFrame from device_gnss.csv
        column : Column name
    Returns:
        df: DataFrame with eliminated satellite signals
    """
    idx = df[column].notnull()
    idx &= df['CarrierErrorHz'] < 2.0e6  # carrier frequency error (Hz)
    idx &= df['SvElevationDegrees'] > 10.0  # elevation angle (deg)
    idx &= df['Cn0DbHz'] > 15.0  # C/N0 (dB-Hz)
    idx &= df['MultipathIndicator'] == 0 # Multipath flag

    return df[idx]


# Compute line-of-sight vector from user to satellite
def los_vector(xusr, xsat):
    """
    Args:
        xusr : user position in ECEF (m)
        xsat : satellite position in ECEF (m)
    Returns:
        u: unit line-of-sight vector in ECEF (m)
        rng: distance between user and satellite (m)
    """
    u = xsat - xusr
    rng = np.linalg.norm(u, axis=1).reshape(-1, 1)
    u /= rng

    return u, rng.reshape(-1)


# Compute Jacobian matrix
def jac_pr_residuals(x, xsat, pr, W):
    """
    Args:
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        pr : pseudorange (m)
        W : weight matrix
    Returns:
        W*J : Jacobian matrix
    """
    u, _ = los_vector(x[:3], xsat)
    J = np.hstack([-u, np.ones([len(pr), 1])])  # J = [-ux -uy -uz 1]

    return W @ J


# Compute pseudorange residuals
def pr_residuals(x, xsat, pr, W):
    """
    Args:
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        pr : pseudorange (m)
        W : weight matrix
    Returns:
        residuals*W : pseudorange residuals
    """
    u, rng = los_vector(x[:3], xsat)

    # Approximate correction of the earth rotation (Sagnac effect) often used in GNSS positioning
    rng += OMGE * (xsat[:, 0] * x[1] - xsat[:, 1] * x[0]) / CLIGHT

    # Add GPS L1 clock offset
    residuals = rng - (pr - x[3])

    return residuals @ W


# Compute Jacobian matrix
def jac_prr_residuals(v, vsat, prr, x, xsat, W):
    """
    Args:
        v : current velocity in ECEF (m/s)
        vsat : satellite velocity in ECEF (m/s)
        prr : pseudorange rate (m/s)
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        W : weight matrix
    Returns:
        W*J : Jacobian matrix
    """
    u, _ = los_vector(x[:3], xsat)
    J = np.hstack([-u, np.ones([len(prr), 1])])

    return W @ J


# Compute pseudorange rate residuals
def prr_residuals(v, vsat, prr, x, xsat, W):
    """
    Args:
        v : current velocity in ECEF (m/s)
        vsat : satellite velocity in ECEF (m/s)
        prr : pseudorange rate (m/s)
        x : current position in ECEF (m)
        xsat : satellite position in ECEF (m)
        W : weight matrix
    Returns:
        residuals*W : pseudorange rate residuals
    """
    u, rng = los_vector(x[:3], xsat)
    rate = np.sum((vsat - v[:3]) * u, axis=1) \
           + OMGE / CLIGHT * (vsat[:, 1] * x[0] + xsat[:, 1] * v[0]
                              - vsat[:, 0] * x[1] - xsat[:, 0] * v[1])

    residuals = rate - (prr - v[3])

    return residuals @ W

# Carrier smoothing of pseudarange
def carrier_smoothing(gnss_df):
    """
    Args:
        df : DataFrame from device_gnss.csv
    Returns:
        df: DataFrame with carrier-smoothing pseudorange 'pr_smooth'
    """
    carr_th = 1.5 # carrier phase jump threshold [m] ** 2.0 -> 1.5 **
    pr_th =  20.0 # pseudorange jump threshold [m]

    prsmooth = np.full_like(gnss_df['RawPseudorangeMeters'], np.nan)
    # Loop for each signal
    for (i, (svid_sigtype, df)) in enumerate((gnss_df.groupby(['Svid', 'SignalType']))):
        df = df.replace(
            {'AccumulatedDeltaRangeMeters': {0: np.nan}})  # 0 to NaN

        # Compare time difference between pseudorange/carrier with Doppler
        drng1 = df['AccumulatedDeltaRangeMeters'].diff() - df['PseudorangeRateMetersPerSecond']
        drng2 = df['RawPseudorangeMeters'].diff() - df['PseudorangeRateMetersPerSecond']

        # Check cycle-slip
        slip1 = (df['AccumulatedDeltaRangeState'].to_numpy() & 2**1) != 0  # reset flag
        slip2 = (df['AccumulatedDeltaRangeState'].to_numpy() & 2**2) != 0  # cycle-slip flag
        slip3 = np.fabs(drng1.to_numpy()) > carr_th # Carrier phase jump
        slip4 = np.fabs(drng2.to_numpy()) > pr_th # Pseudorange jump

        idx_slip = slip1 | slip2 | slip3 | slip4
        idx_slip[0] = True

        # groups with continuous carrier phase tracking
        df['group_slip'] = np.cumsum(idx_slip)

        # Psudorange - carrier phase
        df['dpc'] = df['RawPseudorangeMeters'] - df['AccumulatedDeltaRangeMeters']

        # Absolute distance bias of carrier phase
        meandpc = df.groupby('group_slip')['dpc'].mean()
        df = df.merge(meandpc, on='group_slip', suffixes=('', '_Mean'))

        # Index of original gnss_df
        idx = (gnss_df['Svid'] == svid_sigtype[0]) & (
            gnss_df['SignalType'] == svid_sigtype[1])

        # Carrier phase + bias
        prsmooth[idx] = df['AccumulatedDeltaRangeMeters'] + df['dpc_Mean']

    # If carrier smoothing is not possible, use original pseudorange
    idx_nan = np.isnan(prsmooth)
    prsmooth[idx_nan] = gnss_df['RawPseudorangeMeters'][idx_nan]
    gnss_df['pr_smooth'] = prsmooth

    return gnss_df

# Compute distance by Vincenty's formulae
def vincenty_distance(llh1, llh2):
    """
    Args:
        llh1 : [latitude,longitude] (deg)
        llh2 : [latitude,longitude] (deg)
    Returns:
        d : distance between llh1 and llh2 (m)
    """
    d, az = np.array(pmv.vdist(llh1[:, 0], llh1[:, 1], llh2[:, 0], llh2[:, 1]))

    return d


# Compute score
def calc_score(llh, llh_gt):
    """
    Args:
        llh : [latitude,longitude] (deg)
        llh_gt : [latitude,longitude] (deg)
    Returns:
        score : (m)
    """
    d = vincenty_distance(llh, llh_gt)
    score = np.mean([np.quantile(d, 0.50), np.quantile(d, 0.95)])

    return score


# Coordinate conversions (From https://github.com/commaai/laika)
a = 6378137 # 长半轴（单位：米）
b = 6356752.3142
esq = 6.69437999014 * 0.001
e1sq = 6.73949674228 * 0.001

def geodetic2ecef(geodetic, radians=False):
    geodetic = np.array(geodetic)
    input_shape = geodetic.shape
    geodetic = np.atleast_2d(geodetic)

    ratio = 1.0 if radians else (np.pi / 180.0)
    lat = ratio * geodetic[:, 0]
    lon = ratio * geodetic[:, 1]
    alt = geodetic[:, 2]

    xi = np.sqrt(1 - esq * np.sin(lat) ** 2)
    x = (a / xi + alt) * np.cos(lat) * np.cos(lon)
    y = (a / xi + alt) * np.cos(lat) * np.sin(lon)
    z = (a / xi * (1 - esq) + alt) * np.sin(lat)
    ecef = np.array([x, y, z]).T
    return ecef.reshape(input_shape)


################# Robust WLS  ####################
# I used soft_l1 loss function. It is robust, but its computation speed is considerably slower than that of ordinary least squares...
# GNSS single point positioning using pseudorange
def point_positioning(gnss_df):
    # Add nominal frequency to each signal
    # Note: GLONASS is an FDMA signal, so each satellite has a different frequency
    CarrierFrequencyHzRef = gnss_df.groupby(['Svid', 'SignalType'])[
        'CarrierFrequencyHz'].median()
    # CarrierFrequencyHzRef = CarrierFrequencyHzRef.to_frame(name='C')
    gnss_df = gnss_df.merge(CarrierFrequencyHzRef, how='left', on=[
        'Svid', 'SignalType'], suffixes=('', 'Ref'))  # 保留左侧 DataFrame（gnss_df）的所有行，并将右侧CarrierFrequencyHzRef的匹配行合并。
    gnss_df['CarrierErrorHz'] = np.abs(
        (gnss_df['CarrierFrequencyHz'] - gnss_df['CarrierFrequencyHzRef']))

    # Carrier smoothing
    gnss_df = carrier_smoothing(gnss_df)   # 先按周跳条件分组，然后求每组下的伪距-载波相位的差值，并计算这一组差值的平均值，作为载波相位的偏差，从而实现平滑化

    # GNSS single point positioning
    utcTimeMillis = gnss_df['utcTimeMillis'].unique()
    nepoch = len(utcTimeMillis)
    x0 = np.zeros(4)  # [x,y,z,tGPSL1]
    v0 = np.zeros(4)  # [vx,vy,vz,dtGPSL1]
    # x_wls = np.full([nepoch, 3], np.nan)  # For saving position
    # v_wls = np.full([nepoch, 3], np.nan)  # For saving velocity
    x_wls = np.full([nepoch, 3], np.nan)  # For saving position
    # v_wls = np.full([nepoch, 3], 0.0)  # For saving velocity
    v_wls = np.full([nepoch, 3], np.nan)  # For saving velocity from 0 to nan init
    cov_x = np.full([nepoch, 3, 3], np.nan)  # For saving position covariance
    cov_v = np.full([nepoch, 3, 3], np.nan)  # For saving velocity covariance

    # Loop for epochs
    for i, (t_utc, df) in enumerate(tqdm(gnss_df.groupby('utcTimeMillis'), total=nepoch)):
        # Valid satellite selection
        df_pr = satellite_selection(df, 'pr_smooth')
        df_prr = satellite_selection(df, 'PseudorangeRateMetersPerSecond')

        # Corrected pseudorange/pseudorange rate
        pr = (df_pr['pr_smooth'] + df_pr['SvClockBiasMeters'] - df_pr['IsrbMeters'] -
              df_pr['IonosphericDelayMeters'] - df_pr['TroposphericDelayMeters']).to_numpy()
        prr = (df_prr['PseudorangeRateMetersPerSecond'] +
               df_prr['SvClockDriftMetersPerSecond']).to_numpy()

        # Satellite position/velocity
        xsat_pr = df_pr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                         'SvPositionZEcefMeters']].to_numpy()
        xsat_prr = df_prr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                           'SvPositionZEcefMeters']].to_numpy()
        vsat = df_prr[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond',
                       'SvVelocityZEcefMetersPerSecond']].to_numpy()

        # Weight matrix for peseudorange/pseudorange rate
        Wx = np.diag(1 / df_pr['RawPseudorangeUncertaintyMeters'].to_numpy())
        Wv = np.diag(1 / df_prr['PseudorangeRateUncertaintyMetersPerSecond'].to_numpy())

        # Robust WLS requires accurate initial values for convergence,
        # so perform normal WLS for the first time
        if len(df_pr) >= 4:
            # Normal WLS
            if np.all(x0 == 0):
                # 这里调库解决最小二乘，似乎也不用线性化，而是直接求解非线性最小二乘问题，jac_pr_residuals在这里的作用？
                opt = scipy.optimize.least_squares(
                    pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx))
                x0 = opt.x
                # Robust WLS for position estimation
            opt = scipy.optimize.least_squares(
                pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx), loss='soft_l1')  # WLS的循环迭代步骤在这里完成
            if opt.status < 1 or opt.status == 2:  # opt.status < 1 表示最小二乘优化没有成功收敛到一个最优解；opt.status == 2 表示最小二乘优化收敛到一个次优解。
                print(f'i = {i} position lsq status = {opt.status}')
                x_wls[i, :] = opt.x[:3]
            else:
                # Covariance estimation
                cov = np.linalg.inv(opt.jac.T @ Wx @ opt.jac)
                cov_x[i, :, :] = cov[:3, :3]
                x_wls[i, :] = opt.x[:3]
                x0 = opt.x

        # Velocity estimation
        if len(df_prr) >= 4:
            if np.all(v0 == 0):  # Normal WLS
                opt = scipy.optimize.least_squares(
                    prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv))
                v0 = opt.x
            # Robust WLS for velocity estimation
            opt = scipy.optimize.least_squares(
                prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv), loss='soft_l1')
            if opt.status < 1:
                print(f'i = {i} velocity lsq status = {opt.status}')
                v_wls[i, :] = opt.x[:3]
            else:
                # Covariance estimation
                cov = np.linalg.inv(opt.jac.T @ Wv @ opt.jac)
                cov_v[i, :, :] = cov[:3, :3]
                v_wls[i, :] = opt.x[:3]
                v0 = opt.x

    return utcTimeMillis, x_wls, v_wls, cov_x, cov_v

def point_positioning_ntype(gnss_df):
    # Add nominal frequency to each signal
    # Note: GLONASS is an FDMA signal, so each satellite has a different frequency
    CarrierFrequencyHzRef = gnss_df.groupby(['Svid'])[
        'CarrierFrequencyHz'].median()
    gnss_df = gnss_df.merge(CarrierFrequencyHzRef, how='left', on=[
        'Svid'], suffixes=('', 'Ref'))
    gnss_df['CarrierErrorHz'] = np.abs(
        (gnss_df['CarrierFrequencyHz'] - gnss_df['CarrierFrequencyHzRef']))

    # Carrier smoothing
    gnss_df = carrier_smoothing(gnss_df)

    # GNSS single point positioning
    utcTimeMillis = gnss_df['utcTimeMillis'].unique()
    nepoch = len(utcTimeMillis)
    x0 = np.zeros(4)  # [x,y,z,tGPSL1]
    v0 = np.zeros(4)  # [vx,vy,vz,dtGPSL1]
    x_wls = np.full([nepoch, 3], np.nan)  # For saving position
    v_wls = np.full([nepoch, 3], np.nan)  # For saving velocity
    cov_x = np.full([nepoch, 3, 3], np.nan)  # For saving position covariance
    cov_v = np.full([nepoch, 3, 3], np.nan)  # For saving velocity covariance

    # Loop for epochs
    for i, (t_utc, df) in enumerate(tqdm(gnss_df.groupby('utcTimeMillis'), total=nepoch)):
        # Valid satellite selection
        df_pr = satellite_selection(df, 'pr_smooth')
        df_prr = satellite_selection(df, 'PseudorangeRateMetersPerSecond')

        # Corrected pseudorange/pseudorange rate
        pr = (df_pr['pr_smooth'] + df_pr['SvClockBiasMeters'] - df_pr['IsrbMeters'] -
              df_pr['IonosphericDelayMeters'] - df_pr['TroposphericDelayMeters']).to_numpy()
        prr = (df_prr['PseudorangeRateMetersPerSecond'] +
               df_prr['SvClockDriftMetersPerSecond']).to_numpy()

        # Satellite position/velocity
        xsat_pr = df_pr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                         'SvPositionZEcefMeters']].to_numpy()
        xsat_prr = df_prr[['SvPositionXEcefMeters', 'SvPositionYEcefMeters',
                           'SvPositionZEcefMeters']].to_numpy()
        vsat = df_prr[['SvVelocityXEcefMetersPerSecond', 'SvVelocityYEcefMetersPerSecond',
                       'SvVelocityZEcefMetersPerSecond']].to_numpy()

        # Weight matrix for peseudorange/pseudorange rate
        Wx = np.diag(1 / df_pr['RawPseudorangeUncertaintyMeters'].to_numpy())
        Wv = np.diag(1 / df_prr['PseudorangeRateUncertaintyMetersPerSecond'].to_numpy())

        # Robust WLS requires accurate initial values for convergence,
        # so perform normal WLS for the first time
        if len(df_pr) >= 4:
            # Normal WLS
            if np.all(x0 == 0):
                opt = scipy.optimize.least_squares(
                    pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx))
                x0 = opt.x
                # Robust WLS for position estimation
            opt = scipy.optimize.least_squares(
                pr_residuals, x0, jac_pr_residuals, args=(xsat_pr, pr, Wx), loss='soft_l1')
            if opt.status < 1 or opt.status == 2:
                print(f'i = {i} position lsq status = {opt.status}')
            else:
                # Covariance estimation
                cov = np.linalg.inv(opt.jac.T @ Wx @ opt.jac)
                cov_x[i, :, :] = cov[:3, :3]
                x_wls[i, :] = opt.x[:3]
                x0 = opt.x

        # Velocity estimation
        if len(df_prr) >= 4:
            if np.all(v0 == 0):  # Normal WLS
                opt = scipy.optimize.least_squares(
                    prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv))
                v0 = opt.x
            # Robust WLS for velocity estimation
            opt = scipy.optimize.least_squares(
                prr_residuals, v0, jac_prr_residuals, args=(vsat, prr, x0, xsat_prr, Wv), loss='soft_l1')
            if opt.status < 1:
                print(f'i = {i} velocity lsq status = {opt.status}')
            else:
                # Covariance estimation
                cov = np.linalg.inv(opt.jac.T @ Wv @ opt.jac)
                cov_v[i, :, :] = cov[:3, :3]
                v_wls[i, :] = opt.x[:3]
                v0 = opt.x

    return utcTimeMillis, x_wls, v_wls, cov_x, cov_v

# Simple outlier detection and interpolation
def exclude_interpolate_outlier(x_wls, v_wls, cov_x, cov_v):
    # Up velocity / height threshold
    v_up_th = 2.6  # m/s  2.0 -> 2.6
    height_th = 200.0  # m
    v_out_sigma = 3.0  # m/s
    x_out_sigma = 30.0  # m

    # Coordinate conversion
    x_llh = np.array(pm.ecef2geodetic(x_wls[:, 0], x_wls[:, 1], x_wls[:, 2])).T
    x_llh_mean = np.nanmean(x_llh, axis=0)
    v_enu = np.array(pm.ecef2enuv(
        v_wls[:, 0], v_wls[:, 1], v_wls[:, 2], x_llh_mean[0], x_llh_mean[1])).T

    # Up velocity jump detection
    # Cars don't jump suddenly!
    idx_v_out = np.abs(v_enu[:, 2]) > v_up_th
    idx_v_out |= np.isnan(v_enu[:, 2])
    v_wls[idx_v_out, :] = np.nan
    cov_v[idx_v_out] = v_out_sigma ** 2 * np.eye(3)
    outliernum_v=np.count_nonzero(idx_v_out)
    print(f'Number of velocity outliers {outliernum_v}')

    # Height check
    hmedian = np.nanmedian(x_llh[:, 2])
    idx_x_out = np.abs(x_llh[:, 2] - hmedian) > height_th
    idx_x_out |= np.isnan(x_llh[:, 2])
    x_wls[idx_x_out, :] = np.nan
    cov_x[idx_x_out] = x_out_sigma ** 2 * np.eye(3)
    outliernum_x=np.count_nonzero(idx_x_out)
    print(f'Number of position outliers {outliernum_x}')

    # Interpolate NaNs at beginning and end of array
    x_df = pd.DataFrame({'x': x_wls[:, 0], 'y': x_wls[:, 1], 'z': x_wls[:, 2]})
    x_df = x_df.interpolate(limit_area='outside', limit_direction='both')
    x_df = x_df.interpolate('spline', order=3)

    # Interpolate all NaN data
    v_df = pd.DataFrame({'x': v_wls[:, 0], 'y': v_wls[:, 1], 'z': v_wls[:, 2]})
    v_df = v_df.interpolate(limit_area='outside', limit_direction='both')
    v_df = v_df.interpolate('spline', order=3)

    return x_df.to_numpy(), v_df.to_numpy(), cov_x, cov_v#, outliernum_x, outliernum_v


###################  Kalman Smoother  ####################
# Application of Kalman smoother; integration of velocity and position obtained by WLS. The covariance matrix is a fixed value.
# Kalman filter
def Kalman_filter(zs, us, cov_zs, cov_us, phone):   # 状态量：位置zs，控制量：速度us
    # Parameters
    sigma_mahalanobis = 30.0  # Mahalanobis distance for rejecting innovation
    sigma_v = 0.6
    sigma_x = 6

    n, dim_x = zs.shape
    F = np.eye(3)  # Transition matrix
    H = np.eye(3)  # Measurement function

    # Initial state and covariance
    x = zs[0, :3].T  # State
    P = 5.0 ** 2 * np.eye(3)  # State covariance
    I = np.eye(dim_x)

    x_kf = np.zeros([n, dim_x])
    P_kf = np.zeros([n, dim_x, dim_x])

    # Kalman filtering
    for i, (u, z) in enumerate(zip(us, zs)):
        # First step
        if i == 0:
            x_kf[i] = x.T
            P_kf[i] = P
            continue

        # Prediction step
        # Q = cov_us[i]  # Estimated WLS velocity covariance
        Q = sigma_v ** 2 * np.eye(3)
        x = F @ x + u.T
        P = (F @ P) @ F.T + Q

        # Check outliers for observation
        d = distance.mahalanobis(z, H @ x, np.linalg.inv(P))

        # Update step
        if d < sigma_mahalanobis:
            # R = cov_zs[i]  # Estimated WLS position covariance
            R = sigma_x ** 2 * np.eye(3)
            y = z.T - H @ x
            S = (H @ P) @ H.T + R
            K = (P @ H.T) @ np.linalg.inv(S)
            x = x + K @ y
            P = (I - (K @ H)) @ P
        else:
            # If observation update is not available, increase covariance
            P += 10 ** 2 * Q

        x_kf[i] = x.T
        P_kf[i] = P

    return x_kf, P_kf

# Forward + backward Kalman filter and smoothing
def Kalman_smoothing(x_wls, v_wls, cov_x, cov_v, phone):
    n, dim_x = x_wls.shape

    # # For some unknown reason, the speed estimation is wrong only for XiaomiMi8
    # # so the variance is increased
    # if phone == 'XiaomiMi8':
    #     v_wls = np.vstack([(v_wls[:-1, :] + v_wls[1:, :]) / 2, np.zeros([1, 3])])
    #     cov_v = 1000.0 ** 2 * cov_v

    # Forward
    v = np.vstack([np.zeros([1, 3]), (v_wls[:-1, :] + v_wls[1:, :]) / 2])
    x_f, P_f = Kalman_filter(x_wls, v, cov_x, cov_v, phone)

    # Backward
    v = -np.flipud(v_wls)
    v = np.vstack([np.zeros([1, 3]), (v[:-1, :] + v[1:, :]) / 2])
    cov_xf = np.flip(cov_x, axis=0)
    cov_vf = np.flip(cov_v, axis=0)
    x_b, P_b = Kalman_filter(np.flipud(x_wls), v, cov_xf, cov_vf, phone)

    # Smoothing
    x_fb = np.zeros_like(x_f)
    P_fb = np.zeros_like(P_f)
    for (f, b) in zip(range(n), range(n - 1, -1, -1)):
        P_fi = np.linalg.inv(P_f[f])
        P_bi = np.linalg.inv(P_b[b])

        P_fb[f] = np.linalg.inv(P_fi + P_bi)
        x_fb[f] = P_fb[f] @ (P_fi @ x_f[f] + P_bi @ x_b[b])

    return x_fb, x_f, np.flipud(x_b)

def remove_nans(x_wls):
    nan_pos = np.argwhere(np.isnan(np.reshape(x_wls[:, 0], [len(x_wls[:, 0]), 1])))[:, 0]
    numnan = len(nan_pos)  # np.sum(np.isnan(x_wls))
    len_x = len(x_wls)
    x_wls_rnan = x_wls.copy()
    if numnan > 0 and numnan < 0.1 * len_x:
        cnt = 1
        for pos in nan_pos:
            if cnt < numnan:
                if pos > 2 and nan_pos[cnt] - pos > 2:
                    x_wls_rnan[pos, :] = (x_wls_rnan[pos - 2, :] + x_wls_rnan[pos - 1, :] + x_wls_rnan[pos + 1,
                                                                                            :] + x_wls_rnan[pos + 2,
                                                                                                 :]) / 4
                elif pos > 1 and nan_pos[cnt] - pos > 1:
                    x_wls_rnan[pos, :] = (x_wls_rnan[pos - 1, :] + x_wls_rnan[pos + 1, :]) / 2
            else:
                if pos > 2 and pos < len_x - 2:
                    x_wls_rnan[pos, :] = (x_wls_rnan[pos - 2, :] + x_wls_rnan[pos - 1, :] + x_wls_rnan[pos + 1,
                                                                                            :] + x_wls_rnan[pos + 2,
                                                                                                 :]) / 4
                elif pos > 1 and pos < len_x - 1:
                    x_wls_rnan[pos, :] = (x_wls_rnan[pos - 1, :] + x_wls_rnan[pos + 1, :]) / 2
            cnt += 1
    print(f'orig nans: {numnan}, removed nans to {np.sum(np.isnan(x_wls_rnan)) / np.shape(x_wls)[1]}')
    return x_wls_rnan

# Forward + backward Kalman filter and smoothing
def Kalman_smoothing_enuESKF(enu_kf, v_enu_wls, RM, df, cov_x, cov_v, phone):
    n, _ = enu_kf.shape
    # # For some unknown reason, the speed estimation is wrong only for XiaomiMi8
    # # so the variance is increased
    # if phone == 'XiaomiMi8':
    #     v_wls = np.vstack([(v_wls[:-1, :] + v_wls[1:, :]) / 2, np.zeros([1, 3])])
    #     cov_v = 1000.0 ** 2 * cov_v

    # Forward
    v = np.vstack([np.zeros([1, 3]), (v_enu_wls[:-1, :] + v_enu_wls[1:, :]) / 2])
    x_f, P_f, acc_enu_all = Error_State_Kalman_filter_enu(enu_kf, v, RM,  df, cov_x, cov_v, phone)

    return x_f , acc_enu_all #, x_f, np.flipud(x_b)

# Kalman filter
def Error_State_Kalman_filter_enu(zs, us, RM, df, cov_zs, cov_us, phone):
    # Parameters
    sigma_mahalanobis = 30.0  # Mahalanobis distance for rejecting innovation
    # if phone == 'pixel6pro' or phone == 'pixel4xl':
    #     sigma_v = 2
    # elif phone == 'pixel7pro':
    #     sigma_v = 5
    # else:
    #     sigma_v = 2
    sigma_v = 1
    sigma_x = 6
    dim_x = 9

    # IMU measurement
    accs = df[['MeasurementX_acc','MeasurementY_acc','MeasurementZ_acc']].values

    n, _ = zs.shape
    F = np.array([[1,0,0,1,0,0,0,0,0],
                  [0,1,0,0,1,0,0,0,0],
                  [0,0,1,0,0,1,0,0,0],
                  [0,0,0,1,0,0,1,0,0],
                  [0,0,0,0,1,0,0,1,0],
                  [0,0,0,0,0,1,0,0,1],
                  [0,0,0,0,0,0,1,0,0],
                  [0,0,0,0,0,0,0,1,0],
                  [0,0,0,0,0,0,0,0,1]])  # Transition matrix
    H = np.hstack((np.eye(6),np.zeros((6,3))))  # Measurement function

    # Initial state and covariance
    dx = np.zeros([dim_x, 1])  # error State
    p_imu = zs[0, :3].reshape((3, 1)) # wls的enu位置坐标初始化imu初始坐标
    v_imu = us[0, :3].reshape((3, 1)) # wls的enu速度初始化imu初始速度
    P = 5.0 ** 2 * np.eye(dim_x)  # State covariance
    I = np.eye(dim_x)

    ex_kf = np.zeros([n, 3])
    acc_enu_all = np.zeros([n, 3])
    P_kf = np.zeros([n, dim_x, dim_x])

    # Kalman filtering
    for i, (u, z, acc) in enumerate(zip(us, zs, accs)):
        # data reshape
        acc = acc.reshape((3, 1))
        acc = acc - dx[6:9]  # 补偿加速度误差
        z = z.reshape((3, 1))
        u = u.reshape((3, 1))
        # First step
        if i == 0:
            ex_kf[i] = p_imu.T
            P_kf[i] = P
            continue

        # imu Strapdown computation
        R_t = RM[i,:,:] # 旋转矩阵
        acc_enu = R_t @ acc
        v_imu = v_imu + acc_enu
        p_imu = p_imu + v_imu

        # Prediction step
        # Q = cov_us[i]  # Estimated WLS velocity covariance
        Q = sigma_v ** 2 * np.eye(dim_x)
        dx = F @ dx
        P = (F @ P) @ F.T + Q

        # Check outliers for observation
        dz = np.vstack((p_imu-z,v_imu-u)) # imu和gnss计算的位置和速度误差
        d = distance.mahalanobis(dz[:,0], H @ dx, np.linalg.inv(P[:6,:6]))

        # Update step
        if d < sigma_mahalanobis:
            # R = cov_zs[i]  # Estimated WLS position covariance
            R = sigma_x ** 2 * np.eye(6)
            R[2,2] = R[2,2] * 2
            y = dz - H @ dx
            S = (H @ P) @ H.T + R
            K = (P @ H.T) @ np.linalg.inv(S)
            dx = dx + K @ y
            P = (I - (K @ H)) @ P
        else:
            # If observation update is not available, increase covariance
            P += 10 ** 2 * Q

        # correct state
        p_imu = p_imu - dx[0:3]
        v_imu = v_imu - dx[3:6]

        ex_kf[i] = p_imu.T
        P_kf[i] = P
        acc_enu_all[i] = acc_enu.T

    return ex_kf, P_kf, acc_enu_all


def Kalman_smoothing_enuESKF_HF(enu_kf, v_enu_wls, RM_HF, df, raw_acc_df, cov_x, cov_v, phone):
    """
    高频加速度推算
    """
    n, _ = enu_kf.shape
    # # For some unknown reason, the speed estimation is wrong only for XiaomiMi8
    # # so the variance is increased
    # if phone == 'XiaomiMi8':
    #     v_wls = np.vstack([(v_wls[:-1, :] + v_wls[1:, :]) / 2, np.zeros([1, 3])])
    #     cov_v = 1000.0 ** 2 * cov_v

    # Forward
    v = np.vstack([np.zeros([1, 3]), (v_enu_wls[:-1, :] + v_enu_wls[1:, :]) / 2])
    x_f, P_f, raw_acc_df = Error_State_Kalman_filter_enu_HF(enu_kf, v, RM_HF, df, raw_acc_df, cov_x, cov_v, phone)

    return x_f, raw_acc_df  # , x_f, np.flipud(x_b)

def Error_State_Kalman_filter_enu_HF(zs, us, RM_HF, df, raw_acc_df, cov_zs, cov_us, phone):
    # Parameters
    sigma_mahalanobis = 30.0  # Mahalanobis distance for rejecting innovation

    # if phone == 'pixel6pro' or phone == 'pixel4xl':
    #     sigma_v = 0.3
    # elif phone == 'pixel7pro':
    #     sigma_v = 5
    # else:
    #     sigma_v = 2
    sigma_v = 1
    sigma_x = 6
    dim_x = 9
    outlier_acc = 9 # m/s^2
    outlier_acc_z = 5  # m/s^2

    # IMU measurement
    accs = df[['MeasurementX_acc','MeasurementY_acc','MeasurementZ_acc']].values
    raw_acc = raw_acc_df[['MeasurementX_acc','MeasurementY_acc','MeasurementZ_acc']].values
    raw_acc_utc = raw_acc_df['utcTimeMillis'].values
    merge_utc = df['utcTimeMillis'].values
    # 新增列用于存储加速度enu
    Acc_e_list,Acc_n_list,Acc_u_list = [],[],[]

    n, _ = zs.shape
    H = np.hstack((np.eye(6),np.zeros((6,3))))  # Measurement function

    # Initial state and covariance
    dx = np.zeros([dim_x, 1])  # error State
    p_imu = zs[0, :3].reshape((3, 1)) # wls的enu位置坐标初始化imu初始坐标
    v_imu = us[0, :3].reshape((3, 1)) # wls的enu速度初始化imu初始速度
    P = 5.0 ** 2 * np.eye(dim_x)  # State covariance
    I = np.eye(dim_x)

    ex_kf = np.zeros([n, 3])
    P_kf = np.zeros([n, dim_x, dim_x])
    i = 0
    ex_kf[i] = p_imu.T # 初始化状态 First step
    P_kf[i] = P
    # Kalman filtering
    acc_hf_pre = np.zeros([3, 1])
    i = i + 1

    for idx, (utc_hf, acc_hf) in enumerate(zip(raw_acc_utc, raw_acc)):
        # data reshape
        acc_hf = acc_hf.reshape((3, 1))
        acc_hf = acc_hf - dx[6:9] # 补偿加速度误差
        R_t = RM_HF[idx,:,:] # 旋转矩阵
        acc_enu = R_t @ acc_hf
        Acc_e_list.append(acc_enu[0,0])
        Acc_n_list.append(acc_enu[1,0])
        Acc_u_list.append(acc_enu[2,0])
        # outlier exclude
        # if np.sqrt(acc_hf[0]**2 + acc_hf[1]**2 + acc_hf[2]**2) > outlier_acc:
        #     acc_hf = acc_hf_pre
        # elif abs(acc_hf[2]) > outlier_acc_z:
        #     acc_hf[2] = acc_hf_pre[2]

        # 只有加速度计utc接近位置的utc才开始推算
        if (i==1) and abs(utc_hf - merge_utc[i])>13:
            acc_hf_pre = acc_hf
            continue
        if i > len(merge_utc)-1:
            break
        # imu Strapdown computation
        v_imu = v_imu + acc_enu * (utc_hf - raw_acc_utc[idx-1]) * 1e-3 # 时间差
        p_imu = p_imu + v_imu * (utc_hf - raw_acc_utc[idx-1]) * 1e-3

        # 如果时间戳对齐，进行KF
        # Prediction step
        # Q = cov_us[i]  # Estimated WLS velocity covariance
        if abs(utc_hf - merge_utc[i])<13:
            z = zs[i,:].reshape((3, 1))
            u = us[i,:].reshape((3, 1))
            Q = sigma_v ** 2 * np.eye(dim_x)
            tol = (merge_utc[i]-merge_utc[i-1]) * 1e-3
            F = np.array([[1, 0, 0, tol, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, tol, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, tol, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, tol, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, tol, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0, tol],
                          [0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1]])  # Transition matrix
            dx = F @ dx
            P = (F @ P) @ F.T + Q

            # Check outliers for observation
            dz = np.vstack((p_imu-z,v_imu-u)) # imu和gnss计算的位置和速度误差
            d = distance.mahalanobis(dz[:,0], H @ dx, np.linalg.inv(P[:6,:6]))

            # Update step
            if d < sigma_mahalanobis:
                # R = cov_zs[i]  # Estimated WLS position covariance
                R = sigma_x ** 2 * np.eye(6)
                R[2,2] = R[2,2] * 2
                y = dz - H @ dx
                S = (H @ P) @ H.T + R
                K = (P @ H.T) @ np.linalg.inv(S)
                dx = dx + K @ y
                P = (I - (K @ H)) @ P
            else:
                # If observation update is not available, increase covariance
                P += 10 ** 2 * Q

            # correct state
            p_imu = p_imu - dx[0:3]
            v_imu = v_imu - dx[3:6]
            ex_kf[i] = p_imu.T
            P_kf[i] = P
            i = i + 1
        acc_hf_pre = acc_hf

    raw_acc_df['Acc_e'] = Acc_e_list + [0] * (len(raw_acc_df) - len(Acc_e_list))
    raw_acc_df['Acc_n'] = Acc_n_list + [0] * (len(raw_acc_df) - len(Acc_n_list))
    raw_acc_df['Acc_u'] = Acc_u_list + [0] * (len(raw_acc_df) - len(Acc_u_list))
    cols_to_drop = ['MeasurementX_acc','MeasurementY_acc','MeasurementZ_acc','acc_index']
    raw_acc_df.drop(columns=cols_to_drop, inplace=True)
    return ex_kf, P_kf, raw_acc_df

def Kalman_smoothing_enuESKF_v2_HF(enu_kf, v_enu_wls, df, raw_acc_df, raw_mag_df, RM_HF, phone):
    """
    加入姿态误差的推算
    """
    n, _ = enu_kf.shape
    # Forward
    v = np.vstack([np.zeros([1, 3]), (v_enu_wls[:-1, :] + v_enu_wls[1:, :]) / 2])
    x_f, P_f, raw_acc_df = Error_State_Kalman_filter_enu_v2_HF(enu_kf, v, df, raw_acc_df, raw_mag_df, RM_HF, phone)
    return x_f, raw_acc_df  # , x_f, np.flipud(x_b)

def Error_State_Kalman_filter_enu_v2_HF(zs, us, df, raw_acc_df, raw_mag_df, RM_HF,  phone):
    """
        加入姿态误差的推算
    """
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

    # Parameters
    sigma_mahalanobis = 30.0  # Mahalanobis distance for rejecting innovation

    # if phone == 'pixel6pro' or phone == 'pixel4xl':
    #     sigma_v = 0.3
    # elif phone == 'pixel7pro':
    #     sigma_v = 5
    # else:
    #     sigma_v = 2
    sigma_v = 1
    sigma_x = 6
    dim_x = 12 # 位置误差 速度误差 姿态误差 加速度计偏差
    outlier_acc = 9 # m/s^2
    outlier_acc_z = 5  # m/s^2

    # IMU measurement
    raw_acc = raw_acc_df[['MeasurementX_acc','MeasurementY_acc','MeasurementZ_acc']].values
    raw_mag = raw_mag_df[['MeasurementX_mag','MeasurementY_mag','MeasurementZ_mag']].values
    raw_acc_utc = raw_acc_df['utcTimeMillis'].values
    raw_mag_utc = raw_mag_df['utcTimeMillis'].values
    merge_utc = df['utcTimeMillis'].values
    # 新增列用于存储加速度enu
    Acc_e_list,Acc_n_list,Acc_u_list = [],[],[]

    n, _ = zs.shape
    H = np.hstack((np.eye(6),np.zeros((6,dim_x-6))))  # Measurement function

    # Initial state and covariance
    dx = np.zeros([dim_x, 1])  # error State
    p_imu = zs[0, :3].reshape((3, 1)) # wls的enu位置坐标初始化imu初始坐标
    v_imu = us[0, :3].reshape((3, 1)) # wls的enu速度初始化imu初始速度
    P = 5.0 ** 2 * np.eye(dim_x)  # State covariance
    I = np.eye(dim_x)

    ex_kf = np.zeros([n, 3])
    P_kf = np.zeros([n, dim_x, dim_x])
    i = 0 # eskf 初始idx
    idx_mag = 0 # 陀螺仪初始idx
    ex_kf[i] = p_imu.T # 初始化状态 First step
    P_kf[i] = P
    acc_hf_pre = np.zeros([3, 1])
    IM = np.eye(3) # 3维单位阵，用于拼接
    ZM = np.zeros([3, 3])
    i = i + 1
    cnt = 0
    # 计算当地重力
    # ecef_wls = np.array(pm.enu2ecef(zs[0,0], zs[0,1], zs[0,2],df.loc[0, 'LatitudeDegrees'], df.loc[0, 'LongitudeDegrees'],
    #                                 df.loc[0, 'AltitudeMeters']))
    # llh_wls = np.array(pm.ecef2geodetic(ecef_wls[0], ecef_wls[1], ecef_wls[2]))
    # g_acc = gravity_at_height(llh_wls[0],llh_wls[2])
    F_dic = {}

    for idx, (utc_acc_hf, acc_hf) in enumerate(zip(raw_acc_utc, raw_acc)):
        # data reshape
        acc_hf = acc_hf.reshape((3, 1))
        # acc_hf = acc_hf[2,0] - g_acc # z轴减去当地重力
        acc_hf = acc_hf - dx[9:12] # 加速度计偏差补偿
        while (raw_mag_utc[idx_mag]-utc_acc_hf) < -5: # mag的时间比加速度慢
            idx_mag = idx_mag + 1
            cnt = cnt + 1
        if (idx_mag - cnt == 0) and abs(utc_acc_hf - raw_mag_utc[idx_mag]) > 5:
            # 如没磁力计数据，skip
            continue
        elif abs(utc_acc_hf - raw_mag_utc[idx_mag])<=5:
            mag_x, mag_y, mag_z = raw_mag[idx_mag,0],raw_mag[idx_mag,1],raw_mag[idx_mag,2]
            heading = np.arctan2(mag_y, mag_x) * (180 / np.pi) - 90
            roll, pitch = get_roll_pitch_from_accelerometer(acc_hf[0,0], acc_hf[1,0], acc_hf[2,0])
            # TODO 是否需要补偿姿态误差
            # pitch = pitch - dx[6,0]
            # roll = roll - dx[7,0]
            # heading = heading - dx[8,0]
            R_t = cal_rotation_matrix(roll, pitch, heading)
            idx_mag = idx_mag + 1
        elif (idx_mag > 0) and abs(utc_acc_hf - raw_mag_utc[idx_mag])>5:
            roll, pitch = get_roll_pitch_from_accelerometer(acc_hf[0,0], acc_hf[1,0], acc_hf[2,0])
            # pitch = pitch - dx[6,0]
            # roll = roll - dx[7,0]
            R_t = cal_rotation_matrix(roll, pitch, heading)

        # R_t = RM_HF[idx,:,:]
        # TODO 是否需要补偿加速度计误差
        acc_enu = R_t @ acc_hf
        # acc_enu = acc_enu - dx[9:12]  # 加速度计偏差补偿
        Acc_e_list.append(acc_enu[0, 0])
        Acc_n_list.append(acc_enu[1, 0])
        Acc_u_list.append(acc_enu[2, 0])
        # outlier exclude
        # if np.sqrt(acc_hf[0]**2 + acc_hf[1]**2 + acc_hf[2]**2) > outlier_acc:
        #     acc_hf = acc_hf_pre
        # elif abs(acc_hf[2]) > outlier_acc_z:
        #     acc_hf[2] = acc_hf_pre[2]

        # 只有加速度计utc接近位置的utc才开始推算
        if (i==1) and utc_acc_hf - merge_utc[i] < -13:
            acc_hf_pre = acc_hf
            continue
        if i > len(merge_utc)-1:
            break
        # imu Strapdown computation
        v_imu = v_imu + acc_enu * (utc_acc_hf - raw_acc_utc[idx-1]) * 1e-3 # 时间差
        p_imu = p_imu + v_imu * (utc_acc_hf - raw_acc_utc[idx-1]) * 1e-3

        # 如果时间戳对齐，进行KF
        # Prediction step
        # Q = cov_us[i]  # Estimated WLS velocity covariance
        if abs(utc_acc_hf - merge_utc[i])<13:
            # 观测信息提取
            z = zs[i,:].reshape((3, 1))  # 位置
            u = us[i,:].reshape((3, 1))  # 速度
            Q = sigma_v ** 2 * np.eye(dim_x)

            # update RM RN
            ecef_wls = np.array(pm.enu2ecef(z[0], z[1], z[2],
                        df.loc[0, 'LatitudeDegrees'], df.loc[0, 'LongitudeDegrees'],df.loc[0, 'AltitudeMeters']))
            llh_wls = np.array(pm.ecef2geodetic(ecef_wls[0], ecef_wls[1], ecef_wls[2]))
            RM = calculate_RM(llh_wls[0])  # 计算曲率半径
            RN = calculate_RN(llh_wls[0])

            # 计算状态矩阵
            Fv = np.array([[0,acc_enu[2,0],-acc_enu[1,0]],
                           [-acc_enu[2,0],0,acc_enu[0,0]],
                           [acc_enu[1,0],-acc_enu[0,0],0]])
            lat_rad = math.radians(z[0,0])  # 将纬度转为弧度
            tan_lat = math.tan(lat_rad)
            Fd = np.array([[0,  1/(RM+z[2,0]),  0],
                           [-1/(RN+z[2,0]), 0,  0],
                           [-tan_lat/(RN+z[2,0]),   0,  0]])

            tol = (merge_utc[i]-merge_utc[i-1]) * 1e-3 # 时间间隔
            F = np.block([[IM, IM*tol, ZM,    ZM],
                          [ZM, IM,     Fv*tol,R_t*tol],
                          [ZM, Fd*tol,IM,    ZM],
                          [ZM, ZM,     ZM,    IM]])

            # da 为enu坐标下的误差
            # 加速度误差不进行坐标转换
            F_re = np.block([[IM, IM*tol, ZM,    ZM],
                          [ZM, IM,     Fv*tol,IM*tol],
                          [ZM, Fd*tol,IM,    ZM],
                          [ZM, ZM,     ZM,    IM]])

            dx = F @ dx
            P = (F @ P) @ F.T + Q

            # Check outliers for observation
            dz = np.vstack((p_imu-z,v_imu-u)) # imu和gnss计算的位置和速度误差
            d = distance.mahalanobis(dz[:,0], H @ dx, np.linalg.inv(P[:6,:6]))

            # Update step
            if d < sigma_mahalanobis:
                # R = cov_zs[i]  # Estimated WLS position covariance
                R = sigma_x ** 2 * np.eye(6)
                R[2,2] = R[2,2] * 2 # 高程噪声更大
                # R[3:5, 3:5] = R[3:5, 3:5] * 0.5
                y = dz - H @ dx
                S = (H @ P) @ H.T + R
                K = (P @ H.T) @ np.linalg.inv(S)
                dx = dx + K @ y
                P = (I - (K @ H)) @ P
            else:
                # If observation update is not available, increase covariance
                P += 10 ** 2 * Q

            # correct state
            p_imu = p_imu - dx[0:3]
            v_imu = v_imu - dx[3:6]
            ex_kf[i] = p_imu.T
            P_kf[i] = P
            i = i + 1

        acc_hf_pre = acc_hf

    raw_acc_df['Acc_e'] = Acc_e_list + [0] * (len(raw_acc_df) - len(Acc_e_list))
    raw_acc_df['Acc_n'] = Acc_n_list + [0] * (len(raw_acc_df) - len(Acc_n_list))
    raw_acc_df['Acc_u'] = Acc_u_list + [0] * (len(raw_acc_df) - len(Acc_u_list))
    cols_to_drop = ['MeasurementX_acc','MeasurementY_acc','MeasurementZ_acc','acc_index']
    raw_acc_df.drop(columns=cols_to_drop, inplace=True)
    return ex_kf, P_kf, raw_acc_df

def Kalman_smoothing_enuESKF_ALL_HF(enu_kf, v_enu_wls, df, raw_acc_df, raw_mag_df, raw_gyr_df, RM_HF, phone):
    """
    加速度 陀螺仪 磁力计融合 高频
    """
    n, _ = enu_kf.shape
    # Forward
    v = np.vstack([np.zeros([1, 3]), (v_enu_wls[:-1, :] + v_enu_wls[1:, :]) / 2])
    x_f, P_f, df_imuALL = Error_State_Kalman_filter_enu_ALL_HFV2(enu_kf, v, df, raw_acc_df, raw_mag_df, raw_gyr_df, RM_HF, phone)
    return x_f, df_imuALL  # , x_f, np.flipud(x_b)

def Error_State_Kalman_filter_enu_ALL_HFV2(zs, us, df, raw_acc_df, raw_mag_df, raw_gyr_df, RM_HF,  phone):
    """
    加速度 陀螺仪 磁力计融合 高频, 最常规做法
    """
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

    def calculate_att(raw_gyr,idx_gyr,roll,pitch,yaw,err_pitch,err_roll,err_yaw):
        # gyr_x, gyr_y, gyr_z = raw_gyr[idx_gyr, 0]-err_roll, raw_gyr[idx_gyr, 1]-err_pitch,raw_gyr[idx_gyr, 2]-err_yaw
        gyr_x, gyr_y, gyr_z = raw_gyr[idx_gyr, 0]-err_pitch, raw_gyr[idx_gyr, 1]-err_roll,raw_gyr[idx_gyr, 2]-err_yaw
        if idx_gyr == 0:
            # roll = roll + gyr_x * 1e-3
            # pitch = pitch + gyr_y * 1e-3
            roll = roll + gyr_y * 1e-3
            pitch = pitch + gyr_x * 1e-3
            yaw = yaw + gyr_z * 1e-3
        else:
            # roll = roll + gyr_x * (raw_gyr_utc[idx_gyr] - raw_gyr_utc[idx_gyr - 1]) * 1e-3
            # pitch = pitch + gyr_y * (raw_gyr_utc[idx_gyr] - raw_gyr_utc[idx_gyr - 1]) * 1e-3
            roll = roll + gyr_y * (raw_gyr_utc[idx_gyr] - raw_gyr_utc[idx_gyr - 1]) * 1e-3
            pitch = pitch + gyr_x * (raw_gyr_utc[idx_gyr] - raw_gyr_utc[idx_gyr - 1]) * 1e-3
            yaw = yaw + gyr_z * (raw_gyr_utc[idx_gyr] - raw_gyr_utc[idx_gyr - 1]) * 1e-3
        return roll, pitch, yaw

    # Parameters
    sigma_mahalanobis = 30.0  # Mahalanobis distance for rejecting innovation

    # if phone == 'pixel6pro' or phone == 'pixel4xl':
    #     sigma_v = 0.3
    # elif phone == 'pixel7pro':
    #     sigma_v = 5
    # else:
    #     sigma_v = 2
    sigma_v = 3
    sigma_x = 6
    dim_x = 15 # 位置误差 速度误差 姿态误差 磁力计偏差 加速度计偏差
    dim_z = 7
    outlier_acc = 9 # m/s^2
    outlier_acc_z = 5  # m/s^2

    # IMU measurement
    df_imu = pd.merge_asof(raw_acc_df,raw_gyr_df,left_on=['utcTimeMillis'],right_on=['utcTimeMillis'],direction='nearest',tolerance=5).dropna()
    df_imuALL = pd.merge_asof(df_imu,raw_mag_df,left_on=['utcTimeMillis'],right_on=['utcTimeMillis'],direction='nearest',tolerance=5).dropna()
    raw_acc = df_imu[['MeasurementX_acc','MeasurementY_acc','MeasurementZ_acc']].values
    raw_gyr = df_imu[['MeasurementX_gyr','MeasurementY_gyr','MeasurementZ_gyr']].values
    raw_mag = raw_mag_df[['MeasurementX_mag','MeasurementY_mag','MeasurementZ_mag']].values
    raw_acc_utc = df_imu['utcTimeMillis'].values
    raw_gyr_utc = df_imu['utcTimeMillis'].values
    raw_mag_utc = raw_mag_df['utcTimeMillis'].values
    merge_utc = df['utcTimeMillis'].values
    df_imuALL.drop(columns=['acc_index','gyr_index','mag_index'], inplace=True)
    # 新增列用于存储加速度enu
    Acc_e_list,Acc_n_list,Acc_u_list = [],[],[]
    Gyr_e_list,Gyr_n_list,Gyr_u_list = [],[],[]

    n, _ = zs.shape
    H = np.hstack((np.eye(6),np.zeros((6,dim_x-6))))  # Measurement function
    H = np.vstack((H,np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])))

    # Initial state and covariance
    dx = np.zeros([dim_x, 1])  # error State
    p_imu = zs[0, :3].reshape((3, 1)) # wls的enu位置坐标初始化imu初始坐标
    v_imu = us[0, :3].reshape((3, 1)) # wls的enu速度初始化imu初始速度
    roll, pitch, yaw = 0, 0, 0 # 初始化横滚角 俯仰角
    P = 5.0 ** 2 * np.eye(dim_x)  # State covariance
    I = np.eye(dim_x)

    ex_kf = np.zeros([n, 3])
    P_kf = np.zeros([n, dim_x, dim_x])
    i = 0 # eskf 初始idx
    idx_mag = 0 # 磁力计仪初始idx
    ex_kf[i] = p_imu.T # 初始化状态 First step
    P_kf[i] = P
    acc_hf_pre = np.zeros([3, 1])
    IM = np.eye(3) # 3维单位阵，用于拼接
    ZM = np.zeros([3, 3])
    i = i + 1
    cnt = 0

    mag_x, mag_y, mag_z = raw_mag[idx_mag, 0], raw_mag[idx_mag, 1], raw_mag[idx_mag, 2]
    heading_init = np.arctan2(mag_x, mag_y)
    yaw = yaw+heading_init
    for idx, (utc_acc_hf, acc_hf) in enumerate(zip(raw_acc_utc, raw_acc)):
        # data reshape
        acc_hf = acc_hf.reshape((3, 1))
        # acc_hf = acc_hf[2,0] - g_acc # z轴减去当地重力
        acc_hf = acc_hf - dx[9:12] # 加速度计偏差补偿
        if idx_mag >= len(raw_mag_utc): # 超出磁力计边界
            idx_mag -= 1
        else:
            while (raw_mag_utc[idx_mag]-utc_acc_hf) < -5: # mag的时间比加速度慢
                idx_mag = idx_mag + 1
                cnt = cnt + 1

        if (idx_mag - cnt == 0) and abs(utc_acc_hf - raw_mag_utc[idx_mag]) > 5:
            # 如没磁力计数据，skip
            roll, pitch, yaw = calculate_att(raw_gyr, idx, roll, pitch, yaw, dx[12,0], dx[13,0], dx[14,0])
            continue
        elif abs(utc_acc_hf - raw_mag_utc[idx_mag])<=5:
            # 陀螺仪和加速度计时间对准
            roll, pitch, yaw = calculate_att(raw_gyr, idx, roll, pitch, yaw, dx[12,0], dx[13,0], dx[14,0])
            # R_rp = cal_rotation_matrix_rp(roll, pitch)
            # raw_mag_rt = R_rp @ raw_mag[idx_mag].T
            # mag_x, mag_y, mag_z = raw_mag_rt[0], raw_mag_rt[1], raw_mag_rt[2]
            mag_x, mag_y, mag_z = raw_mag[idx_mag, 0], raw_mag[idx_mag, 1], raw_mag[idx_mag, 2]
            heading =  np.arctan2(mag_x, mag_y)   # np.arctan2(mag_y, mag_x) - np.pi/2 # np.arctan2(-mag_y, mag_x) np.arctan2(mag_x, mag_y)
            R_t = cal_rotation_matrix(roll, pitch, yaw)
            idx_mag = idx_mag + 1
        elif (idx_mag > 0) and abs(utc_acc_hf - raw_mag_utc[idx_mag])>5:
            roll, pitch, yaw = calculate_att(raw_gyr, idx, roll, pitch, yaw, dx[12,0], dx[13,0], dx[14,0])
            R_t = cal_rotation_matrix(roll, pitch, yaw)

        # R_t = RM_HF[idx,:,:]
        # TODO 是否需要补偿加速度计误差
        acc_enu = R_t @ acc_hf
        # acc_enu = acc_enu - dx[9:12]  # 加速度计偏差补偿
        Acc_e_list.append(acc_enu[0, 0])
        Acc_n_list.append(acc_enu[1, 0])
        Acc_u_list.append(acc_enu[2, 0])
        # outlier exclude
        # if np.sqrt(acc_hf[0]**2 + acc_hf[1]**2 + acc_hf[2]**2) > outlier_acc:
        #     acc_hf = acc_hf_pre
        # elif abs(acc_hf[2]) > outlier_acc_z:
        #     acc_hf[2] = acc_hf_pre[2]

        # 只有加速度计utc接近位置的utc才开始推算
        if (i==1) and utc_acc_hf - merge_utc[i] < -13:
            acc_hf_pre = acc_hf
            continue
        if i > len(merge_utc)-1:
            break
        # imu Strapdown computation
        v_imu = v_imu + acc_enu * (utc_acc_hf - raw_acc_utc[idx-1]) * 1e-3 # 时间差
        p_imu = p_imu + v_imu * (utc_acc_hf - raw_acc_utc[idx-1]) * 1e-3

        # 如果时间戳对齐，进行KF
        # Prediction step
        # Q = cov_us[i]  # Estimated WLS velocity covariance
        if abs(utc_acc_hf - merge_utc[i])<13:
            # 观测信息提取
            z = zs[i,:].reshape((3, 1))  # 位置
            u = us[i,:].reshape((3, 1))  # 速度
            Q = sigma_v ** 2 * np.eye(dim_x)

            # update RM RN
            ecef_wls = np.array(pm.enu2ecef(z[0], z[1], z[2],
                        df.loc[0, 'LatitudeDegrees'], df.loc[0, 'LongitudeDegrees'],df.loc[0, 'AltitudeMeters']))
            llh_wls = np.array(pm.ecef2geodetic(ecef_wls[0], ecef_wls[1], ecef_wls[2]))
            RM = calculate_RM(llh_wls[0])  # 计算曲率半径
            RN = calculate_RN(llh_wls[0])

            # 计算状态矩阵
            Fv = np.array([[0,acc_enu[2,0],-acc_enu[1,0]],
                           [-acc_enu[2,0],0,acc_enu[0,0]],
                           [acc_enu[1,0],-acc_enu[0,0],0]])
            lat_rad = math.radians(z[0,0])  # 将纬度转为弧度
            tan_lat = math.tan(lat_rad)
            Fd = np.array([[0,  1/(RM+z[2,0]),  0],
                           [-1/(RN+z[2,0]), 0,  0],
                           [-tan_lat/(RN+z[2,0]),   0,  0]])

            tol = (merge_utc[i]-merge_utc[i-1]) * 1e-3 # 时间间隔
            F = np.block([[IM, IM*tol, ZM,    ZM,         ZM],
                          [ZM, IM,     Fv*tol,R_t*tol,    ZM],
                          [ZM, Fd*tol,IM,    ZM,          R_t*tol], # R_t*tol
                          [ZM, ZM,     ZM,    IM,         ZM],
                          [ZM, ZM,     ZM,    ZM,         IM]])

            dx = F @ dx
            P = (F @ P) @ F.T + Q

            # Check outliers for observation
            dz = np.vstack((p_imu-z,v_imu-u)) # imu和gnss计算的位置和速度误差
            dz = np.vstack((dz,yaw-heading))
            hdx = H @ dx
            d = distance.mahalanobis(dz[:6,0], hdx[:6,0], np.linalg.inv(P[:6,:6]))

            # Update step
            if d < sigma_mahalanobis:
                # R = cov_zs[i]  # Estimated WLS position covariance
                R = sigma_x ** 2 * np.eye(dim_z)
                R[2,2] = R[2,2] * 2 # 高程噪声更大
                R[6,6] = R[6, 6] * 3  # 高程噪声更大
                # R[3:5, 3:5] = R[3:5, 3:5] * 0.5
                y = dz - H @ dx
                S = (H @ P) @ H.T + R
                K = (P @ H.T) @ np.linalg.inv(S)
                dx = dx + K @ y
                P = (I - (K @ H)) @ P
            else:
                # If observation update is not available, increase covariance
                P += 10 ** 2 * Q

            # correct state
            p_imu = p_imu - dx[0:3]
            v_imu = v_imu - dx[3:6]
            pitch = pitch - dx[6,0]
            roll = roll - dx[7,0]
            yaw = pitch - dx[8,0]
            ex_kf[i] = p_imu.T
            P_kf[i] = P
            i = i + 1

        acc_hf_pre = acc_hf

    raw_acc_df['Acc_e'] = Acc_e_list + [0] * (len(raw_acc_df) - len(Acc_e_list))
    raw_acc_df['Acc_n'] = Acc_n_list + [0] * (len(raw_acc_df) - len(Acc_n_list))
    raw_acc_df['Acc_u'] = Acc_u_list + [0] * (len(raw_acc_df) - len(Acc_u_list))
    raw_gyr_df['Gyr_e'] = Gyr_e_list + [0] * (len(raw_gyr_df) - len(Gyr_e_list))
    raw_gyr_df['Gyr_n'] = Gyr_n_list + [0] * (len(raw_gyr_df) - len(Gyr_n_list))
    raw_gyr_df['Gyr_u'] = Gyr_u_list + [0] * (len(raw_gyr_df) - len(Gyr_u_list))
    # cols_to_drop = ['MeasurementX_acc','MeasurementY_acc','MeasurementZ_acc','acc_index']
    # cols_to_drop_gyr = ['MeasurementX_gyr', 'MeasurementY_gyr', 'MeasurementZ_gyr', 'gyr_index']
    # raw_acc_df.drop(columns=cols_to_drop, inplace=True)
    # raw_gyr_df.drop(columns=cols_to_drop_gyr, inplace=True)
    # raw_gyr_df.drop(columns=cols_to_drop_gyr, inplace=True)
    return ex_kf, P_kf, df_imuALL

def Error_State_Kalman_filter_enu_ALL_HF(zs, us, df, raw_acc_df, raw_mag_df, raw_gyr_df, RM_HF,  phone):
    """
    加速度 陀螺仪 磁力计融合 高频
    """
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

    def calculate_att(raw_gyr,idx_gyr,roll,pitch,err_pitch,err_roll):
        gyr_x, gyr_y = raw_gyr[idx_gyr, 0]-err_pitch, raw_gyr[idx_gyr, 1]-err_roll
        if idx_gyr == 0:
            roll = roll + gyr_y * 1e-3
            pitch = pitch + gyr_x * 1e-3
        else:
            roll = roll + gyr_y * (raw_gyr_utc[idx_gyr] - raw_gyr_utc[idx_gyr - 1]) * 1e-3
            pitch = pitch + gyr_x * (raw_gyr_utc[idx_gyr] - raw_gyr_utc[idx_gyr - 1]) * 1e-3
        return roll, pitch

    # Parameters
    sigma_mahalanobis = 30.0  # Mahalanobis distance for rejecting innovation

    # if phone == 'pixel6pro' or phone == 'pixel4xl':
    #     sigma_v = 0.3
    # elif phone == 'pixel7pro':
    #     sigma_v = 5
    # else:
    #     sigma_v = 2
    sigma_v = 3
    sigma_x = 6
    dim_x = 14 # 位置误差 速度误差 姿态误差 磁力计偏差 加速度计偏差
    outlier_acc = 9 # m/s^2
    outlier_acc_z = 5  # m/s^2

    # IMU measurement
    raw_acc = raw_acc_df[['MeasurementX_acc','MeasurementY_acc','MeasurementZ_acc']].values
    raw_gyr = raw_gyr_df[['MeasurementX_gyr','MeasurementY_gyr','MeasurementZ_gyr']].values
    raw_mag = raw_mag_df[['MeasurementX_mag','MeasurementY_mag','MeasurementZ_mag']].values
    raw_acc_utc = raw_acc_df['utcTimeMillis'].values
    raw_gyr_utc = raw_acc_df['utcTimeMillis'].values
    raw_mag_utc = raw_mag_df['utcTimeMillis'].values
    merge_utc = df['utcTimeMillis'].values
    # 新增列用于存储加速度enu
    Acc_e_list,Acc_n_list,Acc_u_list = [],[],[]
    Gyr_e_list,Gyr_n_list,Gyr_u_list = [],[],[]

    n, _ = zs.shape
    H = np.hstack((np.eye(6),np.zeros((6,dim_x-6))))  # Measurement function

    # Initial state and covariance
    dx = np.zeros([dim_x, 1])  # error State
    p_imu = zs[0, :3].reshape((3, 1)) # wls的enu位置坐标初始化imu初始坐标
    v_imu = us[0, :3].reshape((3, 1)) # wls的enu速度初始化imu初始速度
    roll, pitch = 0, 0 # 初始化横滚角 俯仰角
    P = 5.0 ** 2 * np.eye(dim_x)  # State covariance
    I = np.eye(dim_x)

    ex_kf = np.zeros([n, 3])
    P_kf = np.zeros([n, dim_x, dim_x])
    i = 0 # eskf 初始idx
    idx_mag = 0 # 磁力计仪初始idx
    idx_gyr = 0 # 陀螺仪初始idx
    ex_kf[i] = p_imu.T # 初始化状态 First step
    P_kf[i] = P
    acc_hf_pre = np.zeros([3, 1])
    IM = np.eye(3) # 3维单位阵，用于拼接
    IM_22 = np.eye(2) # 2维单位阵，用于拼接
    ZM = np.zeros([3, 3])
    ZM_32 = np.zeros([3, 2])  # 2维单位阵，用于拼接
    ZM_23 = np.zeros([2, 3])  # 2维单位阵，用于拼接
    i = i + 1
    cnt = 0
    # 计算当地重力
    # ecef_wls = np.array(pm.enu2ecef(zs[0,0], zs[0,1], zs[0,2],df.loc[0, 'LatitudeDegrees'], df.loc[0, 'LongitudeDegrees'],
    #                                 df.loc[0, 'AltitudeMeters']))
    # llh_wls = np.array(pm.ecef2geodetic(ecef_wls[0], ecef_wls[1], ecef_wls[2]))
    # g_acc = gravity_at_height(llh_wls[0],llh_wls[2])
    F_dic = {}

    for idx, (utc_acc_hf, acc_hf) in enumerate(zip(raw_acc_utc, raw_acc)):
        # data reshape
        acc_hf = acc_hf.reshape((3, 1))
        # acc_hf = acc_hf[2,0] - g_acc # z轴减去当地重力
        acc_hf = acc_hf - dx[9:12] # 加速度计偏差补偿
        while (raw_mag_utc[idx_mag]-utc_acc_hf) < -5: # mag的时间比加速度慢
            idx_mag = idx_mag + 1
            cnt = cnt + 1

        if (idx_mag - cnt == 0) and abs(utc_acc_hf - raw_mag_utc[idx_mag]) > 5:
            # 如没磁力计数据，skip
            if abs(utc_acc_hf - raw_gyr_utc[idx_gyr]) <= 5:
                roll, pitch = calculate_att(raw_gyr, idx_gyr, roll, pitch, dx[12,0], dx[13,0])
                idx_gyr = idx_gyr + 1
            continue
        elif abs(utc_acc_hf - raw_mag_utc[idx_mag])<=5:
            # 陀螺仪和加速度计时间对准
            if abs(utc_acc_hf - raw_gyr_utc[idx_gyr]) <= 5:
                roll, pitch = calculate_att(raw_gyr, idx_gyr, roll, pitch, dx[12,0], dx[13,0])
                idx_gyr += 1
            # TODO 是否需要补偿姿态误差
            # pitch = pitch - dx[6,0]
            # roll = roll - dx[7,0]
            # heading = heading - dx[8,0]
            R_rp = cal_rotation_matrix_rp(roll, pitch)
            raw_mag_rt = R_rp @ raw_mag[idx_mag].T
            mag_x, mag_y, mag_z = raw_mag_rt[0],raw_mag_rt[1],raw_mag_rt[2]
            heading = np.arctan2(-mag_y, mag_x) * (180 / np.pi) # - 90

            R_t = cal_rotation_matrix(roll, pitch, heading)
            idx_mag = idx_mag + 1
        elif (idx_mag > 0) and abs(utc_acc_hf - raw_mag_utc[idx_mag])>5:
            if abs(utc_acc_hf - raw_gyr_utc[idx_gyr]) <= 5:
                roll, pitch = calculate_att(raw_gyr, idx_gyr, roll, pitch, dx[12,0], dx[13,0])
                idx_gyr += 1
            roll_acc, pitch_acc = get_roll_pitch_from_accelerometer_v2(acc_hf[0,0], acc_hf[1,0], acc_hf[2,0])
            # pitch = pitch - dx[6,0]
            # roll = roll - dx[7,0]
            R_t = cal_rotation_matrix(roll, pitch, heading)

        # R_t = RM_HF[idx,:,:]
        # TODO 是否需要补偿加速度计误差
        acc_enu = R_t @ acc_hf
        # acc_enu = acc_enu - dx[9:12]  # 加速度计偏差补偿
        Acc_e_list.append(acc_enu[0, 0])
        Acc_n_list.append(acc_enu[1, 0])
        Acc_u_list.append(acc_enu[2, 0])
        # outlier exclude
        # if np.sqrt(acc_hf[0]**2 + acc_hf[1]**2 + acc_hf[2]**2) > outlier_acc:
        #     acc_hf = acc_hf_pre
        # elif abs(acc_hf[2]) > outlier_acc_z:
        #     acc_hf[2] = acc_hf_pre[2]

        # 只有加速度计utc接近位置的utc才开始推算
        if (i==1) and utc_acc_hf - merge_utc[i] < -13:
            acc_hf_pre = acc_hf
            continue
        if i > len(merge_utc)-1:
            break
        # imu Strapdown computation
        v_imu = v_imu + acc_enu * (utc_acc_hf - raw_acc_utc[idx-1]) * 1e-3 # 时间差
        p_imu = p_imu + v_imu * (utc_acc_hf - raw_acc_utc[idx-1]) * 1e-3

        # 如果时间戳对齐，进行KF
        # Prediction step
        # Q = cov_us[i]  # Estimated WLS velocity covariance
        if abs(utc_acc_hf - merge_utc[i])<13:
            # 观测信息提取
            z = zs[i,:].reshape((3, 1))  # 位置
            u = us[i,:].reshape((3, 1))  # 速度
            Q = sigma_v ** 2 * np.eye(dim_x)

            # update RM RN
            ecef_wls = np.array(pm.enu2ecef(z[0], z[1], z[2],
                        df.loc[0, 'LatitudeDegrees'], df.loc[0, 'LongitudeDegrees'],df.loc[0, 'AltitudeMeters']))
            llh_wls = np.array(pm.ecef2geodetic(ecef_wls[0], ecef_wls[1], ecef_wls[2]))
            RM = calculate_RM(llh_wls[0])  # 计算曲率半径
            RN = calculate_RN(llh_wls[0])

            # 计算状态矩阵
            Fv = np.array([[0,acc_enu[2,0],-acc_enu[1,0]],
                           [-acc_enu[2,0],0,acc_enu[0,0]],
                           [acc_enu[1,0],-acc_enu[0,0],0]])
            lat_rad = math.radians(z[0,0])  # 将纬度转为弧度
            tan_lat = math.tan(lat_rad)
            Fd = np.array([[0,  1/(RM+z[2,0]),  0],
                           [-1/(RN+z[2,0]), 0,  0],
                           [-tan_lat/(RN+z[2,0]),   0,  0]])

            tol = (merge_utc[i]-merge_utc[i-1]) * 1e-3 # 时间间隔
            F = np.block([[IM, IM*tol, ZM,    ZM,         ZM_32],
                          [ZM, IM,     Fv*tol,R_t*tol,    ZM_32],
                          [ZM, Fd*tol,IM,    ZM,          R_t[:,0:-1]*tol],
                          [ZM, ZM,     ZM,    IM,         ZM_32],
                          [ZM_23, ZM_23,     ZM_23,    ZM_23,         IM_22]])

            dx = F @ dx
            P = (F @ P) @ F.T + Q

            # Check outliers for observation
            dz = np.vstack((p_imu-z,v_imu-u)) # imu和gnss计算的位置和速度误差
            d = distance.mahalanobis(dz[:,0], H @ dx, np.linalg.inv(P[:6,:6]))

            # Update step
            if d < sigma_mahalanobis:
                # R = cov_zs[i]  # Estimated WLS position covariance
                R = sigma_x ** 2 * np.eye(6)
                R[2,2] = R[2,2] * 2 # 高程噪声更大
                # R[3:5, 3:5] = R[3:5, 3:5] * 0.5
                y = dz - H @ dx
                S = (H @ P) @ H.T + R
                K = (P @ H.T) @ np.linalg.inv(S)
                dx = dx + K @ y
                P = (I - (K @ H)) @ P
            else:
                # If observation update is not available, increase covariance
                P += 10 ** 2 * Q

            # correct state
            p_imu = p_imu - dx[0:3]
            v_imu = v_imu - dx[3:6]
            ex_kf[i] = p_imu.T
            P_kf[i] = P
            i = i + 1

        acc_hf_pre = acc_hf

    raw_acc_df['Acc_e'] = Acc_e_list + [0] * (len(raw_acc_df) - len(Acc_e_list))
    raw_acc_df['Acc_n'] = Acc_n_list + [0] * (len(raw_acc_df) - len(Acc_n_list))
    raw_acc_df['Acc_u'] = Acc_u_list + [0] * (len(raw_acc_df) - len(Acc_u_list))
    raw_gyr_df['Gyr_e'] = Gyr_e_list + [0] * (len(raw_gyr_df) - len(Gyr_e_list))
    raw_gyr_df['Gyr_n'] = Gyr_n_list + [0] * (len(raw_gyr_df) - len(Gyr_n_list))
    raw_gyr_df['Gyr_u'] = Gyr_u_list + [0] * (len(raw_gyr_df) - len(Gyr_u_list))
    cols_to_drop = ['MeasurementX_acc','MeasurementY_acc','MeasurementZ_acc','acc_index']
    cols_to_drop_gyr = ['MeasurementX_gyr', 'MeasurementY_gyr', 'MeasurementZ_gyr', 'gyr_index']
    raw_acc_df.drop(columns=cols_to_drop, inplace=True)
    raw_gyr_df.drop(columns=cols_to_drop_gyr, inplace=True)
    return ex_kf, P_kf, raw_acc_df

# ------------------根据加速度计计算横滚角与仰俯角---------------------
def get_roll_pitch_from_accelerometer(ax, ay, az):
    magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    pitch = np.arctan2(ax , np.sqrt(ay**2+az**2)) # np.arcsin(ax / magnitude) # 俯仰角 重力算
    roll = np.arctan2(ay, az) # 横滚
    return roll, pitch

# ------------------根据加速度计计算横滚角与仰俯角---------------------
def get_roll_pitch_from_accelerometer_v2(ax, ay, az):
    magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    pitch = np.arctan2(ay , np.sqrt(ax**2+az**2)) # np.arcsin(ax / magnitude) # 俯仰角 重力算
    roll = np.arctan2(ax, az) # 横滚
    return roll, pitch

# -----------------旋转矩阵--------------#
def cal_rotation_matrix(roll, pitch, yaw):
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch) # cos俯仰角
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])
    return R

# -----------------旋转矩阵（roll pitch）--------------#
def cal_rotation_matrix_rp(roll, pitch):
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch) # cos俯仰角
    sp = np.sin(pitch)
    R = np.array([
        [cp, sp*sr, sp*cr],
        [0, cr, -sp],
        [-sp,cp*sr,cp*cr]
    ])
    return R

# 计算某经纬度重力
def gravity_at_height(lat_deg, height_m):
    import math
    # 海平面重力公式（纬度lat_deg）
    phi = math.radians(lat_deg)
    g0 = 9.780318 * (1 + 0.0053024 * math.sin(phi)**2 - 0.0000058 * math.sin(2*phi)**2)
    # 高度修正（地球半径R=6371e3 m）
    R = 6371000
    gh = g0 * (1 - (2 * height_m) / R)
    return gh

# ------------------将Body坐标系下的速度转成ENU下的速度，再转成经纬高的变化率---------------------
def euler_to_rotation_matrix(rolls, pitchs, yaws):
    Rs = np.zeros([len(rolls),3,3])
    for idx, (roll,pitch,yaw) in enumerate(zip(rolls, pitchs, yaws)):
        R = cal_rotation_matrix(roll, pitch, yaw)
        Rs[idx,:,:] = R
    return Rs

def get_rotation_matrix_high_freq(acc_,mag_):
    """
    计算高频加速度旋转矩阵
    :param acc_: raw acc data with high frequency
    :param mag_: raw mag data with high frequency
    :param df_temp: merge data
    :return: rotation_matrix
    """
    df_temp = pd.merge_asof(
        mag_,
        acc_,
        left_on=['utcTimeMillis'],
        right_on=['utcTimeMillis'],
        direction='nearest',
        tolerance=5)

    df_temp = compensate(df_temp, ['MeasurementX_acc', 'MeasurementY_acc', 'MeasurementZ_acc'], max_consecutive)
    df_temp = compensate(df_temp, ['MeasurementX_mag', 'MeasurementY_mag', 'MeasurementZ_mag'], max_consecutive)
    raw_acc = acc_[['MeasurementX_acc','MeasurementY_acc','MeasurementZ_acc']].values
    merge_acc = df_temp[['MeasurementX_acc','MeasurementY_acc','MeasurementZ_acc']].values
    raw_acc_utc = acc_['utcTimeMillis'].values
    merge_utc = df_temp['utcTimeMillis'].values
    # raw_mag = mag_[['MeasurementX_mag','MeasurementY_mag','MeasurementZ_mag']].values
    i = 0
    cnt = 0
    Rs = np.zeros([len(raw_acc), 3, 3])
    for idx in range(len(raw_acc)):
        raw_acc_x,raw_acc_y,raw_acc_z = raw_acc[idx,0],raw_acc[idx,1],raw_acc[idx,2]
        utc_acc_ = raw_acc_utc[idx]
        while (merge_utc[i]-utc_acc_) < -5: # mag的时间比加速度慢
            i = i + 1
            cnt = cnt + 1
        if (i - cnt == 0) and abs(utc_acc_ - merge_utc[i])>5:
            continue
        elif abs(utc_acc_ - merge_utc[i])<=5:
            mag_x = df_temp.loc[i,'MeasurementX_mag']
            mag_y = df_temp.loc[i,'MeasurementY_mag']
            mag_z = df_temp.loc[i,'MeasurementZ_mag']
            heading = np.arctan2(mag_y, mag_x) * (180 / np.pi) - 90
            roll, pitch = get_roll_pitch_from_accelerometer(raw_acc_x, raw_acc_y, raw_acc_z)
            R = cal_rotation_matrix(roll, pitch, heading)
            Rs[idx, :, :] = R
            i = i+1
            if i > len(merge_utc)-1:
                break
        elif (i > 0) and abs(utc_acc_ - merge_utc[i])>5:
            roll, pitch = get_roll_pitch_from_accelerometer(raw_acc_x, raw_acc_y, raw_acc_z)
            R = cal_rotation_matrix(roll, pitch, heading)
            Rs[idx, :, :] = R
    return Rs

def compensate(df, cols_to_process, max_consecutive=4):
    # cols_to_process = ['MeasurementX', 'MeasurementY', 'MeasurementZ']
    for col in cols_to_process:
        data = df[col]
        na_indices = data.index[data.isna()].tolist()

        if not na_indices:
            continue

        # 计算连续空缺段
        segments = []
        start_idx = na_indices[0]
        prev_idx = na_indices[0]

        for idx in na_indices[1:]:
            if idx == prev_idx + 1:
                prev_idx = idx
            else:
                segments.append((start_idx, prev_idx))
                start_idx = idx
                prev_idx = idx
        segments.append((start_idx, prev_idx))

        for start, end in segments:
            length = end - start + 1
            if length > max_consecutive:
                # print(f"列 {col}：连续空缺超过{max_consecutive}，该轨迹剔除")
                raise ValueError(f"连续空缺{length},超过最大容忍{max_consecutive},轨迹排除")
                continue

            pre_idx = start - 1
            post_idx = end + 1

            if pre_idx < 0:
                if post_idx < len(data) and not pd.isna(data.iloc[post_idx]):
                    fill_value = data.iloc[post_idx]
                    for fill_idx in range(start, end + 1):
                        df.at[data.index[fill_idx], col] = fill_value
                continue
            if post_idx >= len(data):
                if pre_idx >= 0 and not pd.isna(data.iloc[pre_idx]):
                    fill_value = data.iloc[pre_idx]
                    for fill_idx in range(start, end + 1):
                        df.at[data.index[fill_idx], col] = fill_value
                continue

            val_start = data.iloc[pre_idx]
            val_end = data.iloc[post_idx]

            if pd.isna(val_start):
                if not pd.isna(val_end):
                    fill_value = val_end
                    for fill_idx in range(start, end + 1):
                        df.at[data.index[fill_idx], col] = fill_value
                continue
            if pd.isna(val_end):
                if not pd.isna(val_start):
                    fill_value = val_start
                    for fill_idx in range(start, end + 1):
                        df.at[data.index[fill_idx], col] = fill_value
                continue

            if length == 1:
                interpolated_value = (val_start + val_end) / 2
                df.at[data.index[start], col] = interpolated_value
            else:
                n = length
                for k in range(n):
                    interpolated_value = val_start + (k / (n - 1)) * (val_end - val_start)
                    df.at[data.index[start + k], col] = interpolated_value

    return df

def rts_smoothing(data, noise_variance=1.0):
    """
    对一维数据应用 RTS 平滑

    :param data: 一维输入数据，numpy数组
    :param noise_variance: 噪声方差
    :return: 平滑后的数据
    """
    n = len(data)
    smoothed_data = np.zeros(n)

    # 初始化状态
    smoothed_data[-1] = data[-1]

    # 后向递推
    for i in range(n - 2, -1, -1):
        # 计算平滑值
        smoothed_data[i] = data[i] + (noise_variance / (noise_variance + 1)) * (smoothed_data[i + 1] - data[i])

    return smoothed_data

################# 没什么用的函数 ################
def fx(x, u):
    # x[e,n,yaw]
    # u[velocity、Angle velocity]
    res = np.zeros(3)
    res[0] = x[0] + (u[0]) * np.cos(x[2])  # x
    res[1] = x[1] + (u[0]) * np.sin(x[2])  # y

    if u[0] <= 20:
        th = 360
    elif u[0] <= 30:
        th = 40
    else:
        th = 10

    if np.rad2deg(u[1]) > th:
        u[1] = np.deg2rad(th)
    if np.rad2deg(u[1]) < - th:
        u[1] = np.deg2rad(- th)
    res[2] = x[2] + u[1]

    return res

def jacobian_fx(x,u):
    v = u[0]
    gry = u[1]#角速度
    theta = x[2]
    return np.array ([[1, 0, -v*np.sin(theta)],
                      [0, 1, v*np.cos(theta)],
                      [0, 0, 1]
                     ])
#%%
def jacobian_fu(x,u):
    v = u[0]
    gry = u[1]#角速度
    theta = x[2]
    return np.array ([[np.cos(theta),0],
                      [np.sin(theta),0],
                      [0, 1]
                     ])

def rotation_R(df):
    mean_ = df[:100].mean()
    ax = - mean_.MeasurementZ
    ay = - mean_.MeasurementX
    az =   mean_.MeasurementY
    roll = (np.arctan2(ay,az))
    pitch = (-np.arctan2(ax,np.sqrt(ay ** 2 + az ** 2)))
    yaw = 0
    c_roll = np.array([[1,             0,             0],
                       [0,  np.cos(roll), -np.sin(roll)],
                       [0,  np.sin(roll),  np.cos(roll)]
                      ])

    c_pitch = np.array([[ np.cos(pitch), 0,  np.sin(pitch)],
                        [ 0,             1,              0],
                        [-np.sin(pitch), 0,  np.cos(pitch)]
                       ])

    c_yaw = np.array([[ np.cos(yaw),-np.sin(yaw),      0],
                      [ np.sin(yaw), np.cos(yaw),      0],
                      [ 0,             0,              1]
                     ])
    R = c_yaw @ c_pitch @ c_roll
    return R

def Kalman_filter_imu(zs, us):
    # Parameters
    sigma_v = 0.6
    sigma_x = 5  # position SD m
    sigma_mahalanobis = 30.0  # Mahalanobis distance for rejecting innovation

    n, dim_x = zs.shape
    Q = sigma_v ** 2 * np.eye(2)  # Process noise

    H = np.eye(3)  # Measurement function
    R = sigma_x ** 2 * np.eye(3)  # Measurement noise

    # Initial state and covariance
    x = zs[0, :3].T  # State
    P = sigma_x ** 2 * np.eye(3)  # State covariance
    I = np.eye(dim_x)

    x_kf = np.zeros([n, dim_x])
    P_kf = np.zeros([n, dim_x, dim_x])

    # Kalman filtering
    for i, (u, z) in enumerate(zip(us, zs)):
        # First step
        if i == 0:
            x_kf[i] = x.T
            P_kf[i] = P
            continue

        # Prediction step
        # add non linear

        F = jacobian_fx(x, u)
        #         print(F.shape,P.shape)
        G = jacobian_fu(x, u)
        temp1 = (G @ Q)
        temp = (G @ Q) @ G.T
        P = (F @ P) @ F.T + (G @ Q) @ G.T

        x = fx(x, u)
        # Check outliers for observation
        d = distance.mahalanobis(z, H @ x, np.linalg.pinv(P))

        # Update step
        if d < sigma_mahalanobis:
            y = z.T - H @ x
            S = (H @ P) @ H.T + R
            K = (P @ H.T) @ np.linalg.inv(S)
            x = x + K @ y
            P = (I - (K @ H)) @ P
        else:
            print()
            # If no observation update is available, increase covariance
            P += 10 ** 2 * ((G @ Q) @ G.T)

        x_kf[i] = x.T
        P_kf[i] = P

    return x_kf, P_kf

# Forward + backward Kalman filter and smoothing
def Kalman_smoothing_imu(en, u):
    n, dim_x = en.shape

    # Forward
    u_f = np.vstack([np.zeros([1, 2]), (u[:-1, :] + u[1:, :]) / 2])
    x_f, P_f = Kalman_filter_imu(en, u_f)
    # Backward
    u = -np.flipud(u)
    u = np.vstack([np.zeros([1, 2]), (u[:-1, :] + u[1:, :]) / 2])
    x_b, P_b = Kalman_filter_imu(np.flipud(en), u)

    # Smoothing
    x_fb = np.zeros_like(x_f)
    P_fb = np.zeros_like(P_f)
    for (f, b) in zip(range(n), range(n - 1, -1, -1)):
        P_fi = np.linalg.inv(P_f[f])
        P_bi = np.linalg.inv(P_b[b])

        P_fb[f] = np.linalg.inv(P_fi + P_bi)
        x_fb[f] = P_fb[f] @ (P_fi @ x_f[f] + P_bi @ x_b[b])

    return x_fb, x_f, np.flipud(x_b)
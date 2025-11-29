# normalize parameters
# for 170 datasets lat [37.7770066, 34.1405127] lon [-118.315262, -122.4331823]
lat_min=34.1405127
lat_max=37.7770066
lon_min=-122.4331823
lon_max=-118.315262
# 79 trajs
e_min=-2711818.517
n_max=-2493465.106
u_min=-4678380.194
e_max=-4269105.318
n_min=3534265.74
u_max=3883909.253

# res_min=-200.55
# res_max=500.0
res_min=-50.55
res_max=100.0
losx_min=-1.0
losx_max=0.91
losy_min=-1.0
losy_max=0.876
losz_min=-0.82
losz_max=0.963

# trajectory lengh
traj_len=50
outlayer_in_end=113
outlayer_in_end_ecef=0

CN0_min=20
CN0_max=53
CN0_minmax=15.800
CN0_maxmin=45.400
CNO_distmin=21.282
CNO_distmax=46.283

PRU_min=0.600
PRU_max=149
PRU_minmax=2.099
PRU_maxmin=54.262
PRU_distmin=0.897
PRU_distmax=21.152

AA_min=0.000
AA_max=359.999
AA_minmax=42.508
AA_maxmin=317.554
AA_distmin=24.960
AA_distmax=338.317

EA_min=-61.278
EA_max=89.142
EA_minmax=12.501
EA_maxmin=67.262
EA_distmin=7.675
EA_distmax=68.257

Prate_min=-875.295
Prate_max=952.739
Prate_minmax=-470.863
Prate_maxmin=655.793
Prate_distmin=-558.267
Prate_distmax=654.926

# 噪声协方差矩阵最值
convv_max= 1000#3501827086565.716
convv_min= -1000#-6823022406278.384
convx_max= 100#10605742.696252795
convx_min= -100#-2661118.7431458216
convv_mean = -79626.21224073913
convv_std = 8645909676.225294

convv_avg_max=1750913543282.858
convv_avg_min=-3411511203139.1914
convx_avg_max=1428.4134553658132
convx_avg_min=-900.2340305669218

convv_diff_mean=-2.4641378567418964e-07
convv_diff_std=116905.95845066001
convx_diff_mean=-0.008068290505206906
convx_diff_std=16286.589339202266

convx_avg_mean=46.11167512983407
convx_avg_std=8204.165332079643
convv_avg_mean=67.85854605946786
convv_avg_std=58452.94755101869

CLIGHT = 299_792_458   # speed of light (m/s)
RE_WGS84 = 6_378_137
OMGE = 7.2921151467E-5  # earth angular velocity (IS-GPS) (rad/s)
g_acc = 9.80665 # Gravitational acceleration
E_2 = 0.00669437999014 # 第一偏心率

a = 6378137 # 长半轴（单位：米）
b = 6356752.3142
esq = 6.69437999014 * 0.001
e1sq = 6.73949674228 * 0.001
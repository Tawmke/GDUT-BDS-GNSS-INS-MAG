# GDUT-BDS-GNSS-INS-MAG
---
author: Jianhao Tang
date: 2025-11-29
changes:
  - "2025-11-29: Jianhao Tang  Initialize file"
---

## Project structure
```
project-root/
├── README.md     # This file
├── env/          # Trajectory type file
├── envRLKF/      # Data processing and DRL environment
├── model/        # Model class
├── funcs/        # Functions 
├── train/        # Raw GNSS measurement data
├── ...
GNSS-INS-MAG-processing.py
rl_control_KFcustom_lospos.py

1) [GNSS/INS/MAG baseline model（GNSS-INS-MAG-processing.py）](./GNSS-INS-MAG-processing.py)
   > Generate the raw GNSS/INS/MAG results and process GNSS data.
2) [DRL-AMTKF (rl_control_KFcustom_lospos.py)](./rl_control_KFcustom_lospos.py)
   > Adaptive measurement/process noise mitigation and covariance tuning by a single agent.
2) [MADRL-AMTKF (rl_control_KFcustom_losposconv_V4multi.py)](./rl_control_KFcustom_losposconv_V4multi.py)
   > Adaptive measurement and process noise mitigation and covariance tuning by two agents simultaneously.

## 1. Data Processing
### 1）Step 1: Process the raw GNSS measurement data in the dataset
Download data in https://www.kaggle.com/competitions/smartphone-decimeter-2023. And then run [carrier-smoothing-robust-wls-kalman-smoother_single.py](./carrier-smoothing-robust-wls-kalman-smoother_single.py), and then solved files will be generated in the corrsponding file folders, such as the position and velocity results by WLS and KF.

### 2) Step 2: Generate GNSS feature data
Run [env_processing_multic_CN0ele_mutipath.py](./envRLKF/env_processing_multic_CN0ele_mutipath.py) to generate the processed GNSS feature data.
Such as the solved position data (`raw_baseline_multic_re.pkl`), the solved noise covariance data (`processed_ecef_covariance_wls.pkl`),
the solved velocity data dictionary`raw_baseline_velocity.pkl`, GNSS feature data `processed_features_multic_RLAKF.pkl`.

## 2. Model Training and Testing
You can run the code [rl_control_KFcustom_lospos.py](./rl_control_KFcustom_lospos.py) and [rl_control_KFcustom_losposconv_V4multi.py](./rl_control_KFcustom_losposconv_V4multi.py) for training and 
testing, and the result statistics will be saved as .csv files. 

In addition, you can save the trajectoy results 
by setting `traj_record=True` in the function `recording_results_ecef_RL4KF`.

## 3. Result Statistics
Run [traj_CDF.py](./plots/MADRL%20ADKF/traj_CDF.py) to generate the CDF results and the average results from the 
saved trajtory results. Run [traj_test_plot_RLAKF.py](./plots/MADRL%20ADKF/traj_test_plot_RLAKF.py) to generate the positioning error trajectory.

## 4. Environment configuration
You can use [environment.yml](environment.yml) to install the experimental environment. It should be noted that the version of stable_baseline3 can only be 1.6.2 and python=3.8.10.


## Reference
 ```
@article{tang2024improving,
  title={Improving GNSS positioning correction using deep reinforcement learning with an adaptive reward augmentation method},
  author={Tang, Jianhao and Li, Zhenni and Hou, Kexian and Li, Peili and Zhao, Haoli and Wang, Qianming and Liu, Ming and Xie, Shengli},
  journal={NAVIGATION: Journal of the Institute of Navigation},
  volume={71},
  number={4},
  year={2024},
  publisher={Institute of Navigation}
}
@article{zhao2024improving,
  title={Improving performances of GNSS positioning correction using multiview deep reinforcement learning with sparse representation},
  author={Zhao, Haoli and Li, Zhenni and Wang, Qianming and Xie, Kan and Xie, Shengli and Liu, Ming and Chen, Ci},
  journal={GPS Solutions},
  volume={28},
  number={3},
  pages={98},
  year={2024},
  publisher={Springer}
}
 ```


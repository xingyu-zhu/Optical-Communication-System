"""optsim: 光纤通信全流程仿真（相干 DP-QAM / WDM 为主，兼顾 IM-DD 与光纤传感）。

模块
----
utils    常数、单位、滤波、重采样、时延、AWGN
params   波段/光纤参数（读取 params/fiber_bands.json）
tx       比特、星座、LLR、PCS、成帧、脉冲成形
devices  DAC/ADC、激光器、调制器、放大器、滤波器、WDM 合波、偏振器件、PD、相干接收机
channel  色散、Manakov SSFM（SPM/XPM/FWM/SRS/自陡峭/PMD）、多跨链路、GN 模型
dsp      接收 DSP 算法库
metrics  BER/SNR/EVM/GMI/MI 等
system   Config、tx_signal、wdm_tx/wdm_rx、rx_dsp
plots    星座图、功率谱、眼图
nn_eq    神经网络均衡器 NNEqualizer（MLP/CNN/BiLSTM）与学习型 DBP（需 torch，按需导入：from optsim.nn_eq import ...）
"""
from . import utils, params, tx, devices, channel, dsp, metrics

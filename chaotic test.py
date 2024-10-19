import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import lag_plot

# 读取CSV文件
df = pd.read_csv('cleaned_station_2_flow_data1-25.csv')

# 选择出站客流 flow_out
flow_out = df['flow_out']

# 1. 自相关函数（ACF）分析
plt.figure(dpi=300,figsize=(10, 6))
plot_acf(flow_out, lags=50)
plt.title("Autocorrelation of Dataset")
plt.show()

# 2. 计算Lyapunov指数以检测混沌性
def lyapunov_exponent(time_series):
    N = len(time_series)
    delta_t = 1  # 时间步长
    diffs = np.abs(np.diff(time_series))  # 计算时间序列的差分
    positive_diffs = diffs[diffs > 0]  # 仅考虑正差分
    if len(positive_diffs) == 0:
        return 0  # 无正差分表示无混沌
    log_diffs = np.log(positive_diffs)
    return np.mean(log_diffs) / delta_t  # Lyapunov指数的近似值

lyap_exp = lyapunov_exponent(flow_out)
print(f"Lyapunov Exponent: {lyap_exp}")

# 3. 计算关联维数
def correlation_dimension(time_series, m=3, tau=1, radius=0.1):
    N = len(time_series)
    tree = KDTree(time_series[:, np.newaxis])
    count = 0
    for i in range(N):
        count += len(tree.query_ball_point(time_series[i], radius))
    return np.log(count / N)

from scipy.spatial import KDTree
cd = correlation_dimension(flow_out.values)
print(f"Correlation Dimension: {cd}")

# 4. 伪吸引子重构（延迟时间嵌入法）
plt.figure(dpi=300,figsize=(10, 6))
lag_plot(flow_out)
plt.title("Lag Plot of Data (Pseudo Attractor)")
plt.show()

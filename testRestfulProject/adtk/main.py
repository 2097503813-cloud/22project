import matplotlib.pyplot as plt
import pandas as pd
from adtk.detector import ThresholdAD
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import QuantileAD
from adtk.detector import GeneralizedESDTestAD
from adtk.detector import PersistAD
from adtk.detector import LevelShiftAD
from adtk.detector import VolatilityShiftAD
from adtk.detector import SeasonalAD
from adtk.detector import AutoregressionAD
from adtk.detector import MinClusterDetector
from sklearn.cluster import KMeans
from adtk.detector import OutlierDetector
from sklearn.neighbors import LocalOutlierFactor
from adtk.detector import PcaAD

df = pd.read_csv('dataset/cpu.csv', index_col="Time", parse_dates=True)
s = df.squeeze()  # 将单列 DataFrame 转换为 Series
s = validate_series(s)
'''ThresholdAD'''
'''

threshold_ad = ThresholdAD(high=30, low=15)
anomalies = threshold_ad.detect(s)

p=plot(s, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker");
# 添加标题和标签
plt.title('时间序列异常检测', fontsize=16, fontweight='bold', pad=20)

plt.show()
'''
'''QuantileAD'''
'''
from adtk.detector import QuantileAD
quantile_ad = QuantileAD(high=0.99, low=0.01)
anomalies = quantile_ad.fit_detect(s)
plot(s, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker");
plt.show()
'''
'''InterQuartileRangeAD'''
from adtk.detector import InterQuartileRangeAD
#iqr_ad = InterQuartileRangeAD(c=1.5)
#anomalies = iqr_ad.fit_detect(s)
#threshold_ad = ThresholdAD(high=30, low=15)
#anomalies = threshold_ad.detect(s)
#quantile_ad = QuantileAD(high=0.99, low=0.01)
#anomalies = quantile_ad.fit_detect(s)
#esd_ad = GeneralizedESDTestAD(alpha=0.3)
#anomalies = esd_ad.fit_detect(s)
#persist_ad = PersistAD(c=3.0, side='positive')
#anomalies = persist_ad.fit_detect(s)
#level_shift_ad = LevelShiftAD(c=6.0, side='both', window=5)
#anomalies = level_shift_ad.fit_detect(s)
#volatility_shift_ad = VolatilityShiftAD(c=6.0, side='positive', window=30)
#anomalies = volatility_shift_ad.fit_detect(s)
#seasonal_ad = SeasonalAD(c=3.0, side="both")
#anomalies = seasonal_ad.fit_detect(s)
#autoregression_ad = AutoregressionAD(n_steps=7*2, step_size=24, c=3.0)
#anomalies = autoregression_ad.fit_detect(s)
#min_cluster_detector = MinClusterDetector(KMeans(n_clusters=3))
#anomalies = min_cluster_detector.fit_detect(df)
#outlier_detector = OutlierDetector(LocalOutlierFactor(contamination=0.05))
#anomalies = outlier_detector.fit_detect(df)
pca_ad = PcaAD(k=1)
anomalies = pca_ad.fit_detect(df)

plot(s, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")
plt.show()

 
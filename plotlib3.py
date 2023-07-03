import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
# 读取CSV文件并加载数据
data_c2fg2 = pd.read_csv('runs/train/exp7/results2.csv')
data_c3g2 = pd.read_csv('runs/train/exp8/results2.csv')
data_v5s = pd.read_csv('runs/train/exp9/results2.csv')

# 提取"loss"和"epoch"的数据列
loss_c2fg2 = data_c2fg2['train/box_loss'] + data_c2fg2['train/obj_loss']
loss_c3g2 = data_c3g2['train/box_loss'] + data_c3g2['train/obj_loss']
loss_v5s = data_v5s['train/box_loss'] + data_v5s['train/obj_loss']
map_c2fg2 = data_c2fg2['metrics/mAP_0.5']
map_c3g2 = data_c3g2['metrics/mAP_0.5']
map_v5s = data_v5s['metrics/mAP_0.5']
epochs = data_v5s['epoch']


y_smoothed = gaussian_filter1d(loss_c2fg2, sigma=5)
# 绘制折线图
plt.plot(epochs, map_c2fg2, color='#42d5ff', label='YOLO-GX')
plt.plot(epochs, map_c3g2, color='red', label='YOLO-Ghost')
plt.plot(epochs, map_v5s, color='blue', label='YOLOV5')
plt.plot(epochs, y_smoothed, label='Interpolated Curve',color = 'yellow')
# 提取150-200之间的数据
focus_epochs = epochs[(epochs >= 180) & (epochs <= 200)]
focus_map_c2fg2 = map_c2fg2[(epochs >= 180) & (epochs <= 200)]
focus_map_c3g2 = map_c3g2[(epochs >= 180) & (epochs <= 200)]
focus_map_v5s = map_v5s[(epochs >= 180) & (epochs <= 200)]



# 添加标题和标签
plt.title('mAP')
plt.xlabel('Epochs')
plt.ylabel('mAP')
plt.legend(loc='lower right')
plt.grid()
# 绘制父图右上角的折线图
ax1 = plt.axes([0.6, 0.6, 0.2, 0.1])
ax1.plot(focus_epochs, focus_map_c2fg2, color='#42d5ff')
ax1.plot(focus_epochs, focus_map_c3g2, color='red')
ax1.plot(focus_epochs, focus_map_v5s, color='blue')
# 显示图形
plt.grid()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件并加载数据
data_c2fg2 = pd.read_csv('runs/train/exp7/results2.csv')
data_c3g2 = pd.read_csv('runs/train/exp8/results2.csv')
data_v5s = pd.read_csv('runs/train/exp9/results2.csv')


# 提取"loss"和"epoch"的数据列
loss_c2fg2 = data_c2fg2['train/box_loss']+data_c2fg2['train/obj_loss']
loss_c3g2 = data_c3g2['train/box_loss']+data_c3g2['train/obj_loss']
loss_v5s = data_v5s['train/box_loss']+data_v5s['train/obj_loss']
vloss_c2fg2 = data_c2fg2['val/box_loss']+data_c2fg2['val/obj_loss']
vloss_c3g2 = data_c3g2['val/box_loss']+data_c3g2['val/obj_loss']
vloss_v5s = data_v5s['val/box_loss']+data_v5s['val/obj_loss']
map_c2fg2 = data_c2fg2['metrics/mAP_0.5']
map_c3g2 = data_c3g2['metrics/mAP_0.5']
map_v5s = data_v5s['metrics/mAP_0.5']
epochs = data_v5s['epoch']

# 绘制折线图
plt.plot(epochs, loss_c2fg2, color = 'green',label='YOLO-GX')
plt.plot(epochs, loss_c3g2, color='red', label='YOLO-Ghost')
plt.plot(epochs, loss_v5s, color = 'blue',label='YOLOV5')
plt.plot(epochs, vloss_c2fg2, color = 'black',label='YOLO-GXv')
plt.plot(epochs, vloss_c3g2, color='#42d5ff', label='YOLO-Ghostv')
plt.plot(epochs, vloss_v5s, color = 'yellow',label='YOLOV5v')
# plt.plot(epochs, map_c2fg2, color = '#42d5ff',label='YOLO-GX')
# plt.plot(epochs, map_c3g2, color='red', label='YOLO-Ghost')
# plt.plot(epochs, map_v5s, color = 'blue',label='YOLOV5')
# 添加标题和标签
plt.title('mAP')
plt.xlabel('Epochs')
plt.ylabel('mAP')
plt.legend()
# 显示图形
plt.show()

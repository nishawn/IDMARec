import numpy as np
import matplotlib.pyplot as plt

# 生成一些示例数据
x = np.arange(10)
y1 = np.random.randint(10, size=10)
y2 = np.random.randint(10, size=10)

# 创建一个图形和两个y轴
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
# 绘制折线图
line1 = ax1.plot(x, y1, label='y1轴', color='royalblue', marker='o', ls='-.')
line2 = ax2.plot(x, y2, label='y2轴', color='tomato', marker=None, ls='--')

# 设置x轴和y轴的标签，指明坐标含义
ax1.set_xlabel('x轴', fontdict={'size': 16})
ax1.set_ylabel('y1轴', fontdict={'size': 16})
ax2.set_ylabel('y2轴', fontdict={'size': 16})
# 添加图表题
plt.title('双y轴折线图')
# 添加图例
plt.legend()
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 展示图片
plt.show()

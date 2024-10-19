import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
# 读取CSV文件
data = pd.read_csv(r'C:\Users\Lenovo\Desktop\code\VMD_Data\has_seasonality\seq_3_sarima.csv')

# 提取预测值和真实值
pred = data['pred']
true = data['true']

# 设置图片尺寸和质量
image_width =4000  # 图片宽度
image_height = 6000  # 图片高度
dpi = 800  # 图片质量（每英寸点数）

# 计算每个图片的时间步范围
num_rows = 3  # 图片行数
num_cols = 3  # 图片列数
num_images = num_rows * num_cols  # 图片数量
time_steps = len(data) // num_images  # 每张图片的时间步数量

# 创建一个大幕布用于合成图片
canvas_width = image_width * num_cols
canvas_height = image_height * num_rows
canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))


# 分别绘制每个图片
for i in range(num_images):
    start_index = i * time_steps
    end_index = start_index + time_steps

    # 提取对应时间步的预测值和真实值
    pred_values = pred[start_index:end_index]
    true_values = true[start_index:end_index]

    # 创建一个新的图像并绘制拟合图像
    fig, ax = plt.subplots(figsize=(image_width / dpi, image_height / dpi), dpi=dpi)
    ax.plot(pred_values, label='predict')
    ax.plot(true_values, label='turth')
    ax.set_xlabel('time')
    ax.set_ylabel('value')
    ax.set_title('Prediction & ground truth')  # 设置英文标题字体
    ax.legend()

    # 保存绘图为临时文件
    temp_filename = f'temp_image_{i}.png'
    plt.savefig(temp_filename, dpi=dpi, bbox_inches='tight')
    plt.close()

    # 将临时文件的图像粘贴到幕布上
    image = Image.open(temp_filename)
    row = i // num_cols
    col = i % num_cols
    canvas.paste(image, (col * image_width, row * image_height))

    # 删除临时文件
    image.close()
    del image

    # 删除临时文件
    os.remove(temp_filename)

# 保存合成后的大幕布图像
canvas.save('combined_image.png', dpi=(dpi, dpi))
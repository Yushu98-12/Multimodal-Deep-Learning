import numpy as np
import pandas as pd


def calculate_statistics(temperatures, rainfall):
    avg_temperature = np.mean(temperatures)
    std_temperature = np.std(temperatures)
    avg_rainfall = np.mean(rainfall)
    std_rainfall = np.std(rainfall)
    return avg_temperature, std_temperature, avg_rainfall, std_rainfall


def calculate_final_result(avg_temperature, std_temperature, avg_rainfall, std_rainfall, temps, rains):
    # 计算干旱程度
    I1 = 0.25 *( ((avg_temp_3-avg_temperature)/std_temperature)-((rain_3-avg_rainfall)/std_rainfall))
    I2 = 0.44 *( ((avg_temp_4-avg_temperature)/std_temperature)-((rain_4-avg_rainfall)/std_rainfall))
    I3 = 0.31 *( ((avg_temp_5-avg_temperature)/std_temperature)-((rain_5-avg_rainfall)/std_rainfall))
    final_result = I1+I2+I3

    return final_result


# 输入数据
year1 = int(input("请输入起始年份："))
year2 = int(input("请输入结束年份："))

data = []
columns = ['年份', '3月平均气温', '4月平均气温', '5月平均气温', '3月平均降雨量', '4月平均降雨量', '5月平均降雨量', '气象干旱程度']

# 输入每年的月份气温和降雨数据
for year in range(year1, year2 + 1):
    avg_temp_3 = float(input(f"请输入{year}年3月的平均气温："))
    avg_temp_4 = float(input(f"请输入{year}年4月的平均气温："))
    avg_temp_5 = float(input(f"请输入{year}年5月的平均气温："))
    rain_3 = float(input(f"请输入{year}年3月的平均降雨量："))
    rain_4 = float(input(f"请输入{year}年4月的平均降雨量："))
    rain_5 = float(input(f"请输入{year}年5月的平均降雨量："))

    avg_temperature, std_temperature, avg_rainfall, std_rainfall = calculate_statistics(
        [avg_temp_3, avg_temp_4, avg_temp_5], [rain_3, rain_4, rain_5])
    final_result = calculate_final_result(avg_temperature, std_temperature, avg_rainfall, std_rainfall,
                                          [avg_temp_3, avg_temp_4, avg_temp_5], [rain_3, rain_4, rain_5])
    data.append([year, avg_temp_3, avg_temp_4, avg_temp_5, rain_3, rain_4, rain_5, final_result])

# 创建DataFrame保存数据
df = pd.DataFrame(data, columns=columns)

# 将DataFrame保存为CSV文件
df.to_csv('data_and_results.csv', index=False)

# 打印表格
print(df)
print("数据和结果已保存为 data_and_results.csv 文件")

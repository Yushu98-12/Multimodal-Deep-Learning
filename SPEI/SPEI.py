import numpy as np
from scipy.stats import expon

def calculate_SPI(precipitation):
    # 计算标准化降水指数 (SPI)
    # precipitation: 降水数据数组

    # 计算降水总量
    total_precipitation = np.sum(precipitation)

    # 计算累计概率
    probabilities = (np.arange(len(precipitation)) + 1) / (len(precipitation) + 1)

    # 用gamma分布拟合降水数据
    params = expon.fit(precipitation)
    cdf = expon.cdf(precipitation, *params)

    # 计算SPI值
    spi = expon.ppf(probabilities, *params)
    spi = (total_precipitation - spi) / total_precipitation  # 标准化为SPI值

    return spi


def calculate_SPEI(precipitation, evapotranspiration, scale):
    # 计算标准化蒸散发指数 (SPEI)
    # precipitation: 降水数据数组
    # evapotranspiration: 蒸散发数据数组
    # scale: 计算SPEI的时间尺度（月度、年度等）

    # 计算SPI值
    spi = calculate_SPI(precipitation)

    # 根据时间尺度计算SPEI
    n = len(spi)
    spei = np.zeros_like(spi)

    for i in range(scale, n):
        spei[i] = spi[i] - calculate_gamma(spi[i-scale:i])

    return spei


def calculate_gamma(spi_values):
    # 计算蒸散发值的gamma函数拟合参数
    # spi_values: SPI值数组

    # 用指数分布拟合SPI值
    params = expon.fit(spi_values)

    # 计算蒸散发值的gamma函数值
    exponential_value = expon.ppf(0.5, *params)

    return exponential_value


# 示例用法
precipitation_data = [100, 150, 120, 80, 60, 50, 40, 30, 20, 10]
evapotranspiration_data = [50, 40, 60, 30, 40, 30, 20, 20, 10, 20]
scale = 1

spei_values = calculate_SPEI(precipitation_data, evapotranspiration_data, scale)
print("当天的SPEI值：", spei_values[-1])

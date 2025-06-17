
"""
学生模板：地壳热扩散数值模拟
文件:earth_crust_diffusion_student.py
重要：函数名称必须与参考答案一致！
"""
import numpy as np
import matplotlib.pyplot as plt


def solve_earth_crust_diffusion(h=1.0, a=1.0, M=21, N=366, years=10):
    """
    实现显式差分法求解地壳热扩散问题
    
    返回:
        tuple: (depth_array, temperature_matrix)
        depth_array: 深度坐标数组 (m)
        temperature_matrix: 温度场矩阵 (°C)
    
    物理背景: 模拟地壳中温度随深度和时间的周期性变化
    数值方法: 显式差分格式
    
    实现步骤:
    1. 设置物理参数和网格参数
    2. 初始化温度场
    3. 应用边界条件
    4. 实现显式差分格式
    5. 返回计算结果
    """
    # TODO: 设置物理参数
    D = 0.1  # 热扩散率 (m^2/day)
    A = 10.0  # 年平均地表温度 (°C)
    B = 12.0  # 地表温度振幅 (°C)
    TAU = 365.0  # 年周期 (days)
    T_BOTTOM = 11.0  # 20米深处温度 (°C)
    T_INITIAL = 10.0  # 初始温度 (°C)
    DEPTH_MAX = 20.0  # 最大深度 (m)
     # 计算稳定性参数
    r = h * D / a**2
    print(f"稳定性参数 r = {r:.4f}")
    # TODO: 初始化数组
    T = np.zeros((M, N)) + T_INITIAL#温度矩阵
    T[-1, :] = T_BOTTOM  # 底部边界条件
   # 时间步进循环
    for year in range(years):
        for j in range(1, N-1):
            # 地表边界条件
            T[0, j] = A + B * np.sin(2 * np.pi * j / TAU)
            
            # 显式差分格式
            T[1:-1, j+1] = T[1:-1, j] + r * (T[2:, j] + T[:-2, j] - 2*T[1:-1, j])
     # 创建深度数组
    depth = np.arange(0, DEPTH_MAX + h, h)
    return depth, T
def plot_seasonal_profiles(depth, temperature, seasons=[90, 180, 270, 365]):
    """
    绘制季节性温度轮廓
    
    参数:
        depth (ndarray): 深度数组
        temperature (ndarray): 温度矩阵
        seasons (list): 季节时间点 (days)
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制各季节的温度轮廓
    for i, day in enumerate(seasons):
        plt.plot(depth, temperature[:, day], 
                label=f'Day {day}', linewidth=2)
    plt.xlabel('Depth (m)')
    plt.ylabel('Temperature (°C)')
    plt.title('Seasonal Temperature Profiles')
    plt.grid(True)
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    # 测试代码
   # 运行模拟
    depth, T = solve_earth_crust_diffusion()
    
    # 绘制季节性温度轮廓
    plot_seasonal_profiles(depth, T)

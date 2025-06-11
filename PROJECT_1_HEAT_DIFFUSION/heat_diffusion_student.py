import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 物理参数
K = 237       # 热导率 (W/m/K)
C = 900       # 比热容 (J/kg/K)
rho = 2700    # 密度 (kg/m^3)
D = K/(C*rho) # 热扩散系数
L = 1         # 铝棒长度 (m)
dx = 0.01     # 空间步长 (m)
dt = 0.5      # 时间步长 (s)
Nx = int(L/dx) + 1 # 空间格点数
Nt = 2000     # 时间步数

def basic_heat_diffusion():
    """
    任务1: 基本热传导模拟
    
    返回:
        np.ndarray: 温度分布数组
    """
    r = D * dt / (dx ** 2)
    print(f"任务1 - 稳定性参数 r = {r:.4f}")
    
    # 初始化温度数组
    u = np.zeros((Nx, Nt))
    u[:, 0] = 100  # 初始温度100K
    u[0, :] = 0    # 左边界条件
    u[-1, :] = 0   # 右边界条件
    
    # 显式有限差分法迭代
    for j in range(Nt - 1):
        u[1:-1, j + 1] = (1 - 2 * r) * u[1:-1, j] + r * (u[2:, j] + u[:-2, j])
    
    return u

def analytical_solution(n_terms=100):
    """
    任务2: 解析解函数
    
    参数:
        n_terms (int): 傅里叶级数项数
    
    返回:
        np.ndarray: 解析解温度分布
    """
    x = np.linspace(0, L, Nx)
    t = np.linspace(0, dt * Nt, Nt)
    X, T = np.meshgrid(x, t)
    s = np.zeros_like(X.T)
    
    # 计算傅里叶级数解
    for i in range(n_terms):
        n = 2 * i + 1  # 只取奇数项
        s += (400 / (n * np.pi)) * np.sin(n * np.pi * X.T / L) * np.exp(-(n * np.pi / L)**2 * D * T.T)
    
    return s

def stability_analysis():
    """
    任务3: 数值解稳定性分析
    """
    # 使用不稳定的参数
    unstable_dt = 0.6  # 使r>0.5
    r = D * unstable_dt / (dx ** 2)
    print(f"任务3 - 稳定性参数 r = {r:.4f} (r>0.5)")
    
    Nx = int(L / dx) + 1
    Nt = 2000
    
    # 初始化温度数组
    u = np.zeros((Nx, Nt))
    u[:, 0] = 100
    u[0, :] = 0
    u[-1, :] = 0
    
    # 显式有限差分法迭代
    for j in range(Nt - 1):
        u[1:-1, j + 1] = (1 - 2 * r) * u[1:-1, j] + r * (u[2:, j] + u[:-2, j])
    
    # 可视化不稳定解
    plot_3d_solution(u, dx, unstable_dt, Nt, title='Task 3: Unstable Solution (r>0.5)')

def different_initial_condition():
    """
    任务4: 不同初始条件模拟
    
    返回:
        np.ndarray: 温度分布数组
    """
    r = D * dt / (dx ** 2)
    print(f"任务4 - 稳定性参数 r = {r:.4f}")
    
    Nx = int(L / dx) + 1
    Nt = 1000
    
    # 初始化温度数组 - 不同初始条件
    u = np.zeros((Nx, Nt))
    u[:int(Nx/2), 0] = 100  # 左半部分初始温度100K
    u[int(Nx/2):, 0] = 50   # 右半部分初始温度50K
    u[0, :] = 0
    u[-1, :] = 0
    
    # 显式有限差分法迭代
    for j in range(Nt - 1):
        u[1:-1, j + 1] = (1 - 2 * r) * u[1:-1, j] + r * (u[2:, j] + u[:-2, j])
    
    # 可视化
    plot_3d_solution(u, dx, dt, Nt, title='Task 4: Temperature Evolution with Different Initial Conditions')
    return u

def heat_diffusion_with_cooling():
    """
    任务5: 包含牛顿冷却定律的热传导
    """
    r = D * dt / (dx ** 2)
    h = 0.1  # 冷却系数
    print(f"任务5 - 稳定性参数 r = {r:.4f}, 冷却系数 h = {h}")
    
    Nx = int(L / dx) + 1
    Nt = 1000
    
    # 初始化温度数组
    u = np.zeros((Nx, Nt))
    u[:, 0] = 100
    u[0, :] = 0
    u[-1, :] = 0
    
    # 显式有限差分法迭代 - 包含冷却项
    for j in range(Nt - 1):
        u[1:-1, j + 1] = (1 - 2 * r - h * dt) * u[1:-1, j] + r * (u[2:, j] + u[:-2, j])
    
    # 可视化
    plot_3d_solution(u, dx, dt, Nt, title='Task 5: Heat Diffusion with Newton Cooling')

def plot_3d_solution(u, dx, dt, Nt, title):
    """
    绘制3D温度分布图
    
    参数:
        u (np.ndarray): 温度分布数组
        dx (float): 空间步长
        dt (float): 时间步长
        Nt (int): 时间步数
        title (str): 图表标题
    
    返回:
        None
    """
    Nx = u.shape[0]
    x = np.linspace(0, dx * (Nx - 1), Nx)
    t = np.linspace(0, dt * Nt, Nt)
    X, T = np.meshgrid(x, t)
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, u.T, cmap='rainbow', linewidth=0, antialiased=False)
    ax.set_xlabel('Position x (m)')
    ax.set_ylabel('Time t (s)')
    ax.set_zlabel('Temperature T (K)')
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

if __name__ == "__main__":
    """
    主函数 - 演示和测试各任务功能
    
    执行顺序:
    1. 基本热传导模拟
    2. 解析解计算
    3. 数值解稳定性分析
    4. 不同初始条件模拟
    5. 包含冷却效应的热传导
    """
    print("=== 铝棒热传导问题学生实现 ===")
    
    print("\n1. 基本热传导模拟")
    u = basic_heat_diffusion()
    plot_3d_solution(u, dx, dt, Nt, title='Task 1: Basic Heat Diffusion Solution')
    
    print("\n2. 解析解")
    s = analytical_solution()
    plot_3d_solution(s, dx, dt, Nt, title='Analytical Solution')
    
    print("\n3. 数值解稳定性分析")
    stability_analysis()
    
    print("\n4. 不同初始条件模拟")
    different_initial_condition()
    
    print("\n5. 包含冷却效应的热传导")
    heat_diffusion_with_cooling()

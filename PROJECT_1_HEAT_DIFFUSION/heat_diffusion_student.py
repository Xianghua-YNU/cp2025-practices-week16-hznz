import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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
    # 计算稳定性参数 r
    r = D * dt / (dx**2)
    print(f"任务1: r = {r:.4f} (稳定性条件要求 r <= 0.5)")
    
    # 初始化温度数组
    u = np.zeros((Nx, Nt))
    
    # 设置初始条件: 除边界外整个铝棒初始温度为100K
    u[1:-1, 0] = 100.0
    
    # 显式有限差分法求解
    for j in range(0, Nt-1):
        # 应用边界条件
        u[0, j] = 0.0    # x=0处边界
        u[-1, j] = 0.0   # x=L处边界
        
        # 内部点更新
        for i in range(1, Nx-1):
            u[i, j+1] = u[i, j] + r * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
    
    return u

def analytical_solution(n_terms=100):
    """
    任务2: 解析解函数
    
    参数:
        n_terms (int): 傅里叶级数项数
    
    返回:
        np.ndarray: 解析解温度分布
    """
    # 创建空间和时间网格
    x = np.linspace(0, L, Nx)
    t = np.arange(0, Nt*dt, dt)
    
    # 初始化温度数组
    u_analytical = np.zeros((Nx, Nt))
    
    # 计算解析解 (只取奇数项)
    for j, time in enumerate(t):
        for i, pos in enumerate(x):
            T_sum = 0.0
            for n in range(1, 2*n_terms, 2):  # n = 1, 3, 5, ..., 2*n_terms-1
                kn = n * np.pi / L
                T_sum += (4 * 100) / (n * np.pi) * np.sin(kn * pos) * np.exp(-kn**2 * D * time)
            u_analytical[i, j] = T_sum
    
    return u_analytical

def stability_analysis():
    """
    任务3: 数值解稳定性分析
    """
    # 使用不稳定的时间步长 (r>0.5)
    unstable_dt = 0.6
    unstable_r = D * unstable_dt / (dx**2)
    print(f"任务3: 使用不稳定的时间步长 dt={unstable_dt}s, r={unstable_r:.4f} (>0.5)")
    
    # 初始化温度数组
    u_unstable = np.zeros((Nx, Nt))
    
    # 设置初始条件
    u_unstable[1:-1, 0] = 100.0
    
    # 使用显式有限差分法求解 (不稳定参数)
    for j in range(0, Nt-1):
        # 边界条件
        u_unstable[0, j] = 0.0
        u_unstable[-1, j] = 0.0
        
        # 内部点更新
        for i in range(1, Nx-1):
            u_unstable[i, j+1] = u_unstable[i, j] + unstable_r * (u_unstable[i+1, j] - 2*u_unstable[i, j] + u_unstable[i-1, j])
    
    # 绘制不稳定解
    plot_3d_solution(u_unstable, dx, unstable_dt, Nt, "数值解稳定性分析 (r > 0.5)")
    
    # 返回结果用于进一步分析
    return u_unstable

def different_initial_condition():
    """
    任务4: 不同初始条件模拟
    
    返回:
        np.ndarray: 温度分布数组
    """
    # 使用稳定的时间步长
    r = D * dt / (dx**2)
    
    # 初始化温度数组
    u = np.zeros((Nx, Nt))
    
    # 设置初始条件: 左边0-0.5m为100K，右边0.5-1m为50K
    x = np.linspace(0, L, Nx)
    for i in range(Nx):
        if x[i] < 0.5:
            u[i, 0] = 100.0
        else:
            u[i, 0] = 50.0
    
    # 显式有限差分法求解
    for j in range(0, Nt-1):
        # 边界条件
        u[0, j] = 0.0
        u[-1, j] = 0.0
        
        # 内部点更新
        for i in range(1, Nx-1):
            u[i, j+1] = u[i, j] + r * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
    
    return u

def heat_diffusion_with_cooling(h=0.01):
    """
    任务5: 包含牛顿冷却定律的热传导
    
    返回:
        np.ndarray: 温度分布数组
    """
    # 使用稳定的时间步长
    r = D * dt / (dx**2)
    print(f"任务5: 牛顿冷却定律 h={h} s⁻¹, r={r:.4f}")
    
    # 初始化温度数组
    u = np.zeros((Nx, Nt))
    
    # 设置初始条件
    u[1:-1, 0] = 100.0
    
    # 显式有限差分法求解 (包含牛顿冷却项)
    for j in range(0, Nt-1):
        # 边界条件
        u[0, j] = 0.0
        u[-1, j] = 0.0
        
        # 内部点更新 (包含冷却项)
        for i in range(1, Nx-1):
            u[i, j+1] = (1 - 2*r - h*dt) * u[i, j] + r * (u[i+1, j] + u[i-1, j])
    
    return u

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
    # 创建空间和时间网格
    x = np.arange(0, L+dx, dx)
    t = np.arange(0, Nt*dt, dt)
    X, T = np.meshgrid(t, x)
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制表面图
    surf = ax.plot_surface(X, T, u, cmap='viridis', rstride=5, cstride=50, 
                          linewidth=0, antialiased=False)
    
    # 设置坐标轴标签
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('位置 (m)')
    ax.set_zlabel('温度 (K)')
    ax.set_title(title)
    
    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # 调整视角
    ax.view_init(elev=30, azim=-120)
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

if __name__ == "__main__":
    """
    主函数 - 演示和测试各任务功能
    """
    print("=== 铝棒热传导问题 ===")
    print(f"热扩散系数 D = {D:.6f} m²/s")
    
    # 任务1: 基本热传导模拟
    print("\n执行任务1: 基本热传导模拟...")
    u_basic = basic_heat_diffusion()
    plot_3d_solution(u_basic, dx, dt, Nt, "基本热传导模拟")
    
    # 任务2: 解析解
    print("\n执行任务2: 解析解计算...")
    u_analytical = analytical_solution(n_terms=50)
    plot_3d_solution(u_analytical, dx, dt, Nt, "热传导解析解")
    
    # 任务3: 稳定性分析
    print("\n执行任务3: 数值解稳定性分析...")
    u_unstable = stability_analysis()
    
    # 任务4: 不同初始条件
    print("\n执行任务4: 不同初始条件模拟...")
    u_diff_ic = different_initial_condition()
    plot_3d_solution(u_diff_ic, dx, dt, Nt, "不同初始条件模拟")
    
    # 任务5: 牛顿冷却定律
    print("\n执行任务5: 包含牛顿冷却定律的热传导...")
    u_cooling = heat_diffusion_with_cooling(h=0.01)
    plot_3d_solution(u_cooling, dx, dt, Nt, "牛顿冷却定律热传导")
    
    print("\n所有任务完成!")

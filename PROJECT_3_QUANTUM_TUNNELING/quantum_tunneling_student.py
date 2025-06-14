"""学生模板：量子隧穿效应
文件：quantum_tunneling_student.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class QuantumTunnelingSolver:
    """量子隧穿求解器类
    
    该类实现了一维含时薛定谔方程的数值求解，用于模拟量子粒子的隧穿效应。
    使用变形的Crank-Nicolson方法进行时间演化，确保数值稳定性和概率守恒。
    """
    
    def __init__(self, Nx=220, Nt=300, x0=40, k0=0.5, d=10, barrier_width=3, barrier_height=1.0):
        """初始化量子隧穿求解器
        
        参数:
            Nx (int): 空间网格点数，默认220
            Nt (int): 时间步数，默认300
            x0 (float): 初始波包中心位置，默认40
            k0 (float): 初始波包动量(波数)，默认0.5
            d (float): 初始波包宽度参数，默认10
            barrier_width (int): 势垒宽度，默认3
            barrier_height (float): 势垒高度，默认1.0
        """
        # 初始化参数
        self.Nx = Nx
        self.Nt = Nt
        self.x0 = x0
        self.k0 = k0
        self.d = d
        self.barrier_width = int(barrier_width)
        self.barrier_height = barrier_height
        
        # 创建空间网格 (0 到 Nx-1)
        self.x = np.arange(Nx)
        
        # 设置势垒
        self.V = self.setup_potential()
        
        # 初始化波函数矩阵和辅助矩阵
        self.B = np.zeros((Nx, Nt), dtype=complex)  # 波函数矩阵
        self.C = np.zeros((Nx, Nt), dtype=complex)  # 辅助变量矩阵

    def wavefun(self, x):
        """高斯波包函数
        
        参数:
            x (np.ndarray): 空间坐标数组
            
        返回:
            np.ndarray: 初始波函数值
            
        数学公式:
            ψ(x,0) = exp(ik₀x) * exp(-(x-x₀)²ln10(2)/d²)
        """
        # 计算高斯包络
        gaussian = np.exp(-(x - self.x0)**2 * (2 * np.log(10)) / self.d**2)
        # 添加动量项
        return np.exp(1j * self.k0 * x) * gaussian

    def setup_potential(self):
        """设置势垒函数
        
        返回:
            np.ndarray: 势垒数组
            
        说明:
            在空间网格中间位置创建矩形势垒
            势垒位置：从 Nx//2 到 Nx//2+barrier_width
            势垒高度：barrier_height
        """
        V = np.zeros(self.Nx)
        barrier_start = self.Nx // 2
        barrier_end = barrier_start + self.barrier_width
        
        # 确保势垒不超过边界
        if barrier_end > self.Nx:
            barrier_end = self.Nx
        
        V[barrier_start:barrier_end] = self.barrier_height
        return V

    def build_coefficient_matrix(self):
        """构建变形的Crank-Nicolson格式的系数矩阵
        
        返回:
            np.ndarray: 系数矩阵A
            
        数学原理:
            对于dt=1, dx=1的情况，哈密顿矩阵的对角元素为: -2+2j-V
            非对角元素为1（表示动能项的有限差分）
            
        矩阵结构:
            三对角矩阵，主对角线为 -2+2j-V[i]，上下对角线为1
        """
        # 主对角线元素
        main_diag = -2 + 2j - self.V
        
        # 上下对角线元素（全1）
        off_diag = np.ones(self.Nx - 1)
        
        # 构建三对角矩阵
        A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        return A

    def solve_schrodinger(self):
        """求解一维含时薛定谔方程
        
        使用Crank-Nicolson方法进行时间演化
        
        返回:
            tuple: (x, V, B, C) - 空间网格, 势垒, 波函数矩阵, chi矩阵
            
        数值方法:
            Crank-Nicolson隐式格式，具有二阶精度和无条件稳定性
            时间演化公式：C[:,t+1] = 4j * solve(A, B[:,t])
                         B[:,t+1] = C[:,t+1] - B[:,t]
        """
        # 构建系数矩阵
        A = self.build_coefficient_matrix()
        
        # 设置初始波函数
        self.B[:, 0] = self.wavefun(self.x)
        
        # 归一化初始波函数
        norm = np.sqrt(np.sum(np.abs(self.B[:, 0])**2))
        self.B[:, 0] /= norm
        
        # 时间演化循环
        for t in range(self.Nt - 1):
            # 计算右侧向量
            rhs = 4j * self.B[:, t]
            
            # 求解线性方程组 A * C[:, t] = rhs
            self.C[:, t] = np.linalg.solve(A, rhs)
            
            # 更新波函数
            self.B[:, t + 1] = self.C[:, t] - self.B[:, t]
        
        return self.x, self.V, self.B, self.C

    def calculate_coefficients(self):
        """计算透射和反射系数
        
        返回:
            tuple: (T, R) - 透射系数和反射系数
            
        物理意义:
            透射系数T：粒子穿过势垒的概率
            反射系数R：粒子被势垒反射的概率
            应满足：T + R ≈ 1（概率守恒）
            
        计算方法:
            T = ∫|ψ(x>barrier)|²dx / ∫|ψ(x)|²dx
            R = ∫|ψ(x<barrier)|²dx / ∫|ψ(x)|²dx
        """
        # 确定势垒位置
        barrier_start = self.Nx // 2
        barrier_end = barrier_start + self.barrier_width
        
        # 获取最终时刻的波函数
        psi_final = self.B[:, -1]
        
        # 计算总概率
        total_prob = np.sum(np.abs(psi_final)**2)
        
        # 计算反射区域概率 (势垒左侧)
        reflection_prob = np.sum(np.abs(psi_final[:barrier_start])**2)
        
        # 计算透射区域概率 (势垒右侧)
        transmission_prob = np.sum(np.abs(psi_final[barrier_end:])**2)
        
        # 计算系数
        T = transmission_prob / total_prob
        R = reflection_prob / total_prob
        
        return T, R

    def plot_evolution(self, time_indices=None):
        """绘制波函数演化图
        
        参数:
            time_indices (list): 要绘制的时间索引列表，默认为[0, Nt//4, Nt//2, 3*Nt//4, Nt-1]
            
        功能:
            在多个子图中显示不同时刻的波函数概率密度和势垒
        """
        if time_indices is None:
            time_indices = [0, self.Nt//4, self.Nt//2, 3*self.Nt//4, self.Nt-1]
        
        num_plots = len(time_indices)
        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 8))
        fig.suptitle('量子隧穿效应 - 波函数演化', fontsize=16)
        
        # 归一化势垒以便显示
        V_normalized = self.V / np.max(self.V) * np.max(np.abs(self.B)**2)
        
        for i, t_idx in enumerate(time_indices):
            ax = axs[i] if num_plots > 1 else axs
            prob_density = np.abs(self.B[:, t_idx])**2
            
            # 绘制概率密度
            ax.plot(self.x, prob_density, 'b-', label=f'|ψ|² (t={t_idx})')
            
            # 绘制势垒
            ax.plot(self.x, V_normalized, 'r-', label='势垒')
            
            ax.set_ylabel('概率密度')
            ax.legend(loc='upper right')
            
            # 只在最后一个子图添加x轴标签
            if i == num_plots - 1 or num_plots == 1:
                ax.set_xlabel('位置 x')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

    def create_animation(self, interval=20):
        """创建波包演化动画
        
        参数:
            interval (int): 动画帧间隔(毫秒)，默认20
            
        返回:
            matplotlib.animation.FuncAnimation: 动画对象
            
        功能:
            实时显示波包在势垒附近的演化过程
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 归一化势垒以便显示
        V_normalized = self.V / np.max(self.V) * np.max(np.abs(self.B)**2)
        
        # 绘制势垒
        ax.plot(self.x, V_normalized, 'r-', label='势垒')
        
        # 初始化概率密度线
        line, = ax.plot(self.x, np.abs(self.B[:, 0])**2, 'b-', label='|ψ|²')
        
        ax.set_xlim(0, self.Nx)
        ax.set_ylim(0, np.max(np.abs(self.B)**2) * 1.1)
        ax.set_xlabel('位置 x')
        ax.set_ylabel('概率密度')
        ax.set_title('量子隧穿效应 - 动态演化')
        ax.legend(loc='upper right')
        ax.grid(True)
        
        # 动画更新函数
        def update(frame):
            prob_density = np.abs(self.B[:, frame])**2
            line.set_ydata(prob_density)
            ax.set_title(f'量子隧穿效应 - 时间步: {frame}/{self.Nt-1}')
            return line,
        
        # 创建动画
        ani = animation.FuncAnimation(
            fig, update, frames=self.Nt, interval=interval, blit=True
        )
        
        plt.close(fig)  # 防止在notebook中显示静态图
        return ani

    def verify_probability_conservation(self):
        """验证概率守恒
        
        返回:
            np.ndarray: 每个时间步的总概率
            
        物理原理:
            量子力学中概率必须守恒：∫|ψ(x,t)|²dx = 常数
            数值计算中应该保持在1附近
        """
        probabilities = np.zeros(self.Nt)
        
        for t in range(self.Nt):
            probabilities[t] = np.sum(np.abs(self.B[:, t])**2)
        
        # 绘制概率守恒图
        plt.figure(figsize=(10, 5))
        plt.plot(range(self.Nt), probabilities, 'b-')
        plt.axhline(1.0, color='r', linestyle='--')
        plt.xlabel('时间步')
        plt.ylabel('总概率')
        plt.title('概率守恒验证')
        plt.grid(True)
        plt.show()
        
        return probabilities

    def demonstrate(self):
        """演示量子隧穿效应
        
        功能:
            1. 求解薛定谔方程
            2. 计算并显示透射和反射系数
            3. 绘制波函数演化图
            4. 验证概率守恒
            5. 创建并显示动画
            
        返回:
            animation对象
        """
        print("开始求解薛定谔方程...")
        self.solve_schrodinger()
        print("求解完成!")
        
        # 计算透射和反射系数
        T, R = self.calculate_coefficients()
        print(f"透射系数 T = {T:.4f}")
        print(f"反射系数 R = {R:.4f}")
        print(f"T + R = {T + R:.4f}")
        
        # 绘制演化图
        print("绘制波函数演化图...")
        self.plot_evolution()
        
        # 验证概率守恒
        print("验证概率守恒...")
        self.verify_probability_conservation()
        
        # 创建动画
        print("创建动画...")
        ani = self.create_animation()
        
        return ani


def demonstrate_quantum_tunneling():
    """便捷的演示函数
    
    创建默认参数的求解器并运行演示
    
    返回:
        animation对象
    """
    solver = QuantumTunnelingSolver()
    return solver.demonstrate()


if __name__ == "__main__":
    # 运行演示
    barrier_width = 3
    barrier_height = 1.0
    solver = QuantumTunnelingSolver(barrier_width=barrier_width, barrier_height=barrier_height)
    animation = solver.demonstrate()

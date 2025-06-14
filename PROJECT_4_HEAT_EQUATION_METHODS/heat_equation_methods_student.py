
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import scipy.linalg
import time

class HeatEquationSolver:
    """
    热传导方程求解器，实现四种不同的数值方法。
    
    求解一维热传导方程：du/dt = alpha * d²u/dx²
    边界条件：u(0,t) = 0, u(L,t) = 0
    初始条件：u(x,0) = phi(x)
    """
    
    def __init__(self, L=20.0, alpha=10.0, nx=21, T_final=25.0):
        """
        初始化热传导方程求解器。
        
        参数:
            L (float): 空间域长度 [0, L]
            alpha (float): 热扩散系数
            nx (int): 空间网格点数
            T_final (float): 最终模拟时间
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T_final = T_final
        
        # 空间网格
        self.x = np.linspace(0, L, nx)
        self.dx = L / (nx - 1)
        
        # 初始化解数组
        self.u_initial = self._set_initial_condition()
        
    def _set_initial_condition(self):
        """
        设置初始条件：u(x,0) = 1 当 10 <= x <= 11，否则为 0。
        
        返回:
            np.ndarray: 初始温度分布
        """
        u = np.zeros(self.nx)
        mask = (self.x >= 10) & (self.x <= 11)
        u[mask] = 1.0
        u[0] = 0.0  # 边界条件
        u[-1] = 0.0  # 边界条件
        return u
    
    def solve_explicit(self, dt=0.01, plot_times=None):
        """
        使用显式有限差分法（FTCS）求解。
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
        
        # 计算稳定性参数
        r = self.alpha * dt / (self.dx**2)
        
        # 检查稳定性条件
        if r > 0.5:
            print(f"警告：显式方法稳定性条件不满足 (r = {r:.4f} > 0.5)，结果可能不稳定!")
        
        # 初始化变量
        u = self.u_initial.copy()
        t = 0.0
        solutions = []
        times = []
        
        # 存储初始解
        if 0 in plot_times:
            solutions.append(u.copy())
            times.append(t)
        
        # 时间步进循环
        num_steps = int(self.T_final / dt) + 1
        for step in range(1, num_steps):
            t = step * dt
            
            # 计算空间二阶导数
            laplace_u = laplace(u, mode='nearest') / (self.dx**2)
            
            # 更新解
            u = u + self.alpha * dt * laplace_u
            
            # 应用边界条件
            u[0] = 0.0
            u[-1] = 0.0
            
            # 在指定时间点存储解
            if any(np.abs(t - pt) < dt/2 for pt in plot_times if pt > 0):
                solutions.append(u.copy())
                times.append(t)
        
        return {
            'times': times,
            'solutions': solutions,
            'method': 'explicit',
            'computation_time': time.time(),
            'stability_parameter': r
        }
    
    def solve_implicit(self, dt=0.1, plot_times=None):
        """
        使用隐式有限差分法（BTCS）求解。
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
        
        # 计算扩散数
        r = self.alpha * dt / (self.dx**2)
        n = self.nx - 2  # 内部节点数
        
        # 构建三对角矩阵
        diag = (1 + 2*r) * np.ones(n)
        off_diag = -r * np.ones(n-1)
        A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        
        # 初始化变量
        u = self.u_initial.copy()
        t = 0.0
        solutions = []
        times = []
        
        # 存储初始解
        if 0 in plot_times:
            solutions.append(u.copy())
            times.append(t)
        
        # 时间步进循环
        num_steps = int(self.T_final / dt) + 1
        for step in range(1, num_steps):
            t = step * dt
            
            # 构建右端项（内部节点）
            b = u[1:-1].copy()
            
            # 求解线性系统
            u_internal = scipy.linalg.solve(A, b)
            
            # 更新解
            u[1:-1] = u_internal
            
            # 在指定时间点存储解
            if any(np.abs(t - pt) < dt/2 for pt in plot_times if pt > 0):
                solutions.append(u.copy())
                times.append(t)
        
        return {
            'times': times,
            'solutions': solutions,
            'method': 'implicit',
            'computation_time': time.time(),
            'stability_parameter': r
        }
    
    def solve_crank_nicolson(self, dt=0.5, plot_times=None):
        """
        使用Crank-Nicolson方法求解。
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
        
        # 计算扩散数
        r = self.alpha * dt / (self.dx**2)
        n = self.nx - 2  # 内部节点数
        
        # 构建左端矩阵 A
        diag = (1 + r) * np.ones(n)
        off_diag = (-r/2) * np.ones(n-1)
        A = np.diag(diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        
        # 初始化变量
        u = self.u_initial.copy()
        t = 0.0
        solutions = []
        times = []
        
        # 存储初始解
        if 0 in plot_times:
            solutions.append(u.copy())
            times.append(t)
        
        # 时间步进循环
        num_steps = int(self.T_final / dt) + 1
        for step in range(1, num_steps):
            t = step * dt
            
            # 构建右端向量
            rhs = np.zeros(n)
            rhs[0] = (r/2)*u[0] + (1 - r)*u[1] + (r/2)*u[2]
            for i in range(1, n-1):
                rhs[i] = (r/2)*u[i] + (1 - r)*u[i+1] + (r/2)*u[i+2]
            rhs[-1] = (r/2)*u[-3] + (1 - r)*u[-2] + (r/2)*u[-1]
            
            # 求解线性系统
            u_internal = scipy.linalg.solve(A, rhs)
            
            # 更新解
            u[1:-1] = u_internal
            
            # 在指定时间点存储解
            if any(np.abs(t - pt) < dt/2 for pt in plot_times if pt > 0):
                solutions.append(u.copy())
                times.append(t)
        
        return {
            'times': times,
            'solutions': solutions,
            'method': 'crank_nicolson',
            'computation_time': time.time(),
            'stability_parameter': r
        }
    
    def _heat_equation_ode(self, t, u_internal):
        """
        用于solve_ivp方法的ODE系统。
        """
        # 重构完整解向量（包含边界条件）
        u_full = np.zeros(self.nx)
        u_full[1:-1] = u_internal
        
        # 使用laplace计算二阶导数
        laplace_u = laplace(u_full, mode='nearest') / (self.dx**2)
        
        # 返回内部节点的时间导数
        return self.alpha * laplace_u[1:-1]
    
    def solve_with_solve_ivp(self, method='BDF', plot_times=None):
        """
        使用scipy.integrate.solve_ivp求解。
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
        
        # 提取内部节点初始条件
        u0_internal = self.u_initial[1:-1].copy()
        
        # 调用solve_ivp求解
        sol = solve_ivp(
            fun=self._heat_equation_ode,
            t_span=(0, self.T_final),
            y0=u0_internal,
            method=method,
            t_eval=plot_times
        )
        
        # 重构包含边界条件的完整解
        solutions = []
        for i in range(sol.y.shape[1]):
            u_full = np.zeros(self.nx)
            u_full[1:-1] = sol.y[:, i]
            solutions.append(u_full)
        
        return {
            'times': sol.t,
            'solutions': solutions,
            'method': 'solve_ivp',
            'computation_time': time.time()
        }
    
    def compare_methods(self, dt_explicit=0.01, dt_implicit=0.1, dt_cn=0.5, 
                       ivp_method='BDF', plot_times=None):
        """
        比较所有四种数值方法。
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
        
        print(f"\n{'='*50}")
        print("开始热传导方程数值解法比较")
        print(f"空间网格点数: {self.nx}, 最终时间: {self.T_final}")
        print(f"显式方法时间步长: {dt_explicit:.4f}")
        print(f"隐式方法时间步长: {dt_implicit:.4f}")
        print(f"Crank-Nicolson方法时间步长: {dt_cn:.4f}")
        print(f"solve_ivp方法: {ivp_method}")
        print(f"比较时间点: {plot_times}")
        print('='*50)
        
        # 调用四种求解方法
        start_time = time.time()
        explicit = self.solve_explicit(dt=dt_explicit, plot_times=plot_times)
        explicit_time = time.time() - start_time
        
        start_time = time.time()
        implicit = self.solve_implicit(dt=dt_implicit, plot_times=plot_times)
        implicit_time = time.time() - start_time
        
        start_time = time.time()
        crank_nicolson = self.solve_crank_nicolson(dt=dt_cn, plot_times=plot_times)
        cn_time = time.time() - start_time
        
        start_time = time.time()
        ivp = self.solve_with_solve_ivp(method=ivp_method, plot_times=plot_times)
        ivp_time = time.time() - start_time
        
        # 更新计算时间
        explicit['computation_time'] = explicit_time
        implicit['computation_time'] = implicit_time
        crank_nicolson['computation_time'] = cn_time
        ivp['computation_time'] = ivp_time
        
        # 打印计算时间
        print("\n计算时间比较:")
        print(f"显式方法: {explicit_time:.6f} 秒")
        print(f"隐式方法: {implicit_time:.6f} 秒")
        print(f"Crank-Nicolson: {cn_time:.6f} 秒")
        print(f"solve_ivp: {ivp_time:.6f} 秒")
        
        return {
            'explicit': explicit,
            'implicit': implicit,
            'crank_nicolson': crank_nicolson,
            'solve_ivp': ivp
        }
    
    def plot_comparison(self, methods_results, save_figure=False, filename='heat_equation_comparison.png'):
        """
        绘制所有方法的比较图。
        """
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('热传导方程不同数值解法比较', fontsize=16)
        
        # 为每种方法创建子图
        for i, (method, results) in enumerate(methods_results.items()):
            ax = axs[i//2, i%2]
            times = results['times']
            solutions = results['solutions']
            
            for j, t in enumerate(times):
                ax.plot(self.x, solutions[j], label=f't={t:.1f}s')
            
            ax.set_title(f'{method}方法')
            ax.set_xlabel('位置 x')
            ax.set_ylabel('温度 u(x,t)')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_figure:
            plt.savefig(filename, dpi=300)
            print(f"图像已保存为: {filename}")
        plt.show()
    
    def analyze_accuracy(self, methods_results, reference_method='solve_ivp'):
        """
        分析不同方法的精度。
        """
        if reference_method not in methods_results:
            print(f"错误：参考方法 '{reference_method}' 不存在！")
            return None
        
        # 获取参考解
        ref_results = methods_results[reference_method]
        ref_solutions = ref_results['solutions']
        ref_times = ref_results['times']
        
        accuracy = {}
        
        for method, results in methods_results.items():
            if method == reference_method:
                continue
                
            solutions = results['solutions']
            times = results['times']
            
            # 确保比较相同时间点
            common_times = set(ref_times) & set(times)
            if not common_times:
                print(f"警告：{method} 和 {reference_method} 没有共同时间点")
                continue
                
            max_errors = []
            avg_errors = []
            
            for t in common_times:
                # 找到参考解索引
                ref_idx = ref_times.index(t)
                ref_sol = ref_solutions[ref_idx]
                
                # 找到当前方法解索引
                sol_idx = times.index(t)
                sol = solutions[sol_idx]
                
                # 计算相对误差
                rel_error = np.abs((sol - ref_sol) / (ref_sol + 1e-10))
                max_errors.append(np.max(rel_error))
                avg_errors.append(np.mean(rel_error))
            
            accuracy[method] = {
                'max_rel_error': np.max(max_errors),
                'avg_rel_error': np.mean(avg_errors)
            }
        
        # 打印精度分析结果
        print("\n精度分析 (以solve_ivp为参考):")
        print("{:<20} {:<15} {:<15}".format("方法", "最大相对误差", "平均相对误差"))
        print("-"*50)
        for method, errors in accuracy.items():
            print("{:<20} {:<15.6f} {:<15.6f}".format(
                method, 
                errors['max_rel_error'], 
                errors['avg_rel_error'])
            )
        
        return accuracy


def main():
    """
    HeatEquationSolver类的演示。
    """
    # 创建求解器实例
    solver = HeatEquationSolver(L=20.0, alpha=10.0, nx=101, T_final=25.0)
    
    # 比较所有方法
    results = solver.compare_methods(
        dt_explicit=0.001,
        dt_implicit=0.1,
        dt_cn=0.5,
        ivp_method='BDF',
        plot_times=[0, 1, 5, 15, 25]
    )
    
    # 绘制比较图
    solver.plot_comparison(results, save_figure=True)
    
    # 分析精度
    accuracy = solver.analyze_accuracy(results)
    
    return solver, results, accuracy


if __name__ == "__main__":
    solver, results, accuracy = main()

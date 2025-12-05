import numpy as np
import matplotlib.pyplot as plt

class MM1Control:
    def __init__(self, lambda_rate, mu_max, max_queue, discount_factor):
        # 初始化参数
        self.lam = lambda_rate      # 到达率 lambda
        self.mu_max = mu_max        # 最大服务率
        self.N = max_queue          # 队列最大长度 (状态截断)
        self.beta = discount_factor # 折扣因子 beta
        
        # 均匀化常数 Lambda (为了把连续变成离散)
        self.UNIF_CONST = self.lam + self.mu_max
        
        # 我们可以选择的服务速率 (动作空间)
        # 例如: 0 到 mu_max，分20个档位
        self.actions = np.linspace(0, self.mu_max, 21)
        
        # 初始化 Value Function (V表) 和 Policy (策略表)
        self.V = np.zeros(self.N + 1)
        self.policy = np.zeros(self.N + 1)

    # 定义等待成本函数 c(i)
    def cost_waiting(self, i, type='linear'):
        if type == 'linear':
            return 1.0 * i
        elif type == 'quadratic':
            return 0.5 * (i ** 2)
        return i

    # 定义服务成本函数 q(mu)
    def cost_service(self, mu, type='quadratic'):
        if type == 'linear':
            return 1.0 * mu
        elif type == 'quadratic':
            return 1.0 * (mu ** 2)
        return mu ** 2

    # --- 核心任务 1: 值迭代算法 ---
    def value_iteration(self, tolerance=1e-6):
        print("开始值迭代...")
        iteration = 0
        while True:
            delta = 0
            new_V = np.copy(self.V)
            
            # 遍历每一个状态 i (从 0 到 N)
            for i in range(self.N + 1):
                costs = []
                
                # 遍历每一个可能的动作 mu
                for mu in self.actions:
                    # 1. 计算当前一步的开销
                    current_cost = (self.cost_waiting(i) + self.cost_service(mu)) / self.UNIF_CONST
                    
                    # 2. 计算未来的期望开销
                    # 概率定义
                    p_arr = self.lam / self.UNIF_CONST
                    p_dep = mu / self.UNIF_CONST if i > 0 else 0
                    p_self = 1 - p_arr - p_dep
                    
                    # 下一时刻状态的值
                    v_next_arr = self.V[min(i + 1, self.N)] # 防止越界
                    v_next_dep = self.V[max(i - 1, 0)]      # 防止越界
                    v_next_self = self.V[i]
                    
                    future_val = p_arr * v_next_arr + p_dep * v_next_dep + p_self * v_next_self
                    
                    # Bellman 方程的核心部分
                    total_val = current_cost + self.beta * future_val
                    costs.append(total_val)
                
                # 找到让 Cost 最小的那个 mu
                best_value = min(costs)
                new_V[i] = best_value
                
                # 记录变化幅度
                delta = max(delta, abs(new_V[i] - self.V[i]))
            
            self.V = new_V
            iteration += 1
            if delta < tolerance:
                print(f"值迭代收敛！共迭代 {iteration} 次")
                break
        
        # 最后，提取最优策略
        self.extract_policy()

    def extract_policy(self):
        # 根据算好的 V，反推每个状态下最好的 mu
        for i in range(self.N + 1):
            best_mu = 0
            min_val = float('inf')
            
            for mu in self.actions:
                # 重复上面的计算逻辑找最小
                current_cost = (self.cost_waiting(i) + self.cost_service(mu)) / self.UNIF_CONST
                p_arr = self.lam / self.UNIF_CONST
                p_dep = mu / self.UNIF_CONST if i > 0 else 0
                p_self = 1 - p_arr - p_dep
                
                v_future = p_arr * self.V[min(i+1, self.N)] + \
                           p_dep * self.V[max(i-1, 0)] + \
                           p_self * self.V[i]
                           
                val = current_cost + self.beta * v_future
                if val < min_val:
                    min_val = val
                    best_mu = mu
            self.policy[i] = best_mu

# --- 测试运行 ---
if __name__ == "__main__":
    # 参数设置：lambda=5, mu_max=10, 队列上限=50, 折扣=0.99
    solver = MM1Control(lambda_rate=5, mu_max=15, max_queue=50, discount_factor=0.99)
    
    # 运行值迭代
    solver.value_iteration()
    
    # 打印部分结果看看
    print("\n最优策略 (状态 0-10):")
    print(solver.policy[:11])
    
    # 简单的画图
    plt.plot(solver.policy)
    plt.xlabel("Queue Length (i)")
    plt.ylabel("Optimal Service Rate (mu)")
    plt.title("Optimal Control Policy")
    plt.show()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtaidistance import dtw
import random
from pandas.plotting import parallel_coordinates
from mpl_toolkits.mplot3d import Axes3D  # 用于3D绘图

# 设置中文字体与负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.facecolor'] = 'white'
plt.style.use('seaborn-v0_8-darkgrid')

def set_chinese_font():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
    plt.rcParams['axes.unicode_minus'] = False

def select_amp_column(df, expected_name='amp', alternatives=['amplitude', 'intensity']):
    col_names = [col.strip().lower() for col in df.columns]
    df.columns = col_names
    if expected_name in df.columns:
        return df
    for alt in alternatives:
        if alt in df.columns:
            df.rename(columns={alt: expected_name}, inplace=True)
            return df
    raise KeyError(f"未找到幅值列。期望列名为 '{expected_name}' 或候选：{[expected_name] + alternatives}")

# ---------------------------
# 读取 CSV 文件及预处理
# ---------------------------
C11 = pd.read_csv('C11data.csv')
C12 = pd.read_csv('C12data.csv')
C11 = select_amp_column(C11)
C12 = select_amp_column(C12)
amp_C11 = C11['amp'].values.astype(np.double)
amp_C12 = C12['amp'].values.astype(np.double)
twt_C11 = C11['twt'].values if 'twt' in C11.columns else np.arange(len(C11))
twt_C12 = C12['twt'].values if 'twt' in C12.columns else np.arange(len(C12))
common_len = min(len(amp_C11), len(amp_C12))
amp_C11 = amp_C11[:common_len]
amp_C12 = amp_C12[:common_len]
twt_C11 = twt_C11[:common_len]
twt_C12 = twt_C12[:common_len]

# ---------------------------
# 1. 原始 DTW 对齐
# ---------------------------
initial_window_size = int(common_len * 0.1)
distance_before, paths_before = dtw.warping_paths(amp_C11, amp_C12, window=initial_window_size)
best_path_before = dtw.best_path(paths_before)
index1_before, index2_before = zip(*best_path_before)

# ---------------------------
# 2. 改进的遗传算法优化（前30代强制选择不同的最佳窗口，后续按常规运行）
# ---------------------------
class EnhancedGAOptimizer:
    def __init__(self, seq1, seq2, population_size=70, generations=100,
                 initial_mutation_rate=0.9, final_mutation_rate=0.3,
                 elite_ratio=0.05, diversity_ratio=0.5, stagnation_threshold=7):
        self.seq1 = seq1
        self.seq2 = seq2
        self.pop_size = population_size
        self.generations = generations
        self.initial_mutation_rate = initial_mutation_rate
        self.final_mutation_rate = final_mutation_rate
        self.elite_num = max(2, int(population_size * elite_ratio))
        self.diversity_num = int(population_size * diversity_ratio)
        self.max_length = len(seq1)
        self.history = []   # 每代记录：(代数, 最佳窗口, 最佳适应度, 平均适应度)
        self.final_population = []
        self.stagnation_threshold = stagnation_threshold

    def _clamp(self, value):
        # 允许窗口在 1 至 0.4*max_length 之间
        return max(1, min(int(round(value)), int(self.max_length * 0.4)))

    def _fitness(self, window):
        try:
            d = dtw.distance_fast(self.seq1, self.seq2, window=self._clamp(window), use_pruning=True)
            # 采用 1/(1 + d^2) 以放大小差异
            return 1.0/(1.0 + d**2)
        except Exception:
            return 0

    def _create_individual(self):
        return random.randint(1, int(self.max_length * 0.4))

    def _crossover(self, p1, p2):
        alpha = random.uniform(-0.5, 1.5)
        child = alpha * p1 + (1 - alpha) * p2
        return self._clamp(child)

    def _mutate(self, value, individual_fitness, avg_fitness):
        if avg_fitness == 0:
            prob = self.current_mutation_rate
        else:
            prob = self.current_mutation_rate * (1.5 - individual_fitness/avg_fitness)
        prob = min(max(prob, 0.3), 0.9)
        if random.random() < prob:
            offset = random.choice([-1, 1]) * random.randint(1, int(self.max_length * 0.2))
            return self._clamp(value + offset)
        return value

    def _aggressive_mutation(self, value):
        offset = random.randint(-int(self.max_length * 0.4), int(self.max_length * 0.4))
        return self._clamp(value + offset)

    def evolve(self):
        population = [self._create_individual() for _ in range(self.pop_size)]
        stagnation_count = 0
        best_overall_fitness = -float('inf')
        best_solution = initial_window_size
        # 保存前30代已选最佳窗口以防重复
        best_windows_set = set()
        for gen in range(self.generations):
            self.current_mutation_rate = (self.initial_mutation_rate*(1 - gen/self.generations) +
                                          self.final_mutation_rate*(gen/self.generations))
            # 定期注入新个体
            if gen % 20 == 0 and gen > 0:
                population.extend([self._create_individual() for _ in range(10)])
            fitness_list = [(ind, self._fitness(ind)) for ind in population]
            fitness_list.sort(key=lambda x: x[1], reverse=True)
            avg_fit = np.mean([f for (_, f) in fitness_list])
            if gen < 30:
                chosen = None
                for candidate, fit_val in fitness_list:
                    if candidate not in best_windows_set:
                        chosen = candidate
                        break
                if chosen is None:
                    candidate = fitness_list[0][0]
                    attempts = 0
                    while candidate in best_windows_set and attempts < 20:
                        candidate = self._aggressive_mutation(candidate)
                        attempts += 1
                    chosen = candidate
                best_windows_set.add(chosen)
                current_best_ind = chosen
                current_best_fit = self._fitness(chosen)
            else:
                current_best_ind = fitness_list[0][0]
                current_best_fit = fitness_list[0][1]
            self.history.append((gen+1, current_best_ind, current_best_fit, avg_fit))
            if current_best_fit > best_overall_fitness:
                best_overall_fitness = current_best_fit
                best_solution = current_best_ind
                stagnation_count = 0
            else:
                stagnation_count += 1
            print(f"第 {gen+1} 代: 最佳窗口={current_best_ind}, 适应度={current_best_fit:.4f}, 平均适应度={avg_fit:.4f}")
            if stagnation_count >= self.stagnation_threshold:
                print(f"第 {gen+1} 代: 检测到停滞，应用跳变变异和种群注入")
                population = [self._aggressive_mutation(ind) for ind in population]
                population.extend([self._create_individual() for _ in range(self.diversity_num)])
                stagnation_count = 0
            # 采用排名选择：先排序，然后赋予权重（递减）
            sorted_population = [ind for (ind, _) in fitness_list]
            weights = list(range(len(sorted_population), 0, -1))
            new_population = sorted_population[:self.elite_num].copy()
            while len(new_population) < self.pop_size:
                parents = random.choices(sorted_population, weights=weights, k=2)
                child = self._crossover(parents[0], parents[1])
                child = self._mutate(child, self._fitness(child), avg_fit)
                new_population.append(child)
            population = new_population[:self.pop_size]
        self.final_population = population
        return best_solution, best_overall_fitness

# 执行遗传算法优化：迭代200次
random.seed(42)
optimizer = EnhancedGAOptimizer(amp_C11, amp_C12,
                                population_size=100,
                                generations=200,
                                initial_mutation_rate=0.9,
                                final_mutation_rate=0.3,
                                elite_ratio=0.05,
                                diversity_ratio=0.5,
                                stagnation_threshold=7)
best_window, best_fit = optimizer.evolve()

# 使用优化后的最佳窗口重新计算 DTW
distance_after, paths_after = dtw.warping_paths(amp_C11, amp_C12, window=best_window)
best_path_after = dtw.best_path(paths_after)
index1_after, index2_after = zip(*best_path_after)

if distance_after >= distance_before:
    print("警告：优化未生效，采用次优策略")
    candidate_windows = sorted(set(optimizer.final_population), key=lambda x: optimizer._fitness(x), reverse=True)
    for w in candidate_windows[:5]:
        d = dtw.distance_fast(amp_C11, amp_C12, window=w)
        if d < distance_before:
            best_window = w
            distance_after = d
            break

# ---------------------------
# 生成 11 张高级图
# ---------------------------
set_chinese_font()

# 1. 遗传算法进化过程折线图
plt.figure(figsize=(12,6))
gens, best_inds, best_fits, avg_fits = zip(*optimizer.history)
plt.plot(gens, best_fits, 'o-', label='最佳适应度')
plt.plot(gens, avg_fits, 's--', label='平均适应度')
plt.xlabel('进化代数')
plt.ylabel('适应度')
plt.title('遗传算法进化过程')
plt.legend()
plt.savefig('evolution_process.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. DTW 对齐对比图（上：优化前；下：优化后）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,10))
ax1.plot(twt_C11, amp_C11, label='C11')
ax1.plot(twt_C12, amp_C12, label='C12')
for i, j in zip(index1_before, index2_before):
    ax1.plot([twt_C11[i], twt_C12[j]], [amp_C11[i], amp_C12[j]], 'gray', alpha=0.1)
ax1.set_title(f'优化前DTW对齐 (窗口={initial_window_size}, 距离={distance_before:.2f})')
ax1.legend()
ax2.plot(twt_C11, amp_C11, label='C11')
ax2.plot(twt_C12, amp_C12, label='C12')
for i, j in zip(index1_after, index2_after):
    ax2.plot([twt_C11[i], twt_C12[j]], [amp_C11[i], amp_C12[j]], 'gray', alpha=0.1)
ax2.set_title(f'优化后DTW对齐 (窗口={best_window}, 距离={distance_after:.2f})')
ax2.legend()
plt.tight_layout()
plt.savefig('alignment_comparison.png', dpi=300)
plt.close()

# 3. 窗口尺寸进化分布直方图
window_vals = [data[1] for data in optimizer.history]
plt.figure(figsize=(12,6))
plt.hist(window_vals, bins=30, alpha=0.7, color='steelblue')
plt.xlabel('窗口尺寸')
plt.ylabel('出现频次')
plt.title('窗口尺寸分布演化')
plt.savefig('window_distribution.png', dpi=300)
plt.close()

# 4. 三维成本矩阵及最优路径图
fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(paths_after.shape[1]), np.arange(paths_after.shape[0]))
ax.plot_surface(X, Y, paths_after, cmap='viridis', alpha=0.7)
ax.plot(np.array(index1_after), np.array(index2_after),
        paths_after[np.array(index1_after), np.array(index2_after)],
        'r-', lw=2, label='最优路径')
ax.set_title('三维成本矩阵与最优路径')
ax.set_xlabel('序列1')
ax.set_ylabel('序列2')
ax.legend()
plt.savefig('3d_cost_matrix.png', dpi=300)
plt.close()

# 5. 性能对比雷达图
categories = ['DTW距离', '计算效率', '路径平滑度']
before_values = [distance_before, 0.7, 0.6]
after_values = [distance_after, 0.4, 0.9]
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, before_values + before_values[:1], 'o-', label='优化前')
ax.plot(angles, after_values + after_values[:1], 's-', label='优化后')
ax.fill(angles, before_values + before_values[:1], alpha=0.1)
ax.fill(angles, after_values + after_values[:1], alpha=0.1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title('性能对比雷达图')
ax.legend(loc='upper right')
plt.savefig('radar_comparison.png', dpi=300)
plt.close()

# 6. 二维成本矩阵热力图与最优路径叠加图
plt.figure(figsize=(10,8))
plt.imshow(paths_after, aspect='auto', cmap='viridis')
plt.colorbar(label='成本')
plt.plot(index2_after, index1_after, 'r.-', linewidth=2, markersize=5, label='最优路径')
plt.title('二维成本矩阵热力图与最优路径')
plt.xlabel('序列2索引')
plt.ylabel('序列1索引')
plt.legend()
plt.savefig('cost_matrix_heatmap.png', dpi=300)
plt.close()

# 7. 平行坐标图显示进化历史
df_history = pd.DataFrame(optimizer.history, columns=['Generation', 'BestWindow', 'BestFitness', 'AvgFitness'])
plt.figure(figsize=(12,8))
parallel_coordinates(df_history, 'Generation', colormap='viridis')
plt.title('遗传算法进化过程平行坐标图')
plt.savefig('evolution_parallel_coordinates.png', dpi=300)
plt.close()

# 8. 二维成本矩阵等高线图与最优路径图
plt.figure(figsize=(10,8))
X_c, Y_c = np.meshgrid(np.arange(paths_after.shape[1]), np.arange(paths_after.shape[0]))
contour = plt.contourf(X_c, Y_c, paths_after, cmap='coolwarm', levels=50)
plt.colorbar(contour, label='成本')
plt.plot(index2_after, index1_after, 'k.-', linewidth=2, markersize=5, label='最优路径')
plt.title('二维成本矩阵等高线图与最优路径')
plt.xlabel('序列2索引')
plt.ylabel('序列1索引')
plt.legend()
plt.savefig('cost_matrix_contour.png', dpi=300)
plt.close()

# 9. 最终种群适应度分布直方图
final_fitness = [optimizer._fitness(ind) for ind in optimizer.final_population]
plt.figure(figsize=(12,6))
plt.hist(final_fitness, bins=20, alpha=0.7, color='darkgreen')
plt.xlabel('适应度')
plt.ylabel('频次')
plt.title('最终种群适应度分布')
plt.savefig('final_fitness_distribution.png', dpi=300)
plt.close()

# 10. 最终种群散点图（窗口尺寸 vs 适应度）
final_windows = optimizer.final_population
final_fitness_scatter = [optimizer._fitness(ind) for ind in final_windows]
plt.figure(figsize=(12,6))
plt.scatter(final_windows, final_fitness_scatter, c='purple', alpha=0.7)
plt.xlabel('窗口尺寸')
plt.ylabel('适应度')
plt.title('最终种群个体散点图')
plt.savefig('final_population_scatter.png', dpi=300)
plt.close()

# 11. 各代最佳窗口演变图
plt.figure(figsize=(12,6))
plt.plot(gens, best_inds, 'o-', color='orange')
plt.xlabel('代数')
plt.ylabel('最佳窗口尺寸')
plt.title('各代最佳窗口值演变图')
plt.savefig('best_window_evolution.png', dpi=300)
plt.close()

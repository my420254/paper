import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 全局学术排版与字体设置
# ==============================================================================
FONT_PATH = '/Times_New_Roman.ttf'
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.family'] = 'Times New Roman'
else:
    print(f"⚠️ 未找到字体文件 {FONT_PATH}，将使用系统默认 serif 字体。")
    plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

DATA_DIR = '/paper_results'

def get_csv_data(dataset, filename):
    filepath = os.path.join(DATA_DIR, dataset, filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

# ==============================================================================
# 2. 敏感性分析图 (图例统一放底部)
# ==============================================================================
def plot_sensitivity():
    print("开始绘制敏感性分析图...")
    # 增加底部空白区域来放图例: figsize 高度稍加，gridspec_kw 调整比例
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2)) 
    
    df_res_temp = get_csv_data('res14', 'sensitivity_cl_temp.csv')
    df_lap_temp = get_csv_data('lap14', 'sensitivity_cl_temp.csv')
    df_res_neg = get_csv_data('res14', 'sensitivity_neg_ratio.csv')
    df_lap_neg = get_csv_data('lap14', 'sensitivity_neg_ratio.csv')

    # 用于统一收集图例句柄
    lines = []
    labels = []

    # --- 图 (a): CL Temperature ---
    ax1 = axes[0]
    if df_res_temp is not None and df_lap_temp is not None:
        l1, = ax1.plot(df_res_temp['param_value'], df_res_temp['test_f1'], 
                       marker='o', color='#1f77b4', linewidth=2.5, markersize=8)
        l2, = ax1.plot(df_lap_temp['param_value'], df_lap_temp['test_f1'], 
                       marker='s', color='#d62728', linewidth=2.5, markersize=8)
        # 收集第一张图的 line，作为全局图例
        if not lines:
            lines.extend([l1, l2])
            labels.extend(['14res', '14lap'])
            
    ax1.set_xlabel('CL Temperature $\\tau$', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1 Score (%)', fontsize=14, fontweight='bold')
    ax1.set_title('(a) Sensitivity to CL Temperature', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- 图 (b): Negative Ratio ---
    ax2 = axes[1]
    if df_res_neg is not None and df_lap_neg is not None:
        ax2.plot(df_res_neg['param_value'], df_res_neg['test_f1'], 
                 marker='o', color='#1f77b4', linewidth=2.5, markersize=8)
        ax2.plot(df_lap_neg['param_value'], df_lap_neg['test_f1'], 
                 marker='s', color='#d62728', linewidth=2.5, markersize=8)
                 
    ax2.set_xlabel('Negative Ratio $\\rho$', fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1 Score (%)', fontsize=14, fontweight='bold')
    ax2.set_title('(b) Sensitivity to Negative Ratio', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 紧凑排版，但底部留出 15% 的空间给全局图例
    plt.tight_layout(rect=[0, 0.12, 1, 1]) 
    
    # 统一在底部中央添加图例
    fig.legend(lines, labels, loc='lower center', ncol=2, 
               frameon=True, edgecolor='black', fontsize=12,
               bbox_to_anchor=(0.5, 0.0), borderaxespad=0.)

    out_file = os.path.join('/newppp', 'sensitivity_analysis.pdf')
    plt.savefig(out_file, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    print(f"✅ 敏感性分析图已保存至: {out_file}\n")


# ==============================================================================
# 3. t-SNE 降维可视化图
# ==============================================================================
def plot_tsne():
    print("开始绘制 t-SNE 可视化图...")
    tsne_file = os.path.join(DATA_DIR, 'res14', 'seed42_full_tsne.npz')
    
    if not os.path.exists(tsne_file):
        print(f"⚠️ 找不到 t-SNE 数据文件: {tsne_file}")
        return
        
    data = np.load(tsne_file)
    ph = data['pair_hidden']
    labels = data['labels']
    
    print(f"读取到 {len(ph)} 个特征向量，正在进行 t-SNE 降维计算...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    ph_2d = tsne.fit_transform(ph)
    
    fig, ax = plt.subplots(figsize=(6.5, 5))
    
    colors = ['#e0e0e0', '#d62728', '#7f7f7f', '#1f77b4'] 
    names = ['Negative Pairs (None)', 'Negative Sentiment', 'Neutral Sentiment', 'Positive Sentiment']
    
    draw_order = [0, 3, 1, 2] 
    
    for i in draw_order: 
        mask = (labels == i)
        if not np.any(mask): continue
        is_bg = (i == 0)
        ax.scatter(ph_2d[mask, 0], ph_2d[mask, 1], 
                   c=colors[i], label=names[i], 
                   alpha=0.3 if is_bg else 0.85, 
                   s=15 if is_bg else 45,        
                   edgecolors='none' if is_bg else 'white', 
                   linewidth=0.5 if not is_bg else 0)
                   
    ax.set_title('t-SNE Visualization of Pair Representations', fontsize=14, fontweight='bold')
    ax.set_xticks([]) 
    ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
        
    ax.legend(loc='lower left', frameon=True, edgecolor='black', fontsize=10)
    
    plt.tight_layout()
    out_file = os.path.join('/newppp', 'tsne_visualization.pdf')
    plt.savefig(out_file, format='pdf', bbox_inches='tight', dpi=300)
    plt.show()
    print(f"✅ t-SNE 可视化图已保存至: {out_file}\n")

plot_sensitivity()
plot_tsne()

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.patches as patches

def plot_hidden_states_pca(feature_dir, output_path="pca_visualization_refined.png"):
    method_features = {}
    
    # 1. 映射字典
    name_mapping = {
        "Baseline": "Manual",
        "Qwen": "MPO",
        "MPO": "DPL-RAG"
    }
    
    for file in os.listdir(feature_dir):
        if file.endswith("_features.npy"):
            original_name = file.replace("_features.npy", "")
            display_name = name_mapping.get(original_name, original_name)
            method_features[display_name] = np.load(os.path.join(feature_dir, file))
            
    if not method_features:
        print("未找到特征文件！")
        return

    # 2. PCA 拟合
    all_features = np.vstack(list(method_features.values()))
    pca = PCA(n_components=2)
    pca.fit(all_features)

    # 3. 绘图设置
    plt.style.use('seaborn-v0_8-paper') 
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    
    # 默认调色板
    default_colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, len(method_features)))
    markers = ['o', 's', '^', 'p', 'H', 'D', '*']
    
    for i, (method_name, features) in enumerate(method_features.items()):
        reduced_features = pca.transform(features)
        x = reduced_features[:, 0]
        y = reduced_features[:, 1]
        
        # 指定颜色：如果是 DPL-RAG 则强制为红色，否则使用调色板
        current_color = 'red' if method_name == "DPL-RAG" else default_colors[i]
        
        # 绘制散点
        ax.scatter(
            x, y, 
            label=method_name, 
            color=current_color,
            marker=markers[i % len(markers)],
            s=45,
            alpha=0.8,
            edgecolors='white', 
            linewidths=0.5,
            zorder=3
        )

        # 绘制置信椭圆
        cov = np.cov(x, y)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        ell = patches.Ellipse(
            xy=(np.mean(x), np.mean(y)),
            width=lambda_[0]*2*2, height=lambda_[1]*2*2,
            angle=np.rad2deg(np.arccos(v[0, 0])),
            color=current_color # 椭圆颜色与散点同步
        )
        ell.set_facecolor(current_color)
        ell.set_alpha(0.1) 
        ax.add_artist(ell)

    # 4. 细节美化（保持不变）
    ax.set_xlabel('PC 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('PC 2', fontsize=12, fontweight='bold')
    ax.set_title('PCA Visualization of Hidden States', fontsize=14, pad=20)
    
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(method_features), 
        frameon=False,
        fontsize=10
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"PCA图（DPL-RAG已设为红色）保存至 {output_path}")

if __name__ == "__main__":
    plot_hidden_states_pca(feature_dir="./logs/pca_features")
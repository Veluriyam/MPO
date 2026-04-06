import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_hidden_states_pca(feature_dir, output_path="pca_visualization.png"):
    method_features = {}
    
    # 1. 加载所有方法的特征数据
    for file in os.listdir(feature_dir):
        if file.endswith("_features.npy"):
            method_name = file.replace("_features.npy", "")
            method_features[method_name] = np.load(os.path.join(feature_dir, file))
            
    if not method_features:
        print("未找到特征文件！")
        return

    # 2. 合并数据进行统一的PCA拟合
    all_features = np.vstack(list(method_features.values()))
    pca = PCA(n_components=2)
    pca.fit(all_features)

    # 3. 绘图
    plt.figure(figsize=(8, 6))
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    for i, (method_name, features) in enumerate(method_features.items()):
        reduced_features = pca.transform(features)
        plt.scatter(
            reduced_features[:, 0], 
            reduced_features[:, 1], 
            label=method_name, 
            marker=markers[i % len(markers)],
            alpha=0.7
        )

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Visualization of Hidden States by PCA')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    print(f"PCA图已保存至 {output_path}")

if __name__ == "__main__":
    plot_hidden_states_pca(feature_dir="./logs/pca_features")
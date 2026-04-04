import os
import json
import shutil

def prepare_cub_for_mpo(source_dir, target_dir, target_class="cuckoo"):
    # 辅助函数：读取 txt 映射文件
    def read_txt(filename):
        with open(os.path.join(source_dir, filename), 'r') as f:
            return {line.split()[0]: line.split()[1] for line in f}

    print("正在读取标注文件...")
    images = read_txt('images.txt')
    classes = read_txt('classes.txt')
    labels = read_txt('image_class_labels.txt')
    splits = read_txt('train_test_split.txt')

    train_data, test_data = [], []

    # 遍历并过滤出 cuckoo 数据
    for img_id, filepath in images.items():
        class_name = classes[labels[img_id]].lower()
        if target_class in class_name:
            clean_label = class_name.split('.')[-1]
            item = {"filename": filepath, "label": clean_label}
            
            if splits[img_id] == '1': # 1 为训练集
                train_data.append(item)
            else:                     # 0 为测试集
                test_data.append(item)

    # 1. 创建目标目录并保存 JSON
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, f'{target_class}_train.json'), 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(target_dir, f'{target_class}_test.json'), 'w') as f:
        json.dump(test_data, f, indent=4)

    # 2. 复制 images 文件夹 (如果目标路径下还不存在)
    target_images_dir = os.path.join(target_dir, 'images')
    if not os.path.exists(target_images_dir):
        print(f"正在复制 images 文件夹到 {target_images_dir}，请稍候...")
        shutil.copytree(os.path.join(source_dir, 'images'), target_images_dir)

    print(f"\n处理完成！")
    print(f"- 训练集 {target_class}_train.json: {len(train_data)} 张图像")
    print(f"- 测试集 {target_class}_test.json: {len(test_data)} 张图像")
    print(f"- 输出目录: {os.path.abspath(target_dir)}")

if __name__ == "__main__":
    # 配置你的实际路径
    SOURCE_DIR = r"/lizhengton/workspace/yp/MPO/datasets/test/CUB-200-2011/CUB-200-2011/CUB-200-2011"
    TARGET_DIR = "/lizhengton/workspace/yp/MPO/datasets/classification/cub/"
    
    prepare_cub_for_mpo(SOURCE_DIR, TARGET_DIR)
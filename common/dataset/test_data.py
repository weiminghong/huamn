import os
import random

def split_dataset(image_dir, train_txt_path, val_txt_path, train_ratio=0.95):
    """
    将目标文件夹下的图片按 0.95 的比例划分为训练集和验证集。
    """
    # 0. 确保输出目录存在
    os.makedirs(os.path.dirname(train_txt_path), exist_ok=True)

    # 1. 获取目录下所有的图片文件名
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_images = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(image_dir, f))
    ]

    # 2. 检查文件夹是否为空
    total_images = len(all_images)
    if total_images == 0:
        print(f"❌ 错误：在 {image_dir} 中没有找到任何图片文件！")
        return

    print(f"📂 共找到 {total_images} 张图片。")

    # 3. 打乱图片顺序
    random.seed(42)  # 固定随机种子，保证结果可复现
    random.shuffle(all_images)

    # 4. 计算划分的索引点
    split_index = int(total_images * train_ratio)

    # 5. 切片划分列表
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    print(f"✂️  划分比例: {train_ratio*100}% / {(1-train_ratio)*100}%")
    print(f"   - 训练集 (train): {len(train_images)} 张")
    print(f"   - 验证集 (val): {len(val_images)} 张")

    # 6. 将文件名写入 txt 文件
    with open(train_txt_path, 'w', encoding='utf-8') as f_train:
        for img in train_images:
            f_train.write(img + '\n')

    with open(val_txt_path, 'w', encoding='utf-8') as f_val:
        for img in val_images:
            f_val.write(img + '\n')

    print(f"✅ 划分完成！文件已保存至：")
    print(f" -> {train_txt_path}")
    print(f" -> {val_txt_path}")


if __name__ == '__main__':
    # 1. 你的图片存放路径
    IMAGE_DIR = '/share2/program/weiminghong/coco/coco_person/images'

    # 2. 生成的 txt 文件保存路径
    TRAIN_TXT = '/share2/program/weiminghong/coco/coco_person/data_test/train.txt'
    VAL_TXT = '/share2/program/weiminghong/coco/coco_person/data_test/val.txt'

    # 3. 执行划分
    split_dataset(IMAGE_DIR, TRAIN_TXT, VAL_TXT, train_ratio=0.95)
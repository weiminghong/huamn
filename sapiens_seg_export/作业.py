import os

def fast_check_datasets():
    labels_dir = "/share2/program/weiminghong/coco/coco_person/parsing_sapiens/parsing_labels"
    images_dir = "/share2/program/weiminghong/coco/coco_person/images"

    if not os.path.exists(labels_dir):
        print(f"❌ 找不到目录: {labels_dir}")
        return
    if not os.path.exists(images_dir):
        print(f"❌ 找不到目录: {images_dir}")
        return

    valid_img_exts = {'.jpg', '.jpeg'}
    valid_lbl_exts = {'.png', '.jpg'}

    labels_stems = set()
    images_stems = set()

    print("🚀 开始极速扫描目录 (20万级数据可能需要十几秒，请稍等)...")

    # 1. 扫描并统计 Labels 数量
    print(f"📂 正在扫描 Labels 目录: {labels_dir}")
    with os.scandir(labels_dir) as entries:
        for entry in entries:
            if entry.is_file():
                name, ext = os.path.splitext(entry.name)
                if ext.lower() in valid_lbl_exts:
                    labels_stems.add(name)
    print(f"   ✅ Labels 扫描完毕，共统计到: {len(labels_stems)} 个文件。")

    # 2. 扫描并统计 Images 数量
    print(f"📂 正在扫描 Images 目录: {images_dir}")
    with os.scandir(images_dir) as entries:
        for entry in entries:
            if entry.is_file():
                name, ext = os.path.splitext(entry.name)
                if ext.lower() in valid_img_exts:
                    images_stems.add(name)
    print(f"   ✅ Images 扫描完毕，共统计到: {len(images_stems)} 个文件。")

    # 3. 数量初步核对
    print("\n📊 数量核对结果:")
    if len(labels_stems) == len(images_stems):
        print(f"   🎉 数量一致！(都是 {len(images_stems)} 个)")
    else:
        print(f"   ⚠️ 数量不一致！Labels: {len(labels_stems)} 个 vs Images: {len(images_stems)} 个")

    # 4. 深入找出文件名差异 (即使数量一致，也可能存在张三李四对不上的情况)
    missing_in_labels = images_stems - labels_stems
    missing_in_images = labels_stems - images_stems

    if not missing_in_labels and not missing_in_images:
        print("\n✅ 完美匹配！不仅数量一致，且所有文件完全一一对应。")
    else:
        print("\n🚨 进一步排查：发现不对应的文件！")

        if missing_in_labels:
            print(f"\n❌ 以下 {len(missing_in_labels)} 张原图缺少对应的 Parsing Label (截取前 10 个):")
            for name in list(missing_in_labels)[:10]:
                print(f"   -> {name}")

        if missing_in_images:
            print(f"\n❌ 以下 {len(missing_in_images)} 个 Parsing Label 找不到对应的原图 (截取前 10 个):")
            for name in list(missing_in_images)[:10]:
                print(f"   -> {name}")

if __name__ == "__main__":
    fast_check_datasets()
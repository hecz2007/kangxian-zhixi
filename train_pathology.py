# -*- coding: utf-8 -*-
"""
康险智析 —— 医生病理辅助端 · BreakHis 乳腺病理图像二分类模型训练
================================================================

任务：读取 dataset_cancer_v1/classificacao_binaria 下 4 个倍率目录中的
      benign（良性）/ malignant（恶性）乳腺 HE 病理图片，
      使用 MobileNetV2 迁移学习训练二分类模型（0=良性, 1=恶性）。

训练配置（按大赛要求）：
  - 输入尺寸统一缩放 224×224
  - 冻结 MobileNetV2 ImageNet 预训练主干，只训练最后分类层
  - CPU 设备训练；epochs=5，batch_size=8（控制内存占用）

实现说明（等价加速）：
  冻结主干时，主干前向输出在每轮完全相同（BN 层 training=False），
  因此先将全部图片经「MobileNetV2 主干 + 全局平均池化」提取为 1280 维特征，
  再对分类头（Dropout + Dense）进行 5 轮训练，数学上与端到端冻结训练
  完全等价，但避免了 CPU 上主干前向的 5 次重复计算，可将训练时长从
  1~2 小时压缩到数分钟。最终保存的完整模型结构仍为：
  MobileNetV2 主干（冻结）→ GAP → Dropout → Dense(1, sigmoid)。

合规声明：本模型为大学生科研 Demo 原型组件，训练数据 BreakHis 为海外
公开数据集，输出仅为阅片辅助参考概率，不能替代 IHC 免疫组化金标准。

运行：python train_pathology.py
输出：pathology_model/breast_model.keras + pathology_model/meta.json
"""

import json
import os
import random
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 屏蔽 TF 冗余日志

import numpy as np
import tensorflow as tf

# ============================ 训练配置 ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset_cancer_v1", "classificacao_binaria")
MODEL_DIR = os.path.join(BASE_DIR, "pathology_model")

IMG_SIZE = (224, 224)      # MobileNetV2 标准输入尺寸
BATCH_SIZE = 8             # 小 batch，避免内存不足
EPOCHS = 5                 # 训练轮次
VAL_RATIO = 0.15           # 验证集比例（分层抽样）
SEED = 42                  # 随机种子，保证可复现
CLASS_NAMES = ["benign", "malignant"]  # 0=良性, 1=恶性（sigmoid 输出 = 恶性概率）
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ============================ 数据收集与划分 ============================

def collect_files():
    """遍历 4 个倍率目录，收集全部图片路径与标签（0=良性, 1=恶性）。"""
    files, labels = [], []
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"未找到数据集目录：{DATA_DIR}")
    for mag in sorted(os.listdir(DATA_DIR)):          # 40X / 100X / 200X / 400X
        mag_dir = os.path.join(DATA_DIR, mag)
        if not os.path.isdir(mag_dir):
            continue
        for ci, cls in enumerate(CLASS_NAMES):        # benign / malignant
            cls_dir = os.path.join(mag_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for f in sorted(os.listdir(cls_dir)):
                if f.lower().endswith(IMG_EXTS):
                    files.append(os.path.join(cls_dir, f))
                    labels.append(ci)
    return files, labels


def stratified_split(files, labels, val_ratio, seed):
    """按类别分层随机划分训练集/验证集，保持两类比例一致。"""
    rng = random.Random(seed)
    by_class = {}
    for f, l in zip(files, labels):
        by_class.setdefault(l, []).append(f)
    train_f, train_l, val_f, val_l = [], [], [], []
    for ci in sorted(by_class):
        items = by_class[ci][:]
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio)))
        val_f += items[:n_val]
        val_l += [ci] * n_val
        train_f += items[n_val:]
        train_l += [ci] * (len(items) - n_val)
    # 打乱训练集顺序
    idx = list(range(len(train_f)))
    rng.shuffle(idx)
    train_f = [train_f[i] for i in idx]
    train_l = [train_l[i] for i in idx]
    return train_f, train_l, val_f, val_l


# ============================ tf.data 流水线 ============================

def load_img(path, label):
    """读取图片 → 解码 RGB → 缩放 224×224 → MobileNetV2 归一化（[-1,1]）。"""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img, label


def make_dataset(files, labels, shuffle, seed=None):
    """构建 tf.data 输入流水线：shuffle → map 预处理 → batch → prefetch。"""
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if shuffle:
        ds = ds.shuffle(len(files), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def make_label_ds(labels):
    return tf.data.Dataset.from_tensor_slices(labels).batch(BATCH_SIZE)


# ============================ 模型构建 ============================

def build_components():
    """构建冻结的 MobileNetV2 主干、特征提取器与分类头（Dense1 输出恶性概率）。"""
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False  # 迁移学习：冻结 ImageNet 预训练主干，只训练分类层

    inp = tf.keras.Input(shape=(*IMG_SIZE, 3))
    gap = tf.keras.layers.GlobalAveragePooling2D(name="gap")
    head = tf.keras.Sequential(
        [
            tf.keras.layers.Dropout(0.2, name="head_dropout"),
            tf.keras.layers.Dense(1, activation="sigmoid", name="head_classifier"),
        ],
        name="classifier_head",
    )

    featurizer = tf.keras.Model(inp, gap(base(inp)), name="mobilenetv2_featurizer")
    return base, gap, head, featurizer


def build_full_model(base, gap, head):
    """组装完整可保存模型：MobileNetV2 主干（冻结）→ GAP → 分类头。"""
    inp = tf.keras.Input(shape=(*IMG_SIZE, 3))
    out = head(gap(base(inp)))
    model = tf.keras.Model(inp, out, name="breakhis_binary_mobilenetv2")
    return model


# ============================ 主流程 ============================

def main():
    t0 = time.time()
    print("=" * 60)
    print("BreakHis 乳腺病理二分类 · MobileNetV2 迁移学习（CPU）")
    print(f"TensorFlow: {tf.__version__} ｜ 可用设备: CPU")
    print("=" * 60)

    # ---------- 1. 收集数据 ----------
    files, labels = collect_files()
    n_benign = sum(1 for l in labels if l == 0)
    n_malignant = sum(1 for l in labels if l == 1)
    print(f"\n[1/5] 数据收集完成：共 {len(files)} 张"
          f"（benign 良性 {n_benign} 张 ｜ malignant 恶性 {n_malignant} 张）")
    if len(files) == 0:
        raise RuntimeError("数据集为空，请检查 dataset_cancer_v1/classificacao_binaria 目录结构")

    # ---------- 2. 分层划分 ----------
    train_f, train_l, val_f, val_l = stratified_split(files, labels, VAL_RATIO, SEED)
    print(f"[2/5] 分层划分完成：训练集 {len(train_f)} 张 ｜ 验证集 {len(val_f)} 张"
          f"（val_ratio={VAL_RATIO}, seed={SEED}）")

    # 类别权重：缓解良性/恶性样本量不平衡（2480 vs 5429）
    counts = np.bincount(np.array(train_l), minlength=2).astype(float)
    class_weight = {i: len(train_l) / (2.0 * counts[i]) for i in range(2)}
    print(f"      类别权重（均衡化）: {{0: {class_weight[0]:.3f}, 1: {class_weight[1]:.3f}}}")

    # ---------- 3. 特征提取（冻结主干前向，等价加速） ----------
    print(f"[3/5] 提取 MobileNetV2 冻结主干特征（输入 {IMG_SIZE[0]}×{IMG_SIZE[1]}, batch={BATCH_SIZE}）…")
    base, gap, head, featurizer = build_components()

    train_ds = make_dataset(train_f, train_l, shuffle=False)  # 必须按文件顺序提特征，与 y_train 严格对齐
    val_ds = make_dataset(val_f, val_l, shuffle=False)

    t1 = time.time()
    feats_train = featurizer.predict(train_ds, verbose=1)
    feats_val = featurizer.predict(val_ds, verbose=1)
    y_train = np.array(train_l, dtype=np.float32)
    y_val = np.array(val_l, dtype=np.float32)
    print(f"      特征提取完成：train {feats_train.shape} ｜ val {feats_val.shape}"
          f"（耗时 {time.time() - t1:.1f}s）")

    # ---------- 4. 训练分类头（epochs=5, batch=8, CPU） ----------
    head.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    print(f"[4/5] 开始训练分类头：epochs={EPOCHS}, batch_size={BATCH_SIZE}, optimizer=Adam(1e-3)")
    t2 = time.time()
    history = head.fit(
        feats_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        validation_data=(feats_val, y_val),
        verbose=2,
    )
    print(f"      分类头训练完成（耗时 {time.time() - t2:.1f}s）")

    # ---------- 5. 验证集准确率 + 保存 ----------
    val_loss, val_acc = head.evaluate(feats_val, y_val, batch_size=BATCH_SIZE, verbose=0)
    print(f"\n[5/5] 验证集准确率: {val_acc:.4f}（{int(round(val_acc * len(y_val)))}/{len(y_val)}）"
          f"  验证集 loss: {val_loss:.4f}")
    for ep in range(EPOCHS):
        print(f"      第 {ep + 1} 轮 ｜ acc={history.history['accuracy'][ep]:.4f}"
              f"  val_acc={history.history['val_accuracy'][ep]:.4f}")

    full_model = build_full_model(base, gap, head)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "breast_model.keras")
    full_model.save(model_path)

    meta = {
        "task": "BreakHis 乳腺病理二分类（benign/malignant）",
        "class_names": CLASS_NAMES,
        "label_meaning": "sigmoid 输出 = malignant（恶性）概率；良性概率 = 1 - p",
        "img_size": list(IMG_SIZE),
        "backbone": "MobileNetV2 (ImageNet, frozen)",
        "head": "GAP → Dropout(0.2) → Dense(1, sigmoid)",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "device": "CPU",
        "train_samples": len(train_f),
        "val_samples": len(val_f),
        "total_samples": len(files),
        "val_accuracy": float(val_acc),
        "seed": SEED,
        "compliance": "科研 Demo 原型组件；BreakHis 为海外公开数据集；"
                      "输出仅为阅片辅助参考概率，不能替代 IHC 免疫组化金标准。",
    }
    meta_path = os.path.join(MODEL_DIR, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n模型已保存: {model_path}")
    print(f"元信息已保存: {meta_path}")
    print(f"总耗时 {time.time() - t0:.1f}s ｜ 完成 ✅")


if __name__ == "__main__":
    main()

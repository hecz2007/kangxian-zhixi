# -*- coding: utf-8 -*-
"""
训练脚本：生成 risk_model.pkl（页面1·风险评估模型）
=====================================================

⚠️ 合规声明：
1. 本脚本仅为大学生科研Demo原型的模型训练复现脚本，严禁商用、严禁临床使用。
2. 训练数据为威斯康星乳腺肿瘤诊断数据集（WDBC），系海外公开数据集，
   缺少国内人群样本，模型结论不可落地临床。
3. 本模型输出的"风险分层"仅为科研筛查参考，不构成任何疾病诊断，
   不能替代医生判断，免疫组化（IHC）检测为确诊金标准。

模型说明：
- 数据集：sklearn 内置 WDBC（569 个样本、30 个细胞形态学特征）
- 特征：为保持大众端表单简洁，仅选取 10 个均值（mean*）形态学特征
- 算法：StandardScaler 标准化 + LogisticRegression（逻辑回归，可解释性强）
- 输出：P(恶性) 概率 → 低风险(<1/3) / 中风险(1/3~2/3) / 高风险(>2/3)
        （阈值为本Demo固定设定，无临床依据）

运行方式：python train_model.py
运行后在同目录生成 risk_model.pkl，供 app.py 加载。
"""

import os

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 模型文件保存路径（与 app.py 同目录）
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "risk_model.pkl")


def train_and_save() -> dict:
    """在 WDBC 公开数据集上训练逻辑回归模型，并把模型与元信息打包保存为 pkl。"""

    # ---------- 1. 加载 WDBC 公开数据集 ----------
    # WDBC：Wisconsin Diagnostic Breast Cancer，569 个样本
    # 特征为细针穿刺（FNA）细胞的形态学量化指标：均值/标准误/最差值 各 10 个
    data = load_breast_cancer()
    X, y = data.data, data.target  # 标签：0=恶性(malignant)，1=良性(benign)

    # ---------- 2. 特征选择：仅保留 10 个均值特征 ----------
    # 均值特征与大众端表单的"肿瘤量化指标滑块"一一对应，便于演示与解释
    mean_idx = [i for i, name in enumerate(data.feature_names) if name.startswith("mean")]
    X_mean = X[:, mean_idx]
    feature_names = [str(n) for n in data.feature_names[mean_idx]]

    # ---------- 3. 划分训练/测试集（分层抽样，固定随机种子保证可复现） ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X_mean, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------- 4. 训练：标准化 + 逻辑回归 ----------
    # 逻辑回归系数可解释（权重正负与大小可解释），符合策划书"可解释、可部署、可复核"原则
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)

    # 留出测试集参考准确率：仅为公开数据集上的技术验证指标，不代表临床性能
    test_accuracy = float(pipeline.score(X_test, y_test))

    # ---------- 5. 打包保存（模型 + 元信息一起存入 pkl） ----------
    bundle = {
        "model": pipeline,                          # sklearn Pipeline，可直接 predict / predict_proba
        "feature_names": feature_names,             # 10 个均值特征名（与表单一一对应）
        "target_names": [str(n) for n in data.target_names],  # ['malignant', 'benign']
        "test_accuracy": test_accuracy,             # 留出测试集参考准确率（仅供技术验证）
        "n_samples": int(X_mean.shape[0]),          # 样本量：569
        "dataset": "WDBC（威斯康星乳腺肿瘤诊断数据集，海外公开数据集）",
        "risk_thresholds": (1.0 / 3.0, 2.0 / 3.0),  # 低/中/高分层阈值（Demo设定，无临床依据）
    }
    joblib.dump(bundle, MODEL_PATH)

    print("=" * 60)
    print("risk_model.pkl 训练完成并已保存：", MODEL_PATH)
    print(f"  样本量          ：{bundle['n_samples']}（WDBC 公开数据集）")
    print(f"  特征数          ：{len(feature_names)}（10 个均值形态学特征）")
    print(f"  留出测试集准确率：{test_accuracy:.4f}（仅为技术验证参考，不代表临床性能）")
    print("  ⚠️ 本模型为科研Demo原型，缺少国内人群样本，不可落地临床。")
    print("=" * 60)
    return bundle


if __name__ == "__main__":
    train_and_save()

# -*- coding: utf-8 -*-
"""
康险智析 —— 激素依赖性肿瘤风险评估与病理AI探究原型（大学生科研 Demo）
=====================================================================

⚠️ 合规声明（贯穿系统全部输出）：
1. 本系统是大学生科创决赛演示用的科研 Demo 原型，仅做技术思路验证，严禁商用、严禁临床使用。
2. 项目风险因子选取参考 SEER 癌症流行病学统计资料，病理AI相关研究范式参考 TCGA 公开数据集；
   本 Demo 模型训练使用 WDBC、BreakHis 海外公开数据集。
3. 本系统不采集任何真实病历数据；不提供任何疾病诊断、治疗方案、用药建议；
   所有输出仅供科研筛查参考，不能替代医生诊断，IHC（免疫组化）为确诊金标准。

页面结构（顶部导航切换，共三个页面）：
  欢迎页     ：项目介绍 + 两大功能模块 + 三个核心亮点（对齐参赛样板）
  普通用户端 ：大白话四板块问卷（基本信息/家族与既往史/月经与生育史/生活习惯与症状自查），
               提交后按勾选风险项【模拟】输出低/中/高风险三色卡片 + 就医建议
  医生专用端 ：上传 HE 切片 → MobileNetV2 特征提取展示（1280维特征/Top通道/激活热力图），
               输出激素依赖倾向的【模拟】辅助预判参考，禁止输出良性/恶性结论

技术栈：Python + Streamlit + pandas + scikit-learn + TensorFlow + Pillow
安装依赖：pip install -r requirements.txt
启动命令：streamlit run app.py
"""

import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ============================ 全局配置与常量 ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_DIR = os.path.join(BASE_DIR, "pathology_model")  # BreakHis 训练的病理模型目录

# 顶部导航选项（顺序与文案固定）
NAV_HOME = "🏠 欢迎页"
NAV_RISK = "👤 普通用户端"
NAV_PATH = "🔬 医生专用端"
NAV_OPTIONS = [NAV_HOME, NAV_RISK, NAV_PATH]

# 侧边栏【使用须知】：大赛要求文案，一字不改（expander 标题 + 正文）
SIDEBAR_NOTICE_TITLE = "⚠️ 使用须知（必读）"
SIDEBAR_NOTICE_BODY = (
    "• 本系统是大学生科研Demo原型，严禁商用、严禁临床使用；\n\n"
    "• 项目风险因子选取参考SEER癌症流行病学统计资料，病理AI相关研究范式参考TCGA公开数据集；"
    "本Demo模型训练使用WDBC、BreakHis海外公开数据集；\n\n"
    "• 本系统不采集任何真实病历数据；\n\n"
    "• 本系统不提供任何疾病诊断、治疗方案、用药建议；\n\n"
    "• 所有输出仅供科研筛查参考，不能替代医生诊断，IHC（免疫组化）为确诊金标准。"
)

# 欢迎页警示文字（严格沿用样板文案）
WELCOME_WARNING_BOLD = "⚠️ 本科学生科研探索原型 · 仅做技术思路验证 · 不用于临床诊断"
WELCOME_WARNING_SUB = "未经临床验证，不能替代医生判断与免疫组化（IHC）金标准"

# 普通用户端 结果免责声明（硬性要求）
DISCLAIMER_PAGE1 = (
    "🚫 **本结果仅科研原型筛查参考，不能替代医生诊断，IHC 为确诊金标准。** "
    "本页输出为 Demo 规则模拟结果，非 AI 模型输出、非诊断意见；"
    "训练数据集 WDBC、BreakHis 均为海外公开数据集，缺少国内人群样本，不可落地临床。"
)

# 医生专用端 顶部医疗免责声明
DISCLAIMER_PAGE2 = (
    "🚫 **医疗免责声明**：本页面为医生提供 HE 病理切片的 **AI 辅助参考概率与图像特征提取信息**；"
    "良性/恶性概率为**模型辅助参考输出，不构成确诊结论**；"
    "最终诊断请以医院免疫组化（IHC）病理报告与病理医生判读为准。"
)

# ---------- 普通用户端 · 大白话问卷数据（Demo 模拟计分，规则透明） ----------
# 单选题结构：(题干, key, [(选项文案, 模拟计分), ...])。分数仅用于 Demo 演示分级，无临床依据。

# 生活方式与基础慢病（男女共用，计 0~1 分/题）
Q_LIFE_COMMON = [
    ("吸烟情况", "s_smoke", [("从不吸烟", 0), ("既往吸烟，现已戒烟", 0), ("偶尔吸烟", 1), ("长期经常吸烟", 1)]),
    ("饮酒频率", "s_alcohol", [("从不饮酒", 0), ("每月少于1次", 0), ("每周1~2次", 1), ("每周3次及以上", 1)]),
    ("日常运动", "s_sport", [("几乎不运动", 1), ("每周1~2次", 0), ("每周3次及以上", 0)]),
    ("长期熬夜（经常凌晨1点后入睡）", "s_late", [("否", 0), ("是", 1)]),
    ("是否患有2型糖尿病", "s_diab", [("无", 0), ("有", 1)]),
    ("是否有高血压病史", "s_bp", [("无", 0), ("有", 1)]),
]

# 女性 · 板块2 家族肿瘤史与既往乳腺病史（一级亲属患病史计 2~4 分，其余 1~2 分）
F_Q_FAMILY = [
    ("一级亲属（母亲、姐妹、女儿）是否有乳腺癌、卵巢癌病史", "f_fam1",
     [("无", 0), ("1位亲属患病", 2), ("2位及以上亲属患病", 4)]),
    ("二级亲属（姑姑、姨妈、祖母）是否有乳腺癌病史", "f_fam2", [("无", 0), ("有", 1)]),
    ("是否确诊乳腺增生、乳腺结节", "f_nodule", [("无", 0), ("有，定期复查", 1), ("有，从未复查", 2)]),
    ("是否做过乳腺穿刺、乳腺活检、乳腺相关手术", "f_biopsy", [("无", 0), ("有", 1)]),
    ("是否有其他恶性肿瘤患病史", "f_cancer", [("无", 0), ("有", 2)]),
]

# 女性 · 板块3 生育 / 哺乳 / 外源激素暴露（月经初潮与绝经年龄为滑块，单独渲染）
F_Q_FERTILITY = [
    ("首次足月生育年龄", "f_birth", [("未生育", 2), ("＜25岁", 0), ("25~30岁", 0), ("＞30岁", 1)]),
    ("是否哺乳，累计哺乳时长", "f_nurse",
     [("未生育", 0), ("无哺乳", 1), ("0~6个月", 1), ("6个月~2年", 0), ("2年以上", 0)]),
    ("是否长期使用外源雌激素（包含短效避孕药、更年期激素替代等）", "f_estrogen",
     [("否", 0), ("是，2年以内", 1), ("是，超过2年", 2)]),
]

# 女性 · 板块4 自查症状（计 2 分/题）
F_Q_SYMPTOM = [
    ("近3个月乳房自查：是否摸到硬块、局部增厚", "f_lump",
     [("没有", 0), ("单侧乳房有", 2), ("双侧乳房都有", 2)]),
    ("是否存在乳头异常溢液（尤其是血性、咖啡色液体）", "f_discharge", [("没有", 0), ("有", 2)]),
    ("乳房皮肤改变：凹陷、橘皮样、红肿破溃", "f_skin", [("没有", 0), ("有", 2)]),
    ("腋窝、锁骨上方是否摸到无痛肿大硬块", "f_armpit", [("没有", 0), ("有", 2)]),
]

# 男性 · 板块2 家族肿瘤史与泌尿生殖既往病史
M_Q_FAMILY = [
    ("一级亲属（父亲、兄弟、儿子）是否有前列腺癌病史", "m_fam1",
     [("无", 0), ("1位亲属患病", 2), ("2位及以上亲属患病", 4)]),
    ("二级亲属（祖父、叔伯）是否有前列腺癌病史", "m_fam2", [("无", 0), ("有", 1)]),
    ("是否查出前列腺增生、前列腺结节", "m_nodule", [("无", 0), ("有，定期复查", 1), ("有，从未复查", 2)]),
    ("是否做过前列腺穿刺、前列腺相关手术", "m_biopsy", [("无", 0), ("有", 1)]),
    ("是否有其他恶性肿瘤患病史", "m_cancer", [("无", 0), ("有", 2)]),
]

# 男性 · 板块3 激素、泌尿相关病史
M_Q_HORMONE = [
    ("是否长期服用雄激素、睾酮类药物或相关保健品", "m_andro",
     [("否", 0), ("是，2年以内", 1), ("是，超过2年", 2)]),
    ("是否存在长期排尿不畅", "m_urine", [("否", 0), ("偶尔", 1), ("长期持续", 2)]),
    ("是否夜尿增多（每晚起夜≥2次）", "m_nocturia", [("没有", 0), ("有", 1)]),
    ("是否出现过血尿", "m_blood", [("没有", 0), ("有", 2)]),
]

# 男性 · 板块4 自查症状（计 1~2 分/题）
M_Q_SYMPTOM = [
    ("是否会阴部、下腹部反复坠胀不适", "m_pelvic", [("没有", 0), ("偶尔", 1), ("经常", 2)]),
    ("腹股沟、锁骨上方是否摸到无痛肿大硬块", "m_groin", [("没有", 0), ("有", 2)]),
]
# 就医建议（仅提示就医方向，不含任何诊断、治疗与用药建议）
RISK_ADVICE = {
    "低风险": ("🟢", st.success, "保持定期自检与常规体检习惯即可，无需过度紧张；如有疑虑可至正规医院咨询。"),
    "中风险": ("🟡", st.warning, "建议近期前往正规医院相应专科进一步咨询与检查，由医生评估后续安排。"),
    "高风险": ("🔴", st.error, "提示尽快前往正规医院完善相关检查；最终结论请以医生诊断及 IHC 检测结果为准。"),
}
DEPT_HINT = {"女": "乳腺外科 / 普外科 / 肿瘤科门诊", "男": "泌尿外科 / 肿瘤科门诊"}


# ============================ 资源加载（医生专用端） ============================

@st.cache_resource(show_spinner="正在加载 MobileNetV2 预训练模型（首次运行需联网下载约 14MB 权重）…")
def load_mobilenet():
    """加载 MobileNetV2 特征提取器（迁移学习：冻结 ImageNet 预训练骨干，仅做特征提取）。"""
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D

    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False  # 迁移学习：冻结预训练骨干；微调与激素依赖倾向建模属后续研究计划
    feat_model = tf.keras.Model(
        base.input, GlobalAveragePooling2D()(base.output), name="mobilenetv2_feature_extractor"
    )
    heatmap_model = tf.keras.Model(base.input, base.output, name="mobilenetv2_feature_map")
    return feat_model, heatmap_model


@st.cache_resource(show_spinner="正在加载病理模型（BreakHis 训练）…")
def load_pathology_model():
    """加载 BreakHis 训练的乳腺病理二分类模型与元信息（pathology_model 目录）。"""
    import tensorflow as tf

    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "breast_model.keras"))
    with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def gray_to_heat(gray: np.ndarray) -> np.ndarray:
    """灰度激活图 → 蓝-青-黄-红 伪彩色（仅用于特征激活强度的可视化示意）。"""
    stops = np.array([0, 64, 128, 192, 255], dtype=float)
    r = np.interp(gray, stops, [0, 0, 128, 255, 255])
    g = np.interp(gray, stops, [0, 200, 255, 200, 0])
    b = np.interp(gray, stops, [255, 255, 0, 0, 0])
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


# ============================ 欢迎页 ============================

def render_welcome_page() -> None:
    """欢迎页：居中主视觉 + 窄警示条 + 放大功能卡片（核心）+ 底部弱化亮点说明。"""
    # ---------- 顶部小字（居中） ----------
    st.markdown(
        "<p style='text-align:center;color:#8B7BB5;margin:0 0 0.4rem;font-size:0.95rem;'>"
        "—— 山东省大学生数智技术创新大赛 · 数智技术赋能智慧医疗方向 ——</p>",
        unsafe_allow_html=True,
    )
    # ---------- 主标题（放大、居中、紫色，视觉优先级最高）+ 副标题 ----------
    st.markdown(
        "<h1 style='text-align:center;color:#7C3AED;margin:0 0 0.3rem;font-size:3.4rem;"
        "font-weight:800;letter-spacing:0.06em;'>康险智析</h1>"
        "<p style='text-align:center;color:#6B6480;margin:0;font-size:1.15rem;'>"
        "激素依赖性肿瘤风险评估与病理AI探究原型</p>",
        unsafe_allow_html=True,
    )

    # ---------- 窄警示条（单行，压缩高度，文案保留） ----------
    st.warning(f"**{WELCOME_WARNING_BOLD}** ｜ {WELCOME_WARNING_SUB}")

    # ---------- 两大功能模块（页面核心：放大、加重紫色、醒目入口按钮） ----------
    st.markdown(
        "<h3 style='text-align:center;margin:0.8rem 0 0.4rem;'>🧩 两大功能模块</h3>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown(
                "<div style='background:#F3EEFB;border:2px solid #C4B5FD;border-radius:12px;"
                "padding:0.9rem 1.2rem;'>"
                "<span style='font-size:1.3rem;font-weight:800;color:#5B21B6;'>"
                "👤 普通用户端 · 风险早期预警</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown("按性别填写基础信息与症状，输出低/中/高风险分级提示与就医建议")
            st.button("进入评估 →", key="go_risk", type="primary", width="stretch",
                      on_click=lambda: st.session_state.update(nav=NAV_RISK))
    with col2:
        with st.container(border=True):
            st.markdown(
                "<div style='background:#F3EEFB;border:2px solid #C4B5FD;border-radius:12px;"
                "padding:0.9rem 1.2rem;'>"
                "<span style='font-size:1.3rem;font-weight:800;color:#5B21B6;'>"
                "🔬 医生专用端 · 病理AI辅助</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown("选择癌种、上传HE染色病理图，输出激素依赖倾向的辅助预判参考")
            st.button("进入辅助判读 →", key="go_path", type="primary", width="stretch",
                      on_click=lambda: st.session_state.update(nav=NAV_PATH))

    # ---------- 页面底部：弱化的补充说明（三个核心亮点，小字，不抢主视觉） ----------
    st.divider()
    st.caption(
        "**① 双端闭环**：普通人群预警与临床辅助参考相互独立、逻辑衔接 ｜ "
        "**② 轻量可解释**：规则透明、普通笔记本即可演示，不依赖额外服务 ｜ "
        "**③ 合规边界清晰**：结果仅作科普提示与辅助参考，不替代 IHC 与医生诊断"
    )
    st.caption(
        "📌 本系统是科研 Demo 原型：不接入真实临床数据、不输出诊断结论；普通用户端结果为规则模拟输出，"
        "医生专用端倾向预判为模拟演示数据；所有内容仅供科创演示与科研筛查参考。"
    )


# ============================ 普通用户端 · 风险早期预警 ============================

def _radio_questions(questions, detail):
    """渲染一组大白话单选题（默认空白未选，须用户手动作答），作答明细追加进 detail，返回未作答题数。"""
    unanswered = 0
    for label, key, options in questions:
        choice = st.radio(label, [o for o, _ in options], key=key, index=None)
        if choice is None:
            unanswered += 1
        else:
            detail.append((label, choice, dict(options)[choice]))
    return unanswered


def render_risk_page() -> None:
    """普通用户端：按性别适配的大白话问卷（默认全空白）→ 全部作答后【模拟】输出低/中/高风险三色卡片。"""
    st.markdown("## 👤 普通用户端 · 风险早期预警")
    st.info(
        "本问卷全部为**大白话单选题**，按性别自动适配（女性 · 乳腺方向 ／ 男性 · 前列腺方向）；"
        "**默认不预填任何答案**，需逐题手动作答；提交后由 **Demo 规则模拟**输出风险分级提示，"
        "不接入真实临床数据、不构成诊断。风险因子选取参考 SEER 癌症流行病学统计资料整理。"
    )

    # ---------- ① 基本信息（性别未选时，下方问卷不渲染，避免任何预填） ----------
    st.markdown("### ① 基本信息")
    gender = st.radio("你的性别", ["女", "男"], key="q_gender", index=None, horizontal=True)
    st.slider("年龄（岁）", 18, 100, 18, key="q_age")  # 覆盖 18~100 岁，仅作基础信息，不计分
    c1, c2, c3 = st.columns(3)
    with c1:
        height = st.number_input("身高（cm）", min_value=140, max_value=210, value=163, key="q_height")
    with c2:
        weight = st.number_input("体重（kg）", min_value=30, max_value=150, value=55, key="q_weight")
    with c3:
        bmi = weight / (height / 100.0) ** 2
        st.metric("BMI（自动计算）", f"{bmi:.1f}")
    st.caption("BMI = 体重 ÷ 身高²（m²），仅作基础信息展示，不计入风险分级。")

    if gender is None:
        st.info("👈 请先选择性别，系统将载入对应方向的问卷题目（女性 · 乳腺方向 / 男性 · 前列腺方向）。")
        return

    detail = []      # 作答明细：(题干, 选中项, 模拟计分)
    unanswered = 0   # 未作答题数

    # ---------- ② 家族肿瘤史与既往病史（按性别适配） ----------
    if gender == "女":
        st.markdown("### ② 家族肿瘤史与既往乳腺病史")
        unanswered += _radio_questions(F_Q_FAMILY, detail)
    else:
        st.markdown("### ② 家族肿瘤史与泌尿生殖既往病史")
        unanswered += _radio_questions(M_Q_FAMILY, detail)

    # ---------- ③ 激素暴露史（女性：月经/生育/激素；男性：激素/泌尿病史） ----------
    if gender == "女":
        st.markdown("### ③ 月经、生育、激素暴露史")
        menarche = st.selectbox(
            "月经初潮年龄（岁）", list(range(10, 19)), index=None,
            key="f_menarche", placeholder="请选择",
        )
        if menarche is None:
            unanswered += 1
        else:
            detail.append(("月经初潮年龄", f"{menarche} 岁", 1 if menarche <= 12 else 0))
        meno = st.radio("当前月经状态", ["尚未绝经", "已经绝经"], key="f_meno", index=None, horizontal=True)
        if meno == "已经绝经":  # 条件显示：仅"已经绝经"才填写绝经年龄
            meno_age = st.selectbox(
                "绝经年龄（岁）", list(range(40, 61)), index=None,
                key="f_meno_age", placeholder="请选择",
            )
            if meno_age is None:
                unanswered += 1
            else:
                detail.append(("绝经年龄", f"{meno_age} 岁", 1 if meno_age >= 55 else 0))
        unanswered += _radio_questions(F_Q_FERTILITY, detail)
    else:
        st.markdown("### ③ 激素、泌尿相关病史")
        unanswered += _radio_questions(M_Q_HORMONE, detail)

    # ---------- ④ 生活方式、自查症状与基础慢病（前 6 题男女共用，症状题按性别适配） ----------
    st.markdown("### ④ 生活方式、自查症状与基础慢病")
    unanswered += _radio_questions(Q_LIFE_COMMON, detail)
    unanswered += _radio_questions(F_Q_SYMPTOM if gender == "女" else M_Q_SYMPTOM, detail)

    # ---------- 提交：全部作答后按高危项计分【模拟】输出风险分级（三色卡片 + 就医建议） ----------
    if st.button("开始风险评估", type="primary", width="stretch"):
        if unanswered:
            st.warning(
                f"⚠️ 还有 **{unanswered}** 题未作答；本系统不预填任何答案，请逐题选择完成后再次提交。"
            )
        else:
            score = sum(s for _, _, s in detail)
            level = "低风险" if score <= 2 else ("中风险" if score <= 5 else "高风险")
            icon, box, advice = RISK_ADVICE[level]
            dept = DEPT_HINT[gender]
            st.markdown("### 📊 风险分级提示（Demo 模拟输出）")
            box(
                f"{icon} **{level}**\n\n"
                f"模拟得分：**{score} 分**（分级规则：0–2 分为低风险，3–5 分为中风险，≥6 分为高风险）\n\n"
                f"**🏥 就医建议**：{advice}\n\n"
                f"**可咨询科室方向（仅供参考）**：{dept}"
            )

            # 作答明细（透明可解释，对应核心亮点②"规则透明"）
            with st.expander("🔍 查看本次作答明细与计分规则（透明可解释）"):
                st.dataframe(
                    pd.DataFrame(detail, columns=["题目", "你的选择", "模拟计分"]),
                    width="stretch", hide_index=True,
                )
                st.caption(
                    "计分规则（Demo 演示设定，无临床依据，不构成任何医学建议）："
                    "一级亲属患病史 2~4 分、症状自查类 1~2 分、激素暴露与既往病史 1~2 分、"
                    "生活方式与基础慢病 0~1 分，月经初潮≤12 岁或绝经≥55 岁各 +1 分；"
                    "0–2 分 → 低风险，3–5 分 → 中风险，≥6 分 → 高风险。"
                )

            # 硬性免责声明（紧邻结果，反复强调）
            st.error(DISCLAIMER_PAGE1)
    else:
        st.info("👈 填写完成后点击「开始风险评估」查看模拟风险分级提示；全部题目均为单选，默认无预填，可随时修改。")


# ============================ 医生专用端 · 病理AI辅助 ============================

def render_pathology_page() -> None:
    """医生专用端：HE 切片 → BreakHis 训练模型推理（良性/恶性辅助参考概率）+ MobileNetV2 特征提取展示。"""
    st.markdown("## 🔬 医生专用端 · 病理AI辅助")
    st.error(DISCLAIMER_PAGE2)  # 医疗免责声明：页面顶部强制展示
    st.caption(
        "研究定位：乳腺癌良恶性二分类模型已接入（BreakHis 海外公开数据集训练，验证集准确率 81.0%）；"
        "前列腺方向与激素依赖倾向建模属**后续研究计划**。"
    )

    # ---------- 模型懒加载（特征提取骨干 + 病理推理模型） ----------
    try:
        feat_model, heatmap_model = load_mobilenet()
    except ImportError:
        st.error("未检测到 TensorFlow，请先安装依赖：pip install tensorflow")
        return
    except Exception as e:  # 权重下载失败等网络异常
        st.error(f"MobileNetV2 模型加载失败：{e}（首次运行需联网下载预训练权重，约 14MB）")
        return

    model_path = os.path.join(MODEL_DIR, "breast_model.keras")
    if not os.path.exists(model_path):
        st.error("未找到病理模型文件 pathology_model/breast_model.keras，请先运行 python train_pathology.py 训练生成。")
        return
    try:
        pathology_model, path_meta = load_pathology_model()
    except Exception as e:
        st.error(f"病理模型加载失败：{e}")
        return

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    # ---------- 选择癌种（乳腺癌可用；前列腺方向模型训练中，灰色置底暂不可选） ----------
    st.markdown("**选择癌种**")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(
            "<div style='background:#F3EEFB;border:2px solid #C4B5FD;border-radius:10px;"
            "padding:0.6rem 0.9rem;'><span style='font-size:1.02rem;font-weight:700;color:#5B21B6;'>"
            "✅ 乳腺癌（乳腺 HE 切片）</span><br>"
            "<span style='font-size:0.82rem;color:#6B6480;'>模型已接入 · BreakHis 数据集训练 · 当前可用</span></div>",
            unsafe_allow_html=True,
        )
    with col_c2:
        st.markdown(
            "<div style='background:#F5F5F6;border:1px dashed #C9C5D4;border-radius:10px;"
            "padding:0.6rem 0.9rem;'><span style='font-size:1.02rem;font-weight:700;color:#8A8598;'>"
            "🚧 前列腺癌（前列腺 HE 切片）</span><br>"
            "<span style='font-size:0.82rem;color:#8A8598;'>模型训练中 · 后续迭代接入 · 暂不可选</span></div>",
            unsafe_allow_html=True,
        )
    st.caption("当前癌种：乳腺癌（乳腺 HE 切片）；前列腺癌方向模型训练中，后续迭代接入。")

    # ---------- 输入：上传 / 样例 ----------
    uploaded = st.file_uploader(
        "上传 HE 染色病理切片图像（jpg / jpeg / png / bmp）", type=["jpg", "jpeg", "png", "bmp"]
    )
    sample_path = None
    if uploaded is None:
        sample_path = st.selectbox(
            "或载入内置样例图（程序合成的模拟 HE 图，非真实病理图像，仅演示流程用）",
            ["（暂不使用样例）", "sample_a.png", "sample_b.png"],
        )

    if uploaded is None and sample_path in (None, "（暂不使用样例）"):
        st.info("👈 请上传 HE 病理切片图像，或载入内置样例图，开始模型推理与特征提取演示。")
        return

    # ---------- 读取图像 ----------
    try:
        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            source_desc = "用户上传图像"
        else:
            img = Image.open(os.path.join(STATIC_DIR, sample_path)).convert("RGB")
            source_desc = f"内置样例（{sample_path}，程序合成模拟图，推理结果仅演示流程）"
    except Exception as e:
        st.error(f"图像读取失败，请更换文件后重试：{e}")
        return

    # ---------- 统一预处理（224×224 + MobileNetV2 归一化） ----------
    resized = img.resize((224, 224), Image.BILINEAR)
    batch = preprocess_input(np.asarray(resized, dtype=np.float32))[None, ...]  # 归一化到 [-1, 1]

    # ---------- ① 病理模型推理：良性 / 恶性辅助参考概率 ----------
    st.markdown("### ① 模型推理：良性 / 恶性辅助参考概率")
    prob_mal = float(pathology_model.predict(batch, verbose=0)[0][0])  # sigmoid 输出 = 恶性概率
    prob_mal = min(max(prob_mal, 0.0), 1.0)
    prob_ben = 1.0 - prob_mal

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.metric("良性概率（辅助参考）", f"{prob_ben * 100:.1f}%")
        st.progress(prob_ben, text=f"良性置信度 {prob_ben * 100:.1f}%")
    with col_p2:
        st.metric("恶性概率（辅助参考）", f"{prob_mal * 100:.1f}%")
        st.progress(prob_mal, text=f"恶性置信度 {prob_mal * 100:.1f}%")
    st.warning(
        "⚠️ 以上为 **AI 模型输出的辅助参考概率，不能替代 IHC 免疫组化金标准**，不构成确诊结论；"
        "请病理医生结合临床信息综合判读。"
    )
    with st.expander("ℹ️ 模型信息（透明可查）"):
        img_size = path_meta.get("img_size", [224, 224])
        st.markdown(
            f"- **任务**：{path_meta.get('task', '—')}\n"
            f"- **骨干**：{path_meta.get('backbone', '—')} ｜ **分类头**：{path_meta.get('head', '—')}\n"
            f"- **训练数据**：BreakHis 乳腺 HE 病理图（海外公开数据集），共 {path_meta.get('total_samples', '—')} 张\n"
            f"- **验证集准确率**：{path_meta.get('val_accuracy', 0) * 100:.1f}%\n"
            f"- **输入尺寸**：{img_size[0]}×{img_size[1]} ｜ **输出**：sigmoid 概率（1 = 恶性）\n"
            f"- **合规**：{path_meta.get('compliance', '—')}"
        )

    # ---------- ② 图像预处理信息 ----------
    st.markdown("### ② 图像预处理信息")
    col_ori, col_rz, col_meta = st.columns([2, 2, 3])
    with col_ori:
        st.image(img, caption="原图", width="stretch")
    with col_rz:
        st.image(resized, caption="MobileNetV2 输入（224×224）", width="stretch")
    with col_meta:
        st.metric("原始尺寸", f"{img.width} × {img.height}")
        st.metric("模型输入尺寸", "224 × 224 × 3（RGB）")
        st.caption(f"数据来源：{source_desc}。预处理：双线性插值缩放 + MobileNetV2 归一化。")

    # ---------- ③ 染色与纹理统计（基础图像特征） ----------
    st.markdown("### ③ 染色与纹理统计（基础图像特征）")
    arr = np.asarray(img, dtype=float)
    gray = np.asarray(img.convert("L"), dtype=float)
    hist_p = np.histogram(gray, bins=256, range=(0, 255))[0] / gray.size
    entropy = float(-(hist_p[hist_p > 0] * np.log2(hist_p[hist_p > 0])).sum())

    stat_df = pd.DataFrame(
        {
            "通道均值": [arr[..., c].mean() for c in range(3)],
            "通道标准差": [arr[..., c].std() for c in range(3)],
        },
        index=["R 通道", "G 通道", "B 通道"],
    )
    col_s1, col_s2 = st.columns([2, 3])
    with col_s1:
        st.dataframe(stat_df.round(1), width="stretch")
        st.metric("灰度纹理熵", f"{entropy:.2f}")
    with col_s2:
        st.caption(
            "说明：HE 染色切片中细胞核呈蓝紫色、细胞质呈红色，RGB 通道统计可粗略反映染色深浅分布；"
            "灰度纹理熵反映图像纹理复杂度。**以上均为染色质量与纹理的统计描述，仅供阅片参考，"
            "不具备任何诊断含义。**"
        )

    # ---------- ④ MobileNetV2 深度特征向量 ----------
    st.markdown("### ④ MobileNetV2 深度特征向量（ImageNet 预训练 · 迁移学习特征提取）")
    feats = feat_model.predict(batch, verbose=0)[0]           # 1280 维特征向量
    conv_map = heatmap_model.predict(batch, verbose=0)[0]     # 7×7×1280 卷积特征图

    nz_ratio = float((np.abs(feats) > 1e-6).mean())
    top_ch = int(np.argmax(np.abs(feats)))
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    col_f1.metric("特征维度", f"{feats.shape[0]} 维")
    col_f2.metric("L2 范数", f"{np.linalg.norm(feats):.1f}")
    col_f3.metric("特征均值 / 标准差", f"{feats.mean():.2f} / {feats.std():.2f}")
    col_f4.metric("最大激活通道", f"ch_{top_ch}（{feats[top_ch]:.2f}）")

    top20 = np.argsort(np.abs(feats))[::-1][:20]
    chart_df = pd.DataFrame({"激活强度": feats[top20]}, index=[f"ch_{i}" for i in top20])
    st.bar_chart(chart_df, horizontal=True)
    st.caption(
        f"上图为激活强度排名前 20 的特征通道（非零激活占比 {nz_ratio * 100:.1f}%）。"
        "特征向量由冻结的 MobileNetV2 预训练骨干提取（迁移学习推理），**仅为图像表征信息，"
        "不对应任何医学结论。**"
    )

    # ---------- ⑤ 卷积特征激活热力图 ----------
    st.markdown("### ⑤ 卷积特征激活热力图（模型关注区域示意）")
    heat = conv_map.mean(axis=-1)  # 通道平均激活 → 7×7 空间分布
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_resized = Image.fromarray((heat * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)
    overlay = Image.blend(img, Image.fromarray(gray_to_heat(np.asarray(heat_resized))), alpha=0.45)

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.image(resized, caption="预处理输入图", width="stretch")
    with col_h2:
        st.image(overlay, caption="特征激活热力图叠加", width="stretch")
    st.caption(
        "热力图为深度特征通道平均激活的空间分布示意：暖色区域表示该处图像内容引起的特征响应更强。"
        "**它仅表示模型特征响应强弱，不是病灶标注，更不是良性/恶性判断依据。**"
    )

    # ---------- 结果边界声明（硬性要求） ----------
    st.info(
        "✅ 以上内容均为**阅片辅助参考**：良性/恶性概率为 AI 模型辅助参考输出，**禁止作为确诊依据**；"
        "特征统计与热力图不具备诊断含义。"
        "最终诊断以医院免疫组化（IHC）病理报告与病理医生判读为准。"
    )

    with st.expander("❓ 为什么良性/恶性概率只能作为辅助参考？"):
        st.markdown(
            "1. **形态学局限**：概率基于 HE 形态学特征，与 IHC 等分子级金标准存在固有差异，概率 ≠ 确诊；\n"
            "2. **数据边界**：训练数据 BreakHis 为海外公开数据集，缺少国内人群样本，未经临床验证，不可落地临床；\n"
            "3. **域偏移风险**：不同医院的扫描仪、染色方案差异会显著影响模型实际表现；\n"
            "4. **合规边界**：大学生科研 Demo 原型严禁商用与临床使用；"
            "前列腺方向与激素依赖倾向建模属后续研究计划，所有结论以 IHC 金标准为对照，仅作学术研究。"
        )


# ============================ 顶部导航 / 侧边栏 / 主入口 ============================

def main() -> None:
    st.set_page_config(
        page_title="康险智析 · 科研Demo原型",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ---------- 侧边栏：品牌 + 使用须知（默认折叠，点击展开；文案一字不改） ----------
    with st.sidebar:
        st.markdown("## 🧬 康险智析")
        st.caption("`KANGXIAN ZHIXI · DEMO V0.1` ｜ 本科科研探索原型 · 非临床诊断")
        with st.expander(SIDEBAR_NOTICE_TITLE, expanded=False):
            st.markdown(SIDEBAR_NOTICE_BODY)
        st.divider()
        st.caption("山东第一医科大学 · 本科生创新团队\n\n大学生科创决赛演示版 · 仅用于技术思路验证")

    # ---------- 顶部导航栏（原生组件：水平单选） ----------
    st.radio("页面导航", NAV_OPTIONS, key="nav", horizontal=True, label_visibility="collapsed")
    st.divider()

    # ---------- 页面分发（仅三个页面，仅当前页执行） ----------
    page = st.session_state.get("nav", NAV_HOME)
    if page == NAV_RISK:
        render_risk_page()
    elif page == NAV_PATH:
        render_pathology_page()
    else:
        render_welcome_page()

    # ---------- 全局页脚 ----------
    st.divider()
    st.caption(
        "康险智析 · 大学生科创决赛科研 Demo 原型 ｜ 严禁商用 · 严禁临床使用 ｜ "
        "本系统不采集任何真实病历数据；不提供任何疾病诊断、治疗方案、用药建议 ｜ "
        "所有输出仅供科研筛查参考，不能替代医生诊断，IHC（免疫组化）为确诊金标准"
    )


if __name__ == "__main__":
    main()

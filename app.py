import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局配置（Streamlit Cloud 专属） =====================
st.set_page_config(
    page_title="Child Gastrointestinal Heat Retention Risk Predictor",
    page_icon="🏥",
    layout="wide"  # 宽屏适配75道题
)

# 文件名（与本地项目文件夹中的文件严格一致，Cloud无需路径，直接写文件名）
QUESTION_CSV_PATH = "Website Question Value RF cloud.csv"
MODEL_PATH = "RF_best_model.pkl"
FEATURE_MASK_PATH = "RF_feature_info.pkl"

# 风险等级配置（颜色+阈值+英文建议+可点击链接）
RISK_CONFIG = {
    "low": {
        "threshold": (0, 0.3),  # 0-30%（不含30%）
        "color": "#2ecc71",     # 绿色
        "advice": "The child has a low risk of Gastrointestinal Heat Retention Syndrome. Please continue to maintain the current living habits."
    },
    "medium": {
        "threshold": (0.3, 0.6),# 30%-60%（含30%，不含60%）
        "color": "#f39c12",     # 黄色
        "advice": "The child has a certain risk of Gastrointestinal Heat Retention Syndrome. Please pay attention to whether the child's living habits such as diet, sleep, and exercise are regular, and correct them if necessary."
    },
    "high": {
        "threshold": (0.6, 1.01),# 60%及以上（含60%）
        "color": "#e74c3c",      # 红色
        "advice": """The child has a high risk of Gastrointestinal Heat Retention Syndrome. Please correct the child's living habits from the following aspects:
1. Ensure regular meal times/portions and a balanced diet with meat and vegetable matching. Reduce high-calorie foods (high-sugar, high-fat, fried/grilled). See the <a href="https://odphp.health.gov/our-work/nutrition-physical-activity/dietary-guidelines" target="_blank">2025-2030 Dietary Guidelines for Americans</a>.
2. Ensure adequate and regular sleep. See the <a href="https://www.ncbi.nlm.nih.gov/books/NBK20359/" target="_blank">NIH Sleep Health Course</a>.
3. Ensure adequate regular physical activity (per WHO recommendations). See <a href="https://www.who.int/zh/news-room/fact-sheets/detail/physical-activity" target="_blank">WHO Physical Activity Fact Sheet</a>."""
    }
}

# 🌟 核心：训练时最终筛选的158个有效二值特征的精准映射表（必须和训练时的特征顺序、来源完全一致！）
# 格式：[(题目编号, 该题有效选项的原始赋值, 特征含义), ...]，共158个元素
# 示例：第1个特征是Q1选1（赋值1）对应的哑变量，第2个是Q1选2（赋值2）对应的哑变量，第3个是Q5选1（赋值1）对应的哑变量...
TRAIN_VALID_158_FEATURES = [
    (1, 1, "Q1_Option1_哑变量"),   # 第1个特征：Q1选赋值1的选项，哑变量为1，否则0
    (1, 2, "Q1_Option2_哑变量"),   # 第2个特征：Q1选赋值2的选项，哑变量为1，否则0
    (5, 1, "Q5_Option1_哑变量"),   # 第3个特征：Q5选赋值1的选项，哑变量为1，否则0
    (5, 2, "Q5_Option2_哑变量"),   # 第4个特征：Q5选赋值2的选项，哑变量为1，否则0
    # ... 继续补充，直到满158个元素，顺序必须和训练时一致！
    # 注意：数值题（若有）直接按原始赋值作为特征，无需拆哑变量，直接写(题号, 0, "数值题原始特征")即可
]
# 验证：列表长度必须严格等于158
assert len(TRAIN_VALID_158_FEATURES) == 158, "映射表长度必须为158，和训练时特征数量一致！"

# ===================== 加载核心资源（Cloud 缓存优化，仅加载1次） =====================
@st.cache_resource(ttl=None)  # 永久缓存，Cloud重启才会重新加载
def load_core_resources():
    """加载CSV题目、RF模型、特征掩码，适配Streamlit Cloud，增加编码兜底"""
    # 1. 加载CSV题目文件（核心修改：GBK编码+多编码兜底，解决解码失败）
    df_questions = None
    # 定义兼容的编码顺序，优先GBK（解决0xa1字节问题），兜底utf-8-sig
    encodings = ['gbk', 'gb18030', 'utf-8-sig']
    for enc in encodings:
        try:
            df_questions = pd.read_csv(QUESTION_CSV_PATH, encoding=enc)
            st.success(f"✅ CSV file loaded successfully with encoding: {enc}")
            break
        except Exception as e:
            continue
    
    # 所有编码都失败时报错
    if df_questions is None:
        st.error(f"Failed to load question CSV: All encodings ({encodings}) decode failed! Check file encoding (suggest GBK/UTF-8).")
        st.stop()

    # 验证必要列，缺失则报错
    required_cols = ["Question", "Option1", "Value1", "Option2", "Value2", "Option3", "Value3", "Option4", "Value4"]
    for col in required_cols:
        if col not in df_questions.columns:
            st.error(f"CSV file missing required column: {col}! Check your csv format.")
            st.stop()
    # 过滤空题，保留有效题目
    df_questions = df_questions.dropna(subset=["Question"]).reset_index(drop=True)
    if len(df_questions) != 75:
        st.warning(f"⚠️ {len(df_questions)} questions found in CSV (expected 75). Please confirm!")

    # 2. 加载RF模型
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅ RF model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load RF model: {str(e)}. Check model file name/path!")
        st.stop()

    # 3. 加载特征掩码（无则使用所有特征，优化异常提示）
    best_mask = None
    try:
        feat_info = joblib.load(FEATURE_MASK_PATH)
        best_mask = np.array(feat_info["best_feature_mask"])
        st.success("✅ Feature mask loaded successfully!")
    except FileNotFoundError:
        st.warning(f"⚠️ Feature mask file {FEATURE_MASK_PATH} not found. Using all features!")
    except Exception as e:
        st.warning(f"⚠️ Failed to parse feature mask: {str(e)}. Using all features!")

    return df_questions, model, best_mask

# 执行加载（页面启动时仅1次）
df_questions, rf_model, best_mask = load_core_resources()

# ===================== 核心函数：生成158个有效特征（支持无效选项赋值为0） =====================
def generate_exact_158_feats(user_answers):
    """
    根据用户答题结果，按训练时的映射表精准生成158个有效二值特征
    - 选中有效选项（训练时参与特征筛选的选项）→ 对应哑变量=1，其他=0
    - 选中无效选项（补充的选项，Value为空）→ 该题所有对应哑变量=0
    user_answers：{题目编号: 原始赋值/None}，None表示选中无效选项
    return：(1,158)的数组，严格匹配模型训练时的输入维度和顺序
    """
    final_158_feats = []
    for (q_num, target_val, _) in TRAIN_VALID_158_FEATURES:
        # 获取用户该题的实际结果：有效选项返回赋值（int），无效选项返回None
        user_val = user_answers.get(q_num, None)
        # 赋值规则：
        # 1. 选中有效选项且赋值匹配→1；2. 选中有效选项但赋值不匹配→0；3. 选中无效选项→0
        if user_val is not None and user_val == target_val:
            final_158_feats.append(1)
        else:
            final_158_feats.append(0)
    # 转换为模型要求的输入形状：(1, 158)
    return np.array(final_158_feats).reshape(1, -1)

# ===================== 渲染75道题问卷（支持所有选项填写，区分有效/无效选项） =====================
def render_questionnaire(df_questions):
    """渲染问卷，返回用户答题结果：{题目编号: 有效选项赋值/None}，None表示选中无效选项"""
    st.subheader("🏥 Gastrointestinal Heat Retention Risk Assessment Questionnaire", divider="blue")
    st.caption("Please answer all questions (Single choice / Required for each)")
    user_answers = {}  # 存储{题号: 有效选项赋值/None}

    # 分批次显示：5道/批，避免页面过长（Cloud端渲染更友好）
    batch_size = 5
    total_batches = (len(df_questions) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min((batch_idx+1)*batch_size, len(df_questions))
        batch_q = df_questions.iloc[start:end]

        # 批次标题
        st.markdown(f"### Batch {batch_idx+1}/{total_batches} (Questions {start+1}-{end})")
        # 渲染单题
        for idx, row in batch_q.iterrows():
            q_num = idx + 1
            q_text = f"Q{q_num}: {row['Question']}"
            
            # 🌟 核心修改1：提取所有选项（含有效/无效），用户均可选择
            all_options = []  # 所有可选选项（展示用）
            opt_val_map = {}  # 选项文字→有效赋值/None（None=无效选项）
            for opt_i in [1,2,3,4]:
                opt_col = f"Option{opt_i}"
                val_col = f"Value{opt_i}"
                if pd.notna(row[opt_col]):  # 只要选项有文字，就加入可选列表
                    opt_text = row[opt_col]
                    all_options.append(opt_text)
                    # 有效选项（Value非空）→ 存储赋值；无效选项（Value为空）→ 存储None
                    if pd.notna(row[val_col]):
                        opt_val_map[opt_text] = int(row[val_col])
                    else:
                        opt_val_map[opt_text] = None
            
            # 🌟 核心修改2：单选框支持所有选项，用户均可选择
            selected_opt = st.radio(
                label=q_text,
                options=all_options,  # 所有选项均可选
                key=f"q_{q_num}",
                index=None  # 初始无选择，强制用户点击
            )
            # 存储结果：有效选项存赋值，无效选项存None
            if selected_opt is not None:
                user_answers[q_num] = opt_val_map[selected_opt]
        st.divider()

    # 验证答题完整性
    if len(user_answers) != len(df_questions):
        st.error(f"⚠️ Please answer all {len(df_questions)} questions! Answered {len(user_answers)} so far.")
        return None

    return user_answers

# ===================== 风险预测与结果展示（Cloud 可视化优化） =====================
def show_prediction_result(input_feat):
    """预测概率，按等级展示彩色结果+建议"""
    st.subheader("📊 Risk Prediction Result", divider="red")
    # 模型预测（兼容Pipeline/纯模型，增加异常捕获）
    try:
        if hasattr(rf_model, 'named_steps'):
            # Pipeline模型：先标准化再预测
            scaler = rf_model.named_steps.get('scaler')
            if scaler:
                risk_prob = rf_model.predict_proba(scaler.transform(input_feat))[0,1]
            else:
                risk_prob = rf_model.predict_proba(input_feat)[0,1]
        else:
            # 纯模型：直接预测
            risk_prob = rf_model.predict_proba(input_feat)[0,1]
        risk_pct = round(risk_prob * 100, 2)  # 百分比保留2位小数
        st.success(f"✅ Prediction completed! Risk probability: {risk_pct}%")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}. Check feature order/model compatibility!")
        st.error(f"Input feature shape: {input_feat.shape}, Model expected features: {rf_model.n_features_in_ if hasattr(rf_model, 'n_features_in_') else 'Unknown'}")
        return

    # 匹配风险等级
    if risk_prob < 0.3:
        level = "low"
    elif risk_prob < 0.6:
        level = "medium"
    else:
        level = "high"
    cfg = RISK_CONFIG[level]

    # 彩色大字体显示概率（Cloud页面更醒目，优化样式）
    st.markdown(
        f"""
        <div style="text-align: center; padding: 30px; border-radius: 15px; background: #f0f2f6; margin: 20px 0;">
            <h2 style="color: {cfg['color']}; margin: 0; font-size: 3em; font-weight: bold;">{risk_pct}%</h2>
            <h3 style="color: {cfg['color']}; margin: 10px 0 0 0; font-size: 1.8em;">{level.upper()} RISK</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 展示临床建议（高风险含可点击链接，优化样式）
    st.markdown("### 🩺 Clinical Advice")
    st.markdown(
        f"""
        <div style="padding: 20px; border-left: 8px solid {cfg['color']}; background: #f0f2f6; border-radius: 10px; margin: 10px 0;">
            <p style="font-size: 1.15em; line-height: 1.8;">{cfg['advice']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 审稿专用提示（修正拼写错误 Disclaime → Disclaimer）
    st.caption("⚠️ Disclaimer: This result is for clinical review only and does not replace professional medical diagnosis.")

# ===================== 页面主逻辑（Cloud 交互优化） =====================
def main():
    st.title("🏥 RF Model for Childhood Gastrointestinal Heat Retention Risk Prediction")
    st.caption("Deployed on Streamlit Cloud | For Expert Review")
    # 1. 渲染问卷，获取用户答题结果（{题号: 有效赋值/None}）
    user_answers = render_questionnaire(df_questions)
    # 2. 提交按钮（占满宽度，Cloud端更易点击）
    submit_btn = st.button("📤 Submit Answers & Predict Risk", type="primary", use_container_width=True)
    if submit_btn:
        if user_answers is not None:
            with st.spinner("🔍 Generating features & Predicting risk... Please wait a moment..."):
                # 3. 核心：生成158个有效特征（无效选项对应特征=0）
                input_feat = generate_exact_158_feats(user_answers)
                st.success(f"✅ Feature generation completed! Input shape: {input_feat.shape}")
                # 4. 预测
                show_prediction_result(input_feat)
        else:
            st.warning("⚠️ Please complete all questions before submission!")

if __name__ == "__main__":
    main()

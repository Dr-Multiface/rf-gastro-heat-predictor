import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局配置（Streamlit Cloud 专属） =====================
st.set_page_config(
    page_title="Child GI Heat Retention Risk Predictor",
    page_icon="🏥",
    layout="wide"  # 宽屏适配75道题
)

# 文件名（与本地项目文件夹中的文件严格一致，Cloud无需路径，直接写文件名）
QUESTION_CSV_PATH = "Website Question Value RF cloud.csv"  # 替换为CSV文件
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

# ===================== 加载核心资源（Cloud 缓存优化，仅加载1次） =====================
@st.cache_resource(ttl=None)  # 永久缓存，Cloud重启才会重新加载
def load_core_resources():
    """加载CSV题目、RF模型、特征掩码，适配Streamlit Cloud"""
    # 1. 加载CSV题目文件（核心修改：替代Excel）
    try:
        df_questions = pd.read_csv(QUESTION_CSV_PATH, encoding='utf-8-sig')
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
    except Exception as e:
        st.error(f"Failed to load question CSV: {str(e)}. Check file name/path!")
        st.stop()

    # 2. 加载RF模型
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load RF model: {str(e)}. Check model file name!")
        st.stop()

    # 3. 加载特征掩码（无则使用所有特征）
    try:
        feat_info = joblib.load(FEATURE_MASK_PATH)
        best_mask = np.array(feat_info["best_feature_mask"])
    except Exception as e:
        st.warning(f"⚠️ Failed to load feature mask: {str(e)}. Using all features!")
        best_mask = None

    return df_questions, model, best_mask

# 执行加载（页面启动时仅1次）
df_questions, rf_model, best_mask = load_core_resources()

# ===================== 渲染75道题问卷（单选+必填+分批次） =====================
def render_questionnaire(df_questions):
    """渲染问卷，返回模型输入的特征数组"""
    st.subheader("🏥 Gastrointestinal Heat Retention Risk Assessment Questionnaire", divider="blue")
    st.caption("Please answer all questions (Single choice / Required for each)")
    user_answers = {}  # 存储{题号: 选项赋值}

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
            # 提取有效选项和赋值（过滤空值）
            options, values = [], []
            for opt_i in [1,2,3,4]:
                opt_col = f"Option{opt_i}"
                val_col = f"Value{opt_i}"
                if pd.notna(row[opt_col]) and pd.notna(row[val_col]):
                    options.append(row[opt_col])
                    values.append(int(row[val_col]))  # 赋值转整数，与模型一致
            # 单选框（必填，key唯一，Cloud缓存兼容）
            selected_opt = st.radio(
                label=q_text,
                options=options,
                key=f"q_{q_num}",
                required=True,
                index=None  # 初始无选择，强制用户点击
            )
            # 存储赋值
            if selected_opt:
                user_answers[q_num] = values[options.index(selected_opt)]
        st.divider()

    # 验证答题完整性
    if len(user_answers) != len(df_questions):
        st.error(f"⚠️ Please answer all {len(df_questions)} questions! Answered {len(user_answers)} so far.")
        return None

    # 转换为模型输入格式（按题号排序，特征顺序与训练一致）
    input_feat = np.array([user_answers[q] for q in sorted(user_answers.keys())]).reshape(1, -1)
    # 应用特征掩码
    if best_mask is not None:
        input_feat = input_feat[:, best_mask]
    return input_feat

# ===================== 风险预测与结果展示（Cloud 可视化优化） =====================
def show_prediction_result(input_feat):
    """预测概率，按等级展示彩色结果+建议"""
    st.subheader("📊 Risk Prediction Result", divider="red")
    # 模型预测（兼容Pipeline/纯模型）
    try:
        if hasattr(rf_model, 'named_steps'):
            # Pipeline模型：先标准化再预测
            risk_prob = rf_model.predict_proba(rf_model.named_steps['scaler'].transform(input_feat))[0,1]
        else:
            # 纯模型：直接预测
            risk_prob = rf_model.predict_proba(input_feat)[0,1]
        risk_pct = round(risk_prob * 100, 2)  # 百分比保留2位小数
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}. Check feature order/model compatibility!")
        return

    # 匹配风险等级
    if risk_prob < 0.3:
        level = "low"
    elif risk_prob < 0.6:
        level = "medium"
    else:
        level = "high"
    cfg = RISK_CONFIG[level]

    # 彩色大字体显示概率（Cloud页面更醒目）
    st.markdown(
        f"""
        <div style="text-align: center; padding: 25px; border-radius: 12px; background: #f0f2f6;">
            <h2 style="color: {cfg['color']}; margin: 0;">{risk_pct}%</h2>
            <h3 style="color: {cfg['color']}; margin: 5px 0 0 0;">{level.upper()} RISK</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 展示临床建议（高风险含可点击链接）
    st.markdown("### 🩺 Clinical Advice")
    st.markdown(
        f"""
        <div style="padding: 18px; border-left: 6px solid {cfg['color']}; background: #f0f2f6; border-radius: 8px;">
            <p style="font-size: 1.1em; line-height: 1.6;">{cfg['advice']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 审稿专用提示
    st.caption("⚠️ Disclaime: This result is for clinical review only and does not replace professional medical diagnosis.")

# ===================== 页面主逻辑（Cloud 交互优化） =====================
def main():
    st.title("🏥 RF Model for Childhood Gastrointestinal Heat Retention Risk Prediction")
    st.caption("Deployed on Streamlit Cloud | For Expert Review")
    # 渲染问卷
    input_features = render_questionnaire(df_questions)
    # 提交按钮（占满宽度，Cloud端更易点击）
    if st.button("📤 Submit Answers & Predict Risk", type="primary", use_container_width=True):
        if input_features is not None:
            with st.spinner("🔍 Predicting risk... Please wait a moment."):
                show_prediction_result(input_features)

if __name__ == "__main__":
    main()
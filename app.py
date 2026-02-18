import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局配置（Streamlit Cloud 专属） =====================
st.set_page_config(
    page_title="Child Gastrointestinal Heat Retention Syndrome Risk Predictor",
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
    (1, 1, "Q1_Option1_Feature1"),   # 第1个特征：Q1选赋值1的选项，哑变量为1，否则0
    (2, 1, "Q2_Option1_Feature2"),   # 第2个特征：Q2选赋值1的选项，哑变量为1，否则0
    (3, 1, "Q3_Option1_Feature3"),   # 第3个特征：Q3选赋值1的选项，哑变量为1，否则0
    (4, 1, "Q4_Option1_Feature4"),   # 第4个特征：Q5选2的选项，哑变量为1，否则0
    (5, 1, "Q5_Option1_Feature5"),
    (6, 1, "Q6_Option1_Feature6"),
    (7, 1, "Q7_Option1_Feature7"),
    (7, 2, "Q7_Option2_Feature8"),
    (8, 1, "Q8_Option1_Feature9"),
    (9, 1, "Q9_Option1_Feature10"),
    (9, 2, "Q9_Option2_Feature11"),
    (9, 3, "Q9_Option3_Feature12"),
    (10, 1, "Q10_Option1_Feature13"),
    (10, 2, "Q10_Option2_Feature14"),
    (10, 3, "Q10_Option3_Feature15"),
    (11, 1, "Q11_Option1_Feature16"),
    (11, 2, "Q11_Option2_Feature17"),
    (11, 3, "Q11_Option3_Feature18"),
    (12, 1, "Q12_Option1_Feature19"),
    (13, 1, "Q13_Option1_Feature20"),
    (13, 2, "Q13_Option2_Feature21"),
    (14, 1, "Q14_Option1_Feature22"),
    (15, 1, "Q15_Option1_Feature23"),
    (15, 3, "Q15_Option3_Feature24"),
    (15, 4, "Q15_Option4_Feature25"),
    (16, 1, "Q16_Option1_Feature26"),
    (16, 2, "Q16_Option2_Feature27"),
    (16, 3, "Q16_Option3_Feature28"),
    (17, 1, "Q17_Option1_Feature29"),
    (17, 2, "Q17_Option2_Feature30"),
    (17, 3, "Q17_Option3_Feature31"),
    (18, 1, "Q18_Option1_Feature32"),
    (18, 2, "Q18_Option2_Feature33"),
    (19, 1, "Q19_Option1_Feature34"),
    (19, 2, "Q19_Option2_Feature35"),
    (20, 1, "Q20_Option1_Feature36"),
    (20, 2, "Q20_Option2_Feature37"),
    (21, 1, "Q21_Option1_Feature38"),
    (21, 2, "Q21_Option2_Feature39"),
    (21, 3, "Q21_Option3_Feature40"),
    (22, 1, "Q22_Option1_Feature41"),
    (22, 2, "Q22_Option2_Feature42"),
    (22, 3, "Q22_Option3_Feature43"),
    (23, 1, "Q23_Option1_Feature44"),
    (23, 3, "Q23_Option3_Feature45"),
    (23, 4, "Q23_Option4_Feature46"),
    (24, 1, "Q24_Option1_Feature47"),
    (24, 3, "Q24_Option3_Feature48"),
    (25, 1, "Q25_Option1_Feature49"),
    (26, 1, "Q26_Option1_Feature50"),
    (26, 2, "Q26_Option2_Feature51"),
    (27, 1, "Q27_Option1_Feature52"),
    (27, 3, "Q27_Option3_Feature53"),
    (27, 4, "Q27_Option4_Feature54"),
    (28, 1, "Q28_Option1_Feature55"),
    (28, 2, "Q28_Option2_Feature56"),
    (28, 3, "Q28_Option3_Feature57"),
    (29, 1, "Q29_Option1_Feature58"),
    (29, 2, "Q29_Option2_Feature59"),
    (29, 3, "Q29_Option3_Feature60"),
    (30, 1, "Q30_Option1_Feature61"),
    (30, 2, "Q30_Option2_Feature62"),
    (30, 3, "Q30_Option3_Feature63"),
    (31, 1, "Q31_Option1_Feature64"),
    (31, 2, "Q31_Option2_Feature65"),
    (32, 1, "Q32_Option1_Feature66"),
    (33, 1, "Q33_Option1_Feature67"),
    (33, 2, "Q33_Option2_Feature68"),
    (33, 3, "Q33_Option3_Feature69"),
    (34, 1, "Q34_Option1_Feature70"),
    (34, 2, "Q34_Option2_Feature71"),
    (35, 1, "Q35_Option1_Feature72"),
    (35, 2, "Q35_Option2_Feature73"),
    (36, 1, "Q36_Option1_Feature74"),
    (36, 2, "Q36_Option2_Feature75"),
    (37, 1, "Q37_Option1_Feature76"),
    (37, 2, "Q37_Option2_Feature77"),
    (37, 3, "Q37_Option3_Feature78"),
    (38, 1, "Q38_Option1_Feature79"),
    (38, 2, "Q38_Option2_Feature80"),
    (39, 1, "Q39_Option1_Feature81"),
    (40, 1, "Q40_Option1_Feature82"),
    (40, 2, "Q40_Option2_Feature83"),
    (41, 1, "Q41_Option1_Feature84"),
    (41, 2, "Q41_Option2_Feature85"),
    (42, 1, "Q42_Option1_Feature86"),
    (42, 2, "Q42_Option2_Feature87"),
    (43, 1, "Q43_Option1_Feature88"),
    (43, 2, "Q43_Option2_Feature89"),
    (43, 3, "Q43_Option3_Feature90"),
    (44, 1, "Q43_Option1_Feature91"),
    (44, 2, "Q43_Option2_Feature92"),
    (44, 3, "Q43_Option3_Feature93"),
    (45, 1, "Q45_Option1_Feature94"),
    (45, 3, "Q45_Option3_Feature95"),
    (45, 4, "Q45_Option4_Feature96"),
    (46, 1, "Q46_Option1_Feature97"),
    (46, 3, "Q46_Option3_Feature98"),
    (47, 1, "Q47_Option1_Feature99"),
    (47, 2, "Q47_Option2_Feature100"),
    (47, 3, "Q47_Option3_Feature101"),
    (48, 1, "Q48_Option1_Feature102"),
    (48, 3, "Q48_Option3_Feature103"),
    (49, 1, "Q49_Option1_Feature104"),
    (49, 2, "Q49_Option2_Feature105"),
    (50, 1, "Q50_Option1_Feature106"),
    (50, 2, "Q50_Option2_Feature107"),
    (51, 1, "Q51_Option1_Feature108"),
    (51, 2, "Q51_Option2_Feature109"),
    (52, 1, "Q52_Option1_Feature110"),
    (52, 2, "Q52_Option2_Feature111"),
    (53, 1, "Q53_Option1_Feature112"),
    (53, 2, "Q53_Option2_Feature113"),
    (54, 1, "Q54_Option1_Feature114"),
    (55, 1, "Q55_Option1_Feature115"),
    (55, 2, "Q55_Option2_Feature116"),
    (56, 1, "Q56_Option1_Feature117"),
    (57, 1, "Q57_Option1_Feature118"),
    (58, 1, "Q58_Option1_Feature119"),
    (58, 2, "Q58_Option2_Feature120"),
    (58, 3, "Q58_Option3_Feature121"),
    (59, 1, "Q59_Option1_Feature122"),
    (59, 3, "Q59_Option3_Feature123"),
    (59, 4, "Q59_Option4_Feature124"),
    (60, 1, "Q60_Option1_Feature125"),
    (60, 3, "Q60_Option3_Feature126"),
    (60, 4, "Q60_Option4_Feature127"),
    (61, 1, "Q61_Option1_Feature128"),
    (61, 2, "Q61_Option2_Feature129"),
    (61, 3, "Q61_Option3_Feature130"),
    (62, 1, "Q62_Option1_Feature131"),
    (62, 3, "Q62_Option3_Feature132"),
    (62, 4, "Q62_Option4_Feature133"),
    (63, 1, "Q63_Option1_Feature134"),
    (63, 3, "Q63_Option3_Feature135"),
    (64, 1, "Q64_Option1_Feature136"),
    (64, 3, "Q64_Option3_Feature137"),
    (64, 4, "Q64_Option4_Feature138"),
    (65, 1, "Q65_Option1_Feature139"),
    (65, 3, "Q65_Option3_Feature140"),
    (65, 4, "Q65_Option4_Feature141"),
    (66, 1, "Q66_Option1_Feature142"),
    (67, 1, "Q67_Option1_Feature143"),
    (67, 2, "Q67_Option2_Feature144"),
    (68, 1, "Q68_Option1_Feature145"),
    (68, 2, "Q68_Option2_Feature146"),
    (68, 3, "Q68_Option3_Feature147"),
    (69, 1, "Q69_Option1_Feature148"),
    (69, 2, "Q69_Option2_Feature149"),
    (69, 3, "Q69_Option3_Feature150"),
    (70, 1, "Q70_Option1_Feature151"),
    (71, 1, "Q71_Option1_Feature152"),
    (72, 1, "Q72_Option1_Feature153"),
    (73, 1, "Q73_Option1_Feature154"),
    (73, 2, "Q73_Option2_Feature155"),
    (74, 2, "Q74_Option2_Feature156"),
    (75, 1, "Q75_Option1_Feature157"),
    (75, 2, "Q75_Option2_Feature158"),
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
    st.subheader("🏥 Gastrointestinal Heat Retention Syndrome Risk Assessment Questionnaire", divider="blue")
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
            
            # 🌟 核心修改：为9~58题新增"Always"选项
            all_options = []  # 所有可选选项（展示用）
            opt_val_map = {}  # 选项文字→有效赋值/None（None=无效选项）
            
            # 1. 先加载原有选项（Option1-Option4）
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
            
            # 2. 为9~58题新增"Always"选项（标记为无效选项，赋值None）
            if 9 <= q_num <= 58:
                always_text = "Always"
                all_options.append(always_text)
                opt_val_map[always_text] = None  # Always为无效选项，无赋值
            
            # 🌟 渲染单选框，支持所有选项（含新增的Always）
            selected_opt = st.radio(
                label=q_text,
                options=all_options,  # 所有选项均可选（含Always）
                key=f"q_{q_num}",
                index=None  # 初始无选择，强制用户点击
            )
            # 存储结果：有效选项存赋值，无效选项（含Always）存None
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
    st.title("🏥 RF Model for Childhood Gastrointestinal Heat Retention Syndrome Risk Prediction")
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




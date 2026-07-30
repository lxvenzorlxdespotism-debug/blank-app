import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="⚽ 冷门预测", layout="centered")
st.title("⚽ 冷门价值投注预测")
st.markdown("第一步：上传模型文件（仅首次需要）")

# 文件上传组件
uploaded_file = st.file_uploader("请上传扣子发给你的 .pkl 文件", type=['pkl'])

if uploaded_file is not None:
    # 加载模型
    model_data = pickle.load(uploaded_file)
    lgb_model = model_data['lgb']
    xgb_model = model_data['xgb']
    cb_model = model_data['cb']
    le_league = model_data['league_encoder']
    feature_cols = model_data['feature_cols']
    
    st.success("✅ 模型加载成功！")
    
    # ---------- 预测界面 ----------
    st.header("📊 输入比赛数据")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        psch = st.number_input("Pinnacle 主胜", value=2.10, step=0.01)
        pscd = st.number_input("Pinnacle 平局", value=3.60, step=0.01)
        psca = st.number_input("Pinnacle 客胜", value=3.80, step=0.01)
    with col2:
        b365ch = st.number_input("Bet365 主胜", value=2.05, step=0.01)
        b365cd = st.number_input("Bet365 平局", value=3.50, step=0.01)
        b365ca = st.number_input("Bet365 客胜", value=3.70, step=0.01)
    with col3:
        avgch = st.number_input("平均主胜", value=2.08, step=0.01)
        avgcd = st.number_input("平均平局", value=3.55, step=0.01)
        avgca = st.number_input("平均客胜", value=3.75, step=0.01)
    
    league = st.selectbox("联赛", options=list(le_league.classes_))
    
    if st.button("🔮 开始预测", use_container_width=True):
        # 构造特征
        payout = 1 / (1/psch + 1/pscd + 1/psca)
        div_h = abs(psch - b365ch)
        div_d = abs(pscd - b365cd)
        div_a = abs(psca - b365ca)
        ratio_ha = avgch / avgca
        ratio_hd = avgch / avgcd
        
        try:
            league_enc = le_league.transform([league])[0]
        except:
            league_enc = 0
        
        base_features = [
            avgch, avgcd, avgca, payout,
            div_h, div_d, div_a,
            ratio_ha, ratio_hd,
            0.5, 1.5, 0.5, 1.5,
            0.33, 0.33, 0.33,
            league_enc
        ]
        # 补全到模型需要的维度
        while len(base_features) < len(feature_cols):
            base_features.append(0.0)
        
        features = np.array(base_features).reshape(1, -1)
        
        # 三模型预测
        p1 = lgb_model.predict_proba(features)[0]
        p2 = xgb_model.predict_proba(features)[0]
        p3 = cb_model.predict_proba(features)[0]
        proba = (p1 + p2 + p3) / 3
        
        # 竞彩模拟赔率
        target_sum = 1 / 0.90
        implied_sum = 1/psch + 1/pscd + 1/psca
        scale = implied_sum / target_sum
        jc_cd = pscd / scale
        jc_ca = psca / scale
        
        ev_d = proba[1] * jc_cd - 1
        ev_a = proba[2] * jc_ca - 1
        
        if ev_d > ev_a:
            best = 'D'; best_ev = ev_d; odds = jc_cd; prob = proba[1]
        else:
            best = 'A'; best_ev = ev_a; odds = jc_ca; prob = proba[2]
        
        kelly = (prob * odds - 1) / (odds - 1) if odds > 1 else 0
        kelly = max(0, min(kelly, 2.0))
        
        # 显示结果
        st.subheader("📋 预测结果")
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1: st.metric("主胜概率", f"{proba[0]:.1%}")
        with col_r2: st.metric("平局概率", f"{proba[1]:.1%}")
        with col_r3: st.metric("客胜概率", f"{proba[2]:.1%}")
        
        st.divider()
        col_rec1, col_rec2, col_rec3 = st.columns(3)
        with col_rec1: st.metric("推荐方向", "平局" if best=='D' else "客胜")
        with col_rec2: st.metric("竞彩EV", f"{best_ev:+.2%}")
        with col_rec3: st.metric("凯利仓位", f"{kelly:.1%}")
        
        if best_ev > 0.10:
            st.success("✅ 值得投注！")
        else:
            st.warning("❌ 不值得投注")
else:
    st.info("👆 请先上传你的 .pkl 模型文件")

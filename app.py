# 🌍 비체계적 위험 분석 Streamlit 앱 (확장된 키워드 포함)
import streamlit as st
import requests
import re
import pandas as pd
import datetime
import plotly.express as px
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# 보강된 위험 키워드
def get_risk_keywords():
    return {
        # 기존 + 확장 (총 80개 이상)
        "침공": 1.5, "전쟁": 1.6, "내전": 1.5, "폭탄": 1.5, "군사 행동": 1.6,
        "갈등": 1.4, "분쟁": 1.4, "테러": 1.7, "정변": 1.5, "독재": 1.3,
        "정치 불안": 1.6, "쿠데타": 1.7, "봉쇄": 1.4, "검열": 1.3, "사이버 공격": 1.6,
        "보안 침해": 1.6, "해킹": 1.5, "피싱": 1.5, "랜섬웨어": 1.6, "악성코드": 1.5,
        "침해": 1.5, "정보 유출": 1.6, "개인정보 유출": 1.7, "데이터 유출": 1.6,
        "파산": 1.6, "실업": 1.5, "정리해고": 1.5, "퇴출": 1.4, "채무불이행": 1.7,
        "적자": 1.3, "디폴트": 1.7, "부도": 1.6, "불황": 1.5, "경기 침체": 1.5,
        "긴축": 1.3, "금리 인상": 1.3, "환율 불안": 1.4, "인플레이션": 1.4, "스태그플레이션": 1.5,
        "물가 상승": 1.3, "생산 차질": 1.4, "수급 불균형": 1.4, "공급망 붕괴": 1.6,
        "폭동": 1.5, "시위": 1.3, "불법": 1.2, "사기": 1.4, "횡령": 1.4,
        "조작": 1.5, "부패": 1.5, "불공정": 1.3, "불법 행위": 1.5, "의혹": 1.3,
        "기소": 1.2, "체포": 1.3, "징역": 1.2, "혐의": 1.2,
        "리콜": 1.3, "결함": 1.4, "고장": 1.2, "위험": 1.3, "불량": 1.3,
        "붕괴": 1.4, "침몰": 1.5, "화재": 1.4, "감염병": 1.5, "전염병": 1.5,
        "질병": 1.3, "팬데믹": 1.6, "확산": 1.3, "의료 붕괴": 1.5,
        "불매": 1.3, "탈퇴": 1.2, "제재": 1.4, "고립": 1.3,
        "disruption": 1.5, "fraud": 1.4, "shutdown": 1.4, "crisis": 1.6, "layoff": 1.5,
        "protest": 1.3, "riot": 1.6, "collapse": 1.6, "default": 1.7, "sanctions": 1.5
    }

# 보강된 긍정 키워드
POSITIVE_KEYWORDS = {
    "성장": -0.4, "회복": -0.5, "개선": -0.4, "안정": -0.4, "협력": -0.3,
    "평화": -0.6, "합의": -0.3, "외교": -0.3, "정상회담": -0.3, "협상": -0.3,
    "지원": -0.4, "재건": -0.4, "투자": -0.5, "수출 증가": -0.5, "무역 확대": -0.5,
    "경제 회복": -0.5, "기술 혁신": -0.5, "성공": -0.5, "발전": -0.4,
    "기회": -0.4, "신뢰": -0.3, "긍정": -0.3, "안정세": -0.4, "회복세": -0.4,
    "선진화": -0.3, "개방": -0.3, "완화": -0.3, "호전": -0.3, "긍정 평가": -0.3,
    "합작": -0.3, "상생": -0.3, "유치": -0.3, "창업": -0.4, "일자리 창출": -0.4,
    "기술 개발": -0.4, "혁신": -0.5, "신기술": -0.4, "친환경": -0.3, "지속 가능": -0.4,
    "리더십": -0.3, "민주주의": -0.2, "자율성": -0.3, "공정성": -0.2, "청렴": -0.2,
    "의료 개선": -0.4, "백신 확보": -0.4, "안전망": -0.4,
    "breakthrough": -0.5, "recovery": -0.5, "growth": -0.4, "improvement": -0.4,
    "stability": -0.4, "cooperation": -0.3, "diplomacy": -0.3, "innovation": -0.5,
    "opportunity": -0.4, "peace": -0.5, "support": -0.4
}

@st.cache_resource
def load_sentiment_model():
    tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
    model = BertForSequenceClassification.from_pretrained('beomi/kcbert-base', num_labels=3)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def calculate_risk_score(text, keywords):
    text_lower = text.lower()
    score = 0
    for word, weight in keywords.items():
        count = len(re.findall(re.escape(word), text_lower, re.IGNORECASE))
        score += weight * count
    return score

def kobert_sentiment(pipeline, text):
    result = pipeline(text[:512])[0]
    label = result['label']
    score = result['score']
    if label == 'LABEL_0': return -score
    elif label == 'LABEL_2': return score
    return 0

# UI 시작
st.set_page_config(page_title="Unsystematic Risk Analyzer", layout="wide")
st.title("📉 뉴스 기반 비체계적 위험 분석기")

keyword = st.text_input("🔍 뉴스 키워드", "아르헨티나 정치 불안")
api_key = st.text_input("🔑 GNews API Key", type="password")
custom_article = st.text_area("✍️ 분석할 기사 텍스트 직접 입력 (선택)", height=300)

if st.button("🚀 분석 시작"):
    with st.spinner("모델 로딩 중..."):
        nlp = load_sentiment_model()

    if custom_article.strip():
        articles = [custom_article.strip()]
    else:
        url = f"https://gnews.io/api/v4/search?q={keyword}&lang=ko&token={api_key}"
        r = requests.get(url)
        articles = [a['title'] + ' ' + (a['description'] or '') for a in r.json().get('articles', [])]

    if not articles:
        st.warning("뉴스가 없습니다.")
    else:
        rows = []
        risk_dict = get_risk_keywords()
        for t in articles:
            base_score = calculate_risk_score(t, risk_dict)
            bonus = calculate_risk_score(t, POSITIVE_KEYWORDS)
            final_score = base_score + bonus
            polarity = kobert_sentiment(nlp, t)
            score = round(final_score * (1 - polarity), 2)
            rows.append({"텍스트": t, "위험 점수": final_score, "감성 점수": round(polarity, 2), "최종 위험 지수": score,
                         "시간": datetime.datetime.now()})

        df = pd.DataFrame(rows)
        st.success(f"총 {len(df)}개 문서 분석 완료!")
        st.dataframe(df.sort_values("최종 위험 지수", ascending=False))

        # 시각화
        st.subheader("📊 시각화 결과")
        st.plotly_chart(px.line(df, x="시간", y="최종 위험 지수", title="시간별 위험 추세"))
        st.plotly_chart(px.histogram(df, x="최종 위험 지수", nbins=15, title="위험 분포"))
        st.plotly_chart(px.scatter(df, x="감성 점수", y="위험 점수", color="최종 위험 지수", size="최종 위험 지수",
                                   hover_data=["텍스트"], title="감성 vs 위험 점수 산점도"))

        top5 = df.sort_values("최종 위험 지수", ascending=False).head(5)
        st.subheader("🔥 고위험 기사 Top 5")
        for i, row in top5.iterrows():
            st.markdown(f"**{i+1}. 위험도: {row['최종 위험 지수']} / 감성: {row['감성 점수']}**")
            st.caption(row['텍스트'][:400] + ("..." if len(row['텍스트']) > 400 else ""))

        st.download_button("📥 CSV 다운로드", df.to_csv(index=False).encode(), "risk_analysis.csv", "text/csv")
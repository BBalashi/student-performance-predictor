import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
plt.rcParams["font.family"] = "monospace"
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Code Pro', monospace;
    }

    .block-container {
        padding-top: 0rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 100%;
    }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #0d1117 0%, #0e1f3d 100%);
        border-bottom: 1px solid #1e3a5f;
        padding: 3rem 4rem 2rem 4rem;
        margin: -1rem -3rem 2rem -3rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -1px;
        margin: 0;
    }
    .hero-title span {
        color: #4c9be8;
    }
    .hero-subtitle {
        font-size: 0.9rem;
        color: #8892a4;
        margin-top: 0.5rem;
        letter-spacing: 0.5px;
    }
    .hero-tag {
        display: inline-block;
        background: #1e3a5f;
        color: #4c9be8;
        font-size: 0.7rem;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-right: 0.5rem;
        margin-top: 1rem;
    }

    /* KPI cards */
    [data-testid="stMetric"] {
        background: #0d1117;
        border: 1px solid #1e2d3d;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #8892a4 !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }

    /* Section labels */
    .section-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #4c9be8;
        border-left: 2px solid #4c9be8;
        padding-left: 0.6rem;
        margin-bottom: 1rem;
    }

    /* Predictor form */
    .predictor-card {
        background: #0d1117;
        border: 1px solid #1e2d3d;
        border-radius: 8px;
        padding: 1.5rem;
    }

    /* Input labels */
    label {
        font-size: 0.8rem !important;
        color: #8892a4 !important;
        letter-spacing: 0.5px;
    }

    /* Result box */
    [data-testid="stSuccess"] {
        background: #0d2818;
        border: 1px solid #1a5c35;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 600;
    }

    /* Button */
    [data-testid="stBaseButton-secondary"] {
        background: #4c9be8 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'Source Code Pro', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        padding: 0.6rem 2rem !important;
        width: 100%;
    }

    hr { border-color: #1e2d3d; }
    </style>
""", unsafe_allow_html=True)

# --- Load and prepare data ---
df = pd.read_csv("student-mat.csv", sep=";")
df = df.drop(columns=["G1", "G2"])
df = pd.get_dummies(df)

X = df.drop(columns=["G3"])
y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(
    max_depth=15,
    min_samples_split=5,
    n_estimators=300,
    random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

# --- Hero Header ---
st.markdown("""
    <div class="hero">
        <div class="hero-title">Student <span>Performance</span> Predictor</div>
        <div class="hero-subtitle">Random Forest model trained on 395 Portuguese students · UCI Machine Learning Repository</div>
        <span class="hero-tag">Regression</span>
        <span class="hero-tag">Random Forest</span>
        <span class="hero-tag">Education</span>
    </div>
""", unsafe_allow_html=True)

# --- KPI Cards ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Students", "395")
col2.metric("Features", X.shape[1])
col3.metric("Model MAE", f"{mae:.2f} pts")
col4.metric("Test Set Size", "79 students")

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br>", unsafe_allow_html=True)

# --- Charts ---
BG = "#0d1117"
ACCENT = "#4c9be8"
TEXT = "#8892a4"
GRID = "#1a2535"

chart_col1, spacer, chart_col2 = st.columns([10, 1, 10])

with chart_col1:
    st.markdown('<div class="section-label">Model Accuracy</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Color dots by error magnitude
    errors = np.abs(y_test.values - predictions)
    scatter = ax.scatter(y_test, predictions, c=errors, cmap="RdYlGn_r",
                        alpha=0.8, s=50, edgecolors="none", vmin=0, vmax=8)
    ax.plot([0, 20], [0, 20], color="#ff4b4b", linestyle="--", linewidth=1.2, label="Perfect prediction")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.ax.tick_params(colors=TEXT, labelsize=7)
    cbar.set_label("Error (grade pts)", color=TEXT, fontsize=8)
    cbar.outline.set_edgecolor(GRID)

    ax.set_xlabel("Actual Grade", color=TEXT, fontsize=9)
    ax.set_ylabel("Predicted Grade", color=TEXT, fontsize=9)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.grid(color=GRID, linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    fig.tight_layout()
    st.pyplot(fig)

with chart_col2:
    st.markdown('<div class="section-label">Feature Importance</div>', unsafe_allow_html=True)
    rename_map = {
        "goout": "Goes Out",
        "Fedu": "Father's Education",
        "Walc": "Weekend Alcohol",
        "Medu": "Mother's Education",
        "freetime": "Free Time",
        "traveltime": "Travel Time",
        "studytime": "Study Time",
        "absences": "Absences",
        "failures": "Past Failures",
        "health": "Health",
        "age": "Age",
    }
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance.index = [rename_map.get(col, col) for col in importance.index]
    importance = importance.sort_values(ascending=False).head(10)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    fig2.patch.set_facecolor(BG)
    ax2.set_facecolor(BG)

    colors = [ACCENT if i < 2 else "#2a5a8a" for i in range(len(importance))]
    ax2.barh(importance.index[::-1], importance.values[::-1], color=colors[::-1], height=0.55)

    ax2.tick_params(colors=TEXT, labelsize=8)
    ax2.set_xlabel("Importance Score", color=TEXT, fontsize=9)
    ax2.grid(axis="x", color=GRID, linewidth=0.5, linestyle="--")
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID)

    top2 = mpatches.Patch(color=ACCENT, label="Top predictors")
    rest = mpatches.Patch(color="#2a5a8a", label="Other features")
    ax2.legend(handles=[top2, rest], fontsize=8, facecolor=BG, edgecolor=GRID, labelcolor=TEXT)

    fig2.tight_layout()
    st.pyplot(fig2)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown("<br>", unsafe_allow_html=True)

# --- Live Predictor ---
st.markdown('<div class="section-label">Grade Predictor</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

input_col1, input_col2, result_col = st.columns([5, 5, 4])

with input_col1:
    absences = st.number_input("Number of absences", min_value=0, max_value=120, value=0)
    failures = st.number_input("Past class failures", min_value=0, max_value=100, value=0)
    studytime = st.selectbox("Weekly study time",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "< 2 hours", 2: "2 – 5 hours", 3: "5 – 10 hours", 4: "> 10 hours"}[x])

with input_col2:
    health = st.selectbox("Health status", options=[1, 2, 3, 4, 5],
        format_func=lambda x: {1: "1 — Very bad", 2: "2 — Bad", 3: "3 — Average", 4: "4 — Good", 5: "5 — Excellent"}[x])
    goout = st.selectbox("Goes out with friends", options=[1, 2, 3, 4, 5],
        format_func=lambda x: {1: "1 — Very low", 2: "2 — Low", 3: "3 — Average", 4: "4 — High", 5: "5 — Very high"}[x])
    st.markdown("<br>", unsafe_allow_html=True)
    predict = st.button("Run Prediction")

with result_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if predict:
        input_data = pd.DataFrame(0, index=[0], columns=X.columns, dtype=float)
        input_data["absences"] = float(absences)
        input_data["failures"] = float(failures)
        input_data["studytime"] = float(studytime)
        input_data["health"] = float(health)
        input_data["goout"] = float(goout)
        predicted_grade = model.predict(input_data)[0]

        if predicted_grade >= 14:
            verdict = "Strong performance expected"
            color = "#1a5c35"
            border = "#2ecc71"
        elif predicted_grade >= 10:
            verdict = "Average performance expected"
            color = "#3d3000"
            border = "#f39c12"
        else:
            verdict = "At risk — may need support"
            color = "#3d0000"
            border = "#e74c3c"

        st.markdown(f"""
            <div style="background:{color}; border:1px solid {border}; border-radius:8px;
                        padding:1.5rem; text-align:center; margin-top:0.5rem;">
                <div style="font-size:0.7rem; letter-spacing:3px; text-transform:uppercase;
                            color:{border}; margin-bottom:0.5rem;">Predicted Grade</div>
                <div style="font-size:3.5rem; font-weight:700; color:white; line-height:1;">
                    {predicted_grade:.1f}
                </div>
                <div style="font-size:0.75rem; color:{border}; margin-top:0.3rem;">out of 20</div>
                <hr style="border-color:{border}33; margin:0.8rem 0;">
                <div style="font-size:0.75rem; color:#cccccc;">{verdict}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background:#0d1117; border:1px dashed #1e2d3d; border-radius:8px;
                        padding:1.5rem; text-align:center; margin-top:0.5rem; color:#8892a4;">
                <div style="font-size:0.7rem; letter-spacing:3px; text-transform:uppercase; margin-bottom:0.5rem;">
                    Awaiting Input
                </div>
                <div style="font-size:2rem;">—</div>
                <div style="font-size:0.75rem; margin-top:0.5rem;">
                    Fill in the fields and run prediction
                </div>
            </div>
        """, unsafe_allow_html=True)

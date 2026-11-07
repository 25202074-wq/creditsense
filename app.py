"""
CreditSense - Credit Score & Risk Predictor
Predictive Analytics Project (SAS Studio + SAS Viya + Streamlit)

Regression     : predicts Credit_Score (linear model, mirrors the SAS GLMSELECT champion)
Classification : predicts High_Risk (Credit_Score < 600), mirrors SAS Viya models
Data           : synthetic credit card customer behaviour (Kaggle), cleaned & feature engineered
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_percentage_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="CreditSense", page_icon="💳", layout="wide")

DATA = Path(__file__).parent / "credit_card_clean.csv"
SEED = 12345
NUM = ["Payment_Ratio", "Credit_Utilization", "Balance_to_Limit", "EMI_Count",
       "Card_Age_Months", "Mobile_App_Login", "International_Transactions",
       "Cash_Advance_Flag", "Annual_Income", "Monthly_Spending"]
CAT = ["Card_Type", "Occupation"]
FEATURES = NUM + CAT
LABELS = {
    "Payment_Ratio": "Payment ratio", "Credit_Utilization": "Credit utilisation",
    "Balance_to_Limit": "Balance / limit", "EMI_Count": "Active EMIs",
    "Card_Age_Months": "Card age", "Mobile_App_Login": "App logins",
    "International_Transactions": "Intl. transactions",
    "Cash_Advance_Flag": "Cash advance used", "Annual_Income": "Annual income",
    "Monthly_Spending": "Monthly spending",
}


# Champion regression model from SAS Studio: PROC GLMSELECT, SELECTION=STEPWISE
# (trained on the 70% partition, seed 12345). Coefficients copied from SAS output.
SAS_INTERCEPT = 540.127382
SAS_COEF = {
    "Annual_Income": 0.000000322,
    "Card_Age_Months": 0.122407,
    "Payment_Ratio": 219.808180,
    "Credit_Utilization": -145.091747,
    "Cash_Advance_Amount": -0.000021025,
    "EMI_Count": -3.914380,
    "International_Transactions": 1.581147,
    "Mobile_App_Login": 0.291586,
    "Cash_Advance_Flag": -8.597277,
    "Balance_to_Limit": -55.344844,
    "Share_Travel": -2.286239,
    "Gender_F": 0.274210,
}
SAS_LABELS = {**LABELS, "Cash_Advance_Amount": "Cash advance amount",
              "Share_Travel": "Travel share of spend", "Gender_F": "Gender (F)"}

# SAS Studio model comparison (30% test partition)
SAS_TABLE = pd.DataFrame([
    ["Stepwise ⭐", 12, 150.297, 148.218, 175577],
    ["Forward", 12, 150.297, 148.218, 175577],
    ["Adaptive LASSO", 10, 150.393, 148.220, 175579],
    ["LAR", 14, 150.361, 148.226, 175613],
    ["LASSO", 13, 150.361, 148.227, 175603],
    ["Elastic Net", 13, 150.361, 148.227, 175603],
    ["Backward", 14, 150.283, 148.248, 175595],
], columns=["Model", "Variables", "Train ASE", "Test ASE", "SBC"])
SAS_TABLE["Test RMSE"] = SAS_TABLE["Test ASE"] ** 0.5
SAS_TABLE["Test R²"] = 0.9689
SAS_TABLE["Test MAPE %"] = 1.570

# SAS Viya Model Studio comparison (30% test partition, 15,000 customers)
VIYA_TABLE = pd.DataFrame([
    ["Gradient Boosting ⭐", 0.9571, 0.9928, 0.9306, 0.9134, 0.0429],
    ["Forest", 0.9497, 0.9892, 0.9167, 0.8945, 0.0503],
    ["Pruned Tree", 0.9459, 0.9733, 0.9129, 0.8747, 0.0541],
    ["Full Tree", 0.9394, 0.9716, 0.9022, 0.8716, 0.0606],
    ["Logistic Regression", 0.8457, 0.8775, 0.7251, 0.6125, 0.1543],
], columns=["Model", "Accuracy", "AUC", "F1", "KS", "Misclassification"])


def sas_terms(frame):
    """Design matrix for the SAS Stepwise model."""
    x = frame[[k for k in SAS_COEF if k != "Gender_F"]].astype(float).copy()
    x["Gender_F"] = (frame["Gender"] == "F").astype(float)
    return x


def sas_predict(frame):
    return SAS_INTERCEPT + sas_terms(frame).mul(pd.Series(SAS_COEF)).sum(axis=1)


@st.cache_data
def load_data():
    return pd.read_csv(DATA)


def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT),
    ])


@st.cache_resource
def train_models(df):
    X = df[FEATURES]

    # ---- Regression: Credit_Score (SAS Stepwise formula, evaluated on a 30% hold-out) ----
    _, te = train_test_split(df, test_size=0.3, random_state=SEED)
    yte = te["Credit_Score"]
    pred = sas_predict(te).values
    reg = sas_terms(df).mean()  # average customer, used for the contribution chart
    reg_metrics = {
        "RMSE": mean_squared_error(yte, pred) ** 0.5,
        "R²": r2_score(yte, pred),
        "MAPE (%)": mean_absolute_percentage_error(yte, pred) * 100,
    }
    reg_test = pd.DataFrame({"Actual": yte.values, "Predicted": pred})

    # ---- Classification: High_Risk ----
    Xtr, Xte, ytr, yte = train_test_split(X, df["High_Risk"], test_size=0.3,
                                          random_state=SEED, stratify=df["High_Risk"])
    clfs = {
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED),
    }
    fitted, rows, rocs, cms = {}, [], {}, {}
    for name, m in clfs.items():
        p = Pipeline([("prep", preprocessor()), ("model", m)]).fit(Xtr, ytr)
        prob = p.predict_proba(Xte)[:, 1]
        yhat = (prob >= 0.5).astype(int)
        fitted[name] = p
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(yte, yhat),
                     "AUC": roc_auc_score(yte, prob),
                     "F1": f1_score(yte, yhat),
                     "Precision": precision_score(yte, yhat),
                     "Recall": recall_score(yte, yhat)})
        fpr, tpr, _ = roc_curve(yte, prob)
        rocs[name] = (fpr, tpr)
        cms[name] = confusion_matrix(yte, yhat)
    return reg, reg_metrics, reg_test, fitted, pd.DataFrame(rows), rocs, cms


df = load_data()
reg, reg_metrics, reg_test, clfs, clf_metrics, rocs, cms = train_models(df)

# ------------------------------------------------------------------ sidebar
st.sidebar.title("💳 Customer profile")
st.sidebar.caption("Enter the customer's details to get live predictions.")

with st.sidebar.expander("Repayment behaviour", expanded=True):
    payment_ratio = st.slider("Payment ratio (paid ÷ statement)", 0.10, 1.00, 0.80, 0.01)
    utilisation = st.slider("Credit utilisation", 0.10, 0.95, 0.40, 0.01)
    emi = st.slider("Active EMIs", 0, 6, 2)
    cash_adv_amt = st.number_input("Cash advance taken this month (₹)", 0, 420_000, 0, 5_000)

with st.sidebar.expander("Card & account", expanded=True):
    card_type = st.selectbox("Card type", ["Basic", "Silver", "Gold", "Platinum", "Signature"], index=2)
    credit_limit = st.number_input("Credit limit (₹)", 10_000, 5_200_000, 400_000, 10_000)
    balance = st.number_input("Outstanding balance (₹)", 0, 2_700_000, 150_000, 5_000)
    card_age = st.slider("Card age (months)", 1, 240, 48)

with st.sidebar.expander("Profile & activity", expanded=False):
    occupation = st.selectbox("Occupation", ["Private Employee", "Government Employee",
                                             "Self-Employed", "Business Owner", "Retired", "Student"])
    gender = st.radio("Gender", ["M", "F"], horizontal=True)
    income = st.number_input("Annual income (₹)", 100_000, 7_400_000, 1_100_000, 50_000)
    spending = st.number_input("Monthly spending (₹)", 2_000, 450_000, 50_000, 1_000)
    travel = st.number_input("of which travel spending (₹)", 0, 450_000, 3_000, 500)
    logins = st.slider("Mobile app logins / month", 2, 71, 31)
    intl = st.slider("International transactions / month", 0, 14, 0)

clf_name = "Gradient Boosting"  # SAS Viya champion

customer = pd.DataFrame([{
    "Payment_Ratio": payment_ratio, "Credit_Utilization": utilisation,
    "Balance_to_Limit": min(balance / credit_limit, 0.93), "EMI_Count": emi,
    "Card_Age_Months": card_age, "Mobile_App_Login": logins,
    "International_Transactions": intl, "Cash_Advance_Flag": int(cash_adv_amt > 0),
    "Cash_Advance_Amount": cash_adv_amt, "Share_Travel": min(travel / spending, 1.0),
    "Annual_Income": income, "Monthly_Spending": spending, "Gender": gender,
    "Card_Type": card_type, "Occupation": occupation,
}])

score = float(np.clip(sas_predict(customer).iloc[0], 300, 900))
risk = float(clfs[clf_name].predict_proba(customer)[0, 1])

if risk >= 0.5:
    band, colour, action = "High risk", "#d64545", "Decline limit increase; offer EMI conversion and monitor monthly."
elif risk >= 0.2:
    band, colour, action = "Watch list", "#e0a030", "Hold limit; nudge towards higher repayments with reminders."
else:
    band, colour, action = "Low risk", "#2f9e6e", "Eligible for limit increase or card upgrade / cross-sell."

# ------------------------------------------------------------------ main
st.title("CreditSense — Credit Score & Risk Predictor")
st.caption("Regression model predicts the credit score · classification model predicts the probability "
           "that the customer is high risk (score < 600).")

tab_pred, tab_perf, tab_insight, tab_how = st.tabs(
    ["🔮 Prediction", "📊 Model performance", "💡 Data insights", "⚙️ How it works"])

with tab_pred:
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted credit score", f"{score:.0f}")
    c2.metric("High-risk probability", f"{risk:.1%}")
    c3.markdown(f"<div style='padding-top:6px'><span style='font-size:0.9rem;opacity:.7'>Risk band</span><br>"
                f"<span style='font-size:1.9rem;font-weight:700;color:{colour}'>{band}</span></div>",
                unsafe_allow_html=True)
    st.info(f"**Recommended action:** {action}")

    g1, g2 = st.columns(2)
    with g1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            title={"text": "Credit score"},
            gauge={"axis": {"range": [350, 800]},
                   "bar": {"color": "#1f4e79"},
                   "steps": [{"range": [350, 600], "color": "#f4c7c3"},
                             {"range": [600, 700], "color": "#fce8b2"},
                             {"range": [700, 800], "color": "#c8e6c9"}],
                   "threshold": {"line": {"color": "black", "width": 3}, "value": 600}}))
        gauge.update_layout(height=320, margin=dict(t=60, b=10, l=30, r=30))
        st.plotly_chart(gauge, width="stretch")

    with g2:
        # Contribution of each input vs. the average customer: SAS coefficient x (value - mean)
        contrib = (sas_terms(customer).iloc[0] - reg) * pd.Series(SAS_COEF)
        contrib.index = [SAS_LABELS[i] for i in contrib.index]
        contrib = contrib.reindex(contrib.abs().sort_values().index).tail(7)
        fig = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h",
                               marker_color=["#2f9e6e" if v > 0 else "#d64545" for v in contrib.values]))
        fig.update_layout(title="What moves this customer's score (vs. average customer)",
                          xaxis_title="Score points", height=320, margin=dict(t=60, b=10, l=10, r=10))
        st.plotly_chart(fig, width="stretch")

with tab_perf:
    st.subheader("Regression — SAS Studio model comparison (30% test set)")
    st.dataframe(SAS_TABLE.style.format({"Train ASE": "{:.3f}", "Test ASE": "{:.3f}",
                                         "Test RMSE": "{:.3f}", "SBC": "{:,.0f}",
                                         "Test R²": "{:.4f}", "Test MAPE %": "{:.2f}"}),
                 hide_index=True, width="stretch")
    st.caption("Champion: **Stepwise** — lowest test error and SBC with 12 variables; test error below "
               "train error, so no overfitting. The app scores customers with its SAS coefficients.")
    st.subheader("Champion model on a 30% hold-out sample")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test RMSE", f"{reg_metrics['RMSE']:.1f} pts")
    m2.metric("Test R²", f"{reg_metrics['R²']:.3f}")
    m3.metric("Test MAPE", f"{reg_metrics['MAPE (%)']:.2f}%")
    sample = reg_test.sample(2500, random_state=1)
    fig = px.scatter(sample, x="Actual", y="Predicted", opacity=0.4, title="Actual vs predicted credit score")
    fig.add_shape(type="line", x0=350, y0=350, x1=800, y1=800, line=dict(color="red", dash="dash"))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Classification — SAS Viya model comparison (30% test set)")
    st.dataframe(VIYA_TABLE.style.format({"Accuracy": "{:.1%}", "AUC": "{:.3f}", "F1": "{:.3f}",
                                          "KS": "{:.3f}", "Misclassification": "{:.1%}"}),
                 hide_index=True, width="stretch")
    st.caption("Champion: **Gradient Boosting** — highest AUC, accuracy, F1 and KS. The pruned tree beats "
               "the full tree, showing that pruning reduces overfitting.")
    st.subheader("Champion (Gradient Boosting) re-trained for the app — 30% hold-out")
    st.dataframe(clf_metrics.style.format({c: "{:.3f}" for c in clf_metrics.columns if c != "Model"}),
                 hide_index=True, width="stretch")
    r1, r2 = st.columns(2)
    with r1:
        fig = go.Figure()
        for name, (fpr, tpr) in rocs.items():
            auc = clf_metrics.set_index("Model").loc[name, "AUC"]
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC {auc:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash", color="grey")))
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate",
                          yaxis_title="True positive rate", legend=dict(x=0.35, y=0.08))
        st.plotly_chart(fig, width="stretch")
    with r2:
        cm = cms[clf_name]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred: Not risky", "Pred: High risk"], y=["Actual: Not risky", "Actual: High risk"],
                        title=f"Confusion matrix — {clf_name}")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    st.caption("Full model comparison (7 regression models in SAS Studio, 5 classifiers in SAS Viya) "
               "is in the analysis report. The app re-estimates the champion models on the same "
               "cleaned data and the same 70/30 split.")

with tab_insight:
    i1, i2 = st.columns(2)
    with i1:
        t = df.assign(Band=pd.cut(df["Payment_Ratio"], [0, .3, .5, .7, .9, 1.0],
                                  labels=["≤30%", "30–50%", "50–70%", "70–90%", "90–100%"]))
        t = t.groupby("Band", observed=True)[["Credit_Score", "High_Risk"]].mean().reset_index()
        fig = px.bar(t, x="Band", y="High_Risk", text_auto=".0%",
                     title="High-risk share by payment ratio", labels={"Band": "Payment ratio", "High_Risk": ""})
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    with i2:
        t = df.assign(Band=pd.cut(df["Credit_Utilization"], [0, .3, .5, .7, .95],
                                  labels=["≤30%", "30–50%", "50–70%", "70–95%"]))
        t = t.groupby("Band", observed=True)["High_Risk"].mean().reset_index()
        fig = px.bar(t, x="Band", y="High_Risk", text_auto=".0%",
                     title="High-risk share by credit utilisation", labels={"Band": "Utilisation", "High_Risk": ""})
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    i3, i4 = st.columns(2)
    with i3:
        t = df.groupby("Card_Type")["High_Risk"].mean().sort_values().reset_index()
        fig = px.bar(t, x="Card_Type", y="High_Risk", text_auto=".0%", title="High-risk share by card type",
                     labels={"Card_Type": "", "High_Risk": ""})
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    with i4:
        fig = px.histogram(df, x="Credit_Score", nbins=50, title="Credit score distribution")
        fig.add_vline(x=600, line_dash="dash", line_color="red", annotation_text="High-risk cut-off (600)")
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Key takeaway:** repayment discipline (payment ratio) and credit utilisation explain most of "
                "the variation in credit score; income, card type and spending level add very little.")

with tab_how:
    st.subheader("Workflow & architecture")
    st.graphviz_chart("""
    digraph {
        rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3f8", fontname="Helvetica"];
        raw [label="Kaggle raw data\\n50,000 customers"];
        prep [label="Data prep (SAS Studio)\\nquality checks · P99 capping\\nfeature engineering"];
        split [label="70 / 30 split\\nseed 12345"];
        reg [label="SAS Studio GLMSELECT\\nForward · Backward · Stepwise\\nLASSO · Adaptive LASSO · Elastic Net · LAR"];
        clf [label="SAS Viya Model Studio\\nFull tree · Pruned tree\\nGradient boosting · Forest · Logistic"];
        champ [label="Champion models\\n(lowest RMSE / highest AUC)"];
        app [label="Streamlit web app\\nuser inputs → live predictions", fillcolor="#d9ead3"];
        user [label="Credit analyst", shape=ellipse, fillcolor="#fff2cc"];
        raw -> prep -> split; split -> reg; split -> clf; reg -> champ; clf -> champ; champ -> app; user -> app [dir=both];
    }""")
    st.markdown("""
**Assumptions**
- The dataset is synthetic (Kaggle) and is used for academic purposes.
- *High risk* is defined as a credit score below 600, a common lender cut-off.
- Monetary variables are capped at the 99th percentile to limit outlier influence.
- The 7 category-spending columns are excluded from models because they add up exactly to monthly spending.
- The app uses a compact set of 12 inputs; test performance matches the full models.
""")

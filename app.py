import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

st.set_page_config(page_title="AI-Powered EDA Tool", layout="wide")
st.title("AI-Powered Exploratory Data Analysis")
st.caption("UCI Adult Income Dataset — 32,561 records")


@st.cache_data
def load_data():
    df = pd.read_csv(URL, header=None, names=COLUMNS)
    df = df.apply(lambda x: x.str.strip() if pd.api.types.is_string_dtype(x) else x)
    return df


df = load_data()

# --- Dataset Overview ---
st.header("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Features", len(df.columns) - 1)
col3.metric("Earning >50K", f"{(df['income'] == '>50K').sum():,}")
col4.metric("Earning ≤50K", f"{(df['income'] == '<=50K').sum():,}")

st.dataframe(df.head(10), use_container_width=True)

st.divider()

# --- Charts ---
st.header("Exploratory Analysis")

sns.set_theme(style="darkgrid")

# Row 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("Age Distribution by Income")
    fig, ax = plt.subplots(figsize=(7, 4))
    for income_class, group in df.groupby("income"):
        ax.hist(group["age"], bins=30, alpha=0.6, label=income_class)
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Income Class Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df["income"].value_counts()
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452"])
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f"{v:,}", ha="center", fontweight="bold")
    st.pyplot(fig)
    plt.close()

# Row 2
col1, col2 = st.columns(2)

with col1:
    st.subheader("Education vs Income (>50K %)")
    edu_income = df.groupby("education")["income"].apply(
        lambda x: (x == ">50K").mean() * 100
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(edu_income.index, edu_income.values, color="#4C72B0")
    ax.set_xlabel("% Earning >50K")
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Hours per Week Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["hours-per-week"], bins=40, color="#4C72B0", edgecolor="white")
    ax.axvline(df["hours-per-week"].mean(), color="red", linestyle="--",
               label=f'Mean: {df["hours-per-week"].mean():.1f}h')
    ax.set_xlabel("Hours per Week")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    plt.close()

# Row 3
col1, col2 = st.columns(2)

with col1:
    st.subheader("Income by Sex")
    sex_income = df.groupby(["sex", "income"]).size().unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    sex_income.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Top Occupations by >50K Rate")
    occ_income = df.groupby("occupation")["income"].apply(
        lambda x: (x == ">50K").mean() * 100
    ).sort_values(ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(occ_income.index, occ_income.values, color="#4C72B0")
    ax.set_xlabel("% Earning >50K")
    st.pyplot(fig)
    plt.close()

# Row 4 — Correlation heatmap (full width)
st.subheader("Correlation Heatmap — Numerical Features")
numeric_cols = df.select_dtypes(include="number").drop(columns=["fnlwgt"])
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)
plt.close()

st.divider()

# --- AI Summary ---
st.header("AI-Generated Summary")
st.write("Click below to generate an intelligent summary of the EDA findings using GPT-4o-mini.")

if st.button("Generate AI Summary", type="primary"):
    stats = f"""
Dataset: UCI Adult Income ({len(df):,} records, {len(df.columns)} features)
Income split: {(df['income'] == '>50K').mean()*100:.1f}% earn >50K, {(df['income'] == '<=50K').mean()*100:.1f}% earn <=50K
Average age: {df['age'].mean():.1f} years
Average hours/week: {df['hours-per-week'].mean():.1f}
Top education for >50K: {df[df['income']=='>50K']['education'].mode()[0]}
Top occupation for >50K: {df[df['income']=='>50K']['occupation'].mode()[0]}
Gender split >50K: Males {(df[df['sex']=='Male']['income']=='>50K').mean()*100:.1f}%, Females {(df[df['sex']=='Female']['income']=='>50K').mean()*100:.1f}%
Missing values: {df.isin(['?']).sum().sum()} cells contain '?'
"""
    with st.spinner("Generating summary..."):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analyst. Write a clear, insightful 3-paragraph summary of EDA findings. Focus on patterns, inequalities, and actionable insights. Be specific with numbers."},
                {"role": "user", "content": f"Here are the EDA statistics:\n{stats}\nWrite an intelligent summary."}
            ]
        )
    st.markdown(response.choices[0].message.content)

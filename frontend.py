import streamlit as st
import requests
import os
import pandas as pd

API_URL = "http://localhost:8000/analyze"

st.set_page_config(page_title="XAI Sentiment Analysis", layout="wide")
st.title("🔍 Explainable Sentiment Classification")

# Sidebar
st.sidebar.markdown("### Input Settings")

default_text = (
    "I was skeptical about the new phone, expecting poor performance, "
    "but its fast processor and stunning display completely changed my mind."
)
input_text = st.sidebar.text_area("Enter a sentence to analyze", value=default_text, height=150)
label = st.sidebar.selectbox("Select target label", ["Negative", "Positive"])
label_id = 0 if label == "Negative" else 1

if st.sidebar.button("🔁 Run XAI Analysis"):
    if not input_text.strip():
        st.warning("Please enter a sentence.")
    else:
        with st.spinner("Running analysis..."):
            response = requests.post(API_URL, json={"text": input_text, "target_label": label_id})

        if response.status_code == 200:
            result = response.json()
            st.success("Analysis complete")

            # ──────────────────────────────────────────────
            # 📊 Infidelity Scores Table (Corrected keys)
            # ──────────────────────────────────────────────
            st.subheader("Infidelity Scores")
            infid = result["infidelity_scores"]
            df = pd.DataFrame([
                ["Integrated Gradients", infid["ig"]],
                ["Layer Integrated Gradients", infid["lig"]],
                ["Layer Conductance", infid["lc"]]
            ], columns=["Method", "Infidelity Score"])
            st.table(df)

            # ──────────────────────────────────────────────
            # 🧠 Integrated Gradients
            # ──────────────────────────────────────────────
            st.subheader("Integrated Gradients")
            st.image(result["visualizations"]["ig"], caption="Integrated Gradients – Attribution Plot")

            # ──────────────────────────────────────────────
            # 🔬 Layer Integrated Gradients
            # ──────────────────────────────────────────────
            st.subheader("🔬 Layer Integrated Gradients")
            st.image(result["visualizations"]["lig"], caption="Layer Integrated Gradients – Summary Plot")

            lig_html_path = result["visualizations"]["lig_html"]
            if os.path.exists(lig_html_path):
                st.markdown("**LIG Token-Level Visualization (Captum)**")
                with open(lig_html_path, "r", encoding="utf-8") as f:
                    html_data = f.read()
                    st.components.v1.html(html_data, height=400, scrolling=True)
            else:
                st.warning(f"{lig_html_path} not found.")

            # ──────────────────────────────────────────────
            # ⚡ Layer Conductance
            # ──────────────────────────────────────────────
            st.subheader("Layer Conductance")
            st.image(result["visualizations"]["lc_bars"], caption="Layer Conductance – Token Attribution (Bars)")
            st.image(result["visualizations"]["lc_comparison"], caption="Layer Conductance – Line Plot Comparison")

            lc_html_path = result["visualizations"]["lc_html"]
            if os.path.exists(lc_html_path):
                st.markdown("**LC Token-Level Visualization (Captum)**")
                with open(lc_html_path, "r", encoding="utf-8") as f:
                    html_data = f.read()
                    st.components.v1.html(html_data, height=400, scrolling=True)
            else:
                st.warning(f"{lc_html_path} not found.")
        else:
            st.error("Backend error: " + str(response.text))

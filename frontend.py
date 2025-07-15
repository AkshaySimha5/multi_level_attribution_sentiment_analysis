import streamlit as st
import requests
import os
import pandas as pd

API_URL = "http://localhost:8000/analyze"

st.set_page_config(page_title="XAI Sentiment Analysis", layout="wide")

# Custom CSS for white background and black text
st.markdown("""
<style>
    .stApp {
        background-color: white;
        color: black;
    }
    .stMarkdown {
        color: black;
    }
    .stSelectbox label {
        color: black;
    }
    .stTextArea label {
        color: black;
    }
    .stNumberInput label {
        color: black;
    }
    .stButton button {
        background-color: #f0f0f0;
        color: black;
    }
    .stTable {
        background-color: black;
        color: black;
    }
    .stDataFrame {
        background-color: white;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

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

# Neuron Conductance settings
st.sidebar.markdown("### Neuron Conductance Settings")
neuron_idx = st.sidebar.number_input("Neuron Index", min_value=0, max_value=767, value=50, step=1)

if st.sidebar.button("🔁 Run XAI Analysis"):
    if not input_text.strip():
        st.warning("Please enter a sentence.")
    else:
        with st.spinner("Running analysis..."):
            response = requests.post(API_URL, json={
                "text": input_text, 
                "target_label": label_id,
                "neuron_idx": neuron_idx
            })
         
        if response.status_code == 200:
            result = response.json()
            st.success("Analysis complete")
            
            # Infidelity Scores
            st.subheader("📊 Infidelity Scores")
            infid = result["infidelity_scores"]
            df = pd.DataFrame([
                ["📈 Integrated Gradients", infid["ig"]],
                ["🔬 Layer Integrated Gradients", infid["lig"]],
                ["⚡ Layer Conductance", infid["lc"]],
                ["🔵 Neuron Conductance", infid.get("nc", "N/A")]
            ], columns=["Method", "Infidelity Score"])
            st.table(df)
            
            # Integrated Gradients
            st.subheader("📈 Integrated Gradients")
            st.image(result["visualizations"]["ig"], caption="Integrated Gradients – Attribution Plot")
            
            # Layer Integrated Gradients
            st.subheader("🔬 Layer Integrated Gradients")
            st.image(result["visualizations"]["lig"], caption="Layer Integrated Gradients – Summary Plot")
            
            lig_html_path = result["visualizations"]["lig_html"]
            if os.path.exists(lig_html_path):
                with open(lig_html_path, "r", encoding="utf-8") as f:
                    html_data = f.read()
                    st.components.v1.html(html_data, height=400, scrolling=True)
            
            # Layer Conductance
            st.subheader("⚡ Layer Conductance")
            st.image(result["visualizations"]["lc_bars"], caption="Layer Conductance – Token Attribution")
            st.image(result["visualizations"]["lc_comparison"], caption="Layer Conductance – Comparison")
            
            lc_html_path = result["visualizations"]["lc_html"]
            if os.path.exists(lc_html_path):
                with open(lc_html_path, "r", encoding="utf-8") as f:
                    html_data = f.read()
                    st.components.v1.html(html_data, height=400, scrolling=True)
            
            # Neuron Conductance
            st.subheader(f"🔵 Neuron Conductance (Neuron {neuron_idx})")
            
            if "nc_bar" in result["visualizations"]:
                st.image(result["visualizations"]["nc_bar"], 
                        caption=f"Neuron {neuron_idx} – Attribution Scores")
            
            if "nc_activation_comparison" in result["visualizations"]:
                st.image(result["visualizations"]["nc_activation_comparison"], 
                        caption=f"Neuron {neuron_idx} – Attribution vs Activation")
            
            if "nc_heatmap" in result["visualizations"]:
                st.image(result["visualizations"]["nc_heatmap"], 
                        caption=f"Neuron {neuron_idx} – Attribution Heatmap")
            
            nc_html_path = result["visualizations"].get("nc_html")
            if nc_html_path and os.path.exists(nc_html_path):
                with open(nc_html_path, "r", encoding="utf-8") as f:
                    html_data = f.read()
                    st.components.v1.html(html_data, height=400, scrolling=True)
            
        else:
            st.error("Backend error: " + str(response.text))

# Information Panel
with st.expander("ℹ️ About the Methods"):
    st.markdown("""
    **Integrated Gradients (IG):** Computes attribution by integrating gradients along a straight path from a baseline to the input.
    
    **Layer Integrated Gradients (LIG):** Applies integrated gradients to internal layer representations.
    
    **Layer Conductance (LC):** Measures how much each neuron in a layer contributes to the final prediction.
    
    **Neuron Conductance (NC):** Analyzes the contribution of a specific neuron to the model's prediction by measuring how the neuron's activation influences the output.
    """)
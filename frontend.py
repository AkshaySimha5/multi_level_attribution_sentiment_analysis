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

# NOTE: Removed neuron_idx setting from sidebar

if st.sidebar.button("Run XAI Analysis"):
    if not input_text.strip():
        st.warning("Please enter a sentence.")
    else:
        with st.spinner("Running analysis..."):
            try:
                response = requests.post(API_URL, json={
                    "text": input_text,
                    "target_label": label_id
                })

                if response.status_code == 200:
                    result = response.json()
                    st.success("Analysis complete")

                    # Infidelity Scores
                    st.subheader("Infidelity Scores")
                    infid = result["infidelity_scores"]
                    df = pd.DataFrame([
                        ["Integrated Gradients", infid.get("ig")],
                        ["Layer Integrated Gradients", infid.get("lig")],
                        ["Layer Conductance", infid.get("lc")],
                        ["Kernel SHAP", infid.get("ks", "N/A")]
                    ], columns=["Method", "Infidelity Score"])
                    st.table(df)

                    # Integrated Gradients
                    st.subheader("Integrated Gradients")
                    ig_path = result["visualizations"].get("ig")
                    if ig_path and os.path.exists(ig_path):
                        st.image(ig_path, caption="Integrated Gradients – Attribution Plot")
                    else:
                        st.warning("Integrated Gradients visualization is not available.")

                    # Layer Integrated Gradients
                    st.subheader("Layer Integrated Gradients")
                    lig_path = result["visualizations"].get("lig")
                    if lig_path and os.path.exists(lig_path):
                        st.image(lig_path, caption="Layer Integrated Gradients – Summary Plot")
                    else:
                        st.warning("Layer Integrated Gradients visualization is not available.")

                    lig_html_path = result["visualizations"].get("lig_html")
                    if lig_html_path and os.path.exists(lig_html_path):
                        with open(lig_html_path, "r", encoding="utf-8") as f:
                            html_data = f.read()
                            st.components.v1.html(html_data, height=400, scrolling=True)

                    # Layer Conductance
                    st.subheader("Layer Conductance")
                    lc_bar_path = result["visualizations"].get("lc_bars")
                    lc_comp_path = result["visualizations"].get("lc_comparison")
                    if lc_bar_path and os.path.exists(lc_bar_path):
                        st.image(lc_bar_path, caption="Layer Conductance – Token Attribution")
                    if lc_comp_path and os.path.exists(lc_comp_path):
                        st.image(lc_comp_path, caption="Layer Conductance – Comparison")
                    if not lc_bar_path and not lc_comp_path:
                        st.warning("Layer Conductance visualizations are not available.")

                    lc_html_path = result["visualizations"].get("lc_html")
                    if lc_html_path and os.path.exists(lc_html_path):
                        with open(lc_html_path, "r", encoding="utf-8") as f:
                            html_data = f.read()
                            st.components.v1.html(html_data, height=400, scrolling=True)

                    # Neuron Conductance (Retained)
                    st.subheader("Neuron Conductance")
                    nc_viz = result["visualizations"]
                    nc_images_found = False
                    
                    if nc_viz.get("nc_bar") and os.path.exists(nc_viz["nc_bar"]):
                        st.image(nc_viz["nc_bar"], caption="Neuron – Attribution Scores")
                        nc_images_found = True
                    if nc_viz.get("nc_activation") and os.path.exists(nc_viz["nc_activation"]):
                        st.image(nc_viz["nc_activation"], caption="Neuron – Attribution vs Activation")
                        nc_images_found = True
                    if nc_viz.get("nc_heatmap") and os.path.exists(nc_viz["nc_heatmap"]):
                        st.image(nc_viz["nc_heatmap"], caption="Neuron – Attribution Heatmap")
                        nc_images_found = True
                    
                    if not nc_images_found:
                        st.warning("Neuron Conductance visualizations are not available.")
                        
                    if nc_viz.get("nc_html") and os.path.exists(nc_viz["nc_html"]):
                        with open(nc_viz["nc_html"], "r", encoding="utf-8") as f:
                            html_data = f.read()
                            st.components.v1.html(html_data, height=400, scrolling=True)

                    # Kernel SHAP
                    st.subheader("Kernel SHAP")
                    ks_path = result["visualizations"].get("ks")
                    if ks_path and os.path.exists(ks_path):
                        st.image(ks_path, caption="Kernel SHAP – Token Attribution")
                    else:
                        st.warning("Kernel SHAP visualization is not available.")
                        
                else:
                    st.error(f"Backend error (Status {response.status_code}): {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API server. Please ensure the FastAPI server is running on http://localhost:8000")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

# Footer
st.markdown("---")
# Tip removed as per request

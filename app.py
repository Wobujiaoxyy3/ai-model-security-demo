import streamlit as st
from he_analysis import he_training_analysis
from dlg import deep_leakage_attack

# Set Streamlit Page Config
st.set_page_config(layout="wide", page_title="Security and Privacy in AI Model")

# Apply custom CSS
st.markdown("""
    <style>
    /* Sidebar style */
    [data-testid="stSidebar"] {
        background-color: #2C3E50 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] p {
        color: white !important;
    }
    
    /* Background */
    .main {
        background-color: #F5F7FA;
        padding: 20px;
        border-radius: 10px;
    }

    /* Title */
    .title {
        text-align: center;
        font-size: 26px;
        font-weight: bold;
        color: #2C3E50;
    }
    </style>
""", unsafe_allow_html=True)

# Create Sidebar
st.sidebar.title("Security and Privacy in AI Model")
page = st.sidebar.radio("Select Function", ["Introduction", "Model Inversion Attack", "HE Training Analysis", "Federated Learning Analysis"])

def show_intro():
    st.header("📖 Introduction")
    st.write("This app explores security and privacy issues in AI, including model inversion attacks, homomorphic encryption (HE) training analysis, and federated learning.")

def federated_learning_analysis():
    st.header("🌐 Federated Learning Analysis")
    st.write(
    "Here, we trained a simple MLP model for **fraud transaction detection**. During training, we implemented two different architectures:\
    **Centralized Learning** and **Federated Learning**, and analyzed the model's final performance.")


if page == "Introduction":
    show_intro()
elif page == "Model Inversion Attack":
    st.header("🛠️ Model Inversion Attack - Deep Leakage from Gradients (DLG)")
    with st.container():
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            deep_leakage_attack()
elif page == "HE Training Analysis":
    he_training_analysis()
elif page == "Federated Learning Analysis":
    federated_learning_analysis()

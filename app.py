import streamlit as st
from intro import intro
from he_analysis import he_training_analysis
from dlg import deep_leakage_attack
from fl_anaysis import federated_learning_analysis

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
        font-size: 60px;
        font-weight: bold;
        color: #2C3E50;
    }

    .sidebar-title {
        font-size: 60px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .sidebar-radio-label {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
    }

    .stRadio > div {
        background-color: #2b3e50;
        padding: 10px;
        border-radius: 10px;
    }

    .stRadio label {
        font-size: 16px;
        color: #ffffff;
        padding: 5px;
    }


    </style>
""", unsafe_allow_html=True)

# Create Sidebar
st.sidebar.title(" Security and Privacy in AI-driven digital banking")
# page = st.sidebar.radio("Select Function", ["Introduction", "Model Inversion Attack", "HE Training Analysis", "Federated Learning Analysis"])

# st.sidebar.markdown('<p class="sidebar-title">Security and Privacy in AI Model</p>', unsafe_allow_html=True)

# Sidebar selection
# st.sidebar.markdown('<p class="sidebar-radio-label">Select Function</p>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "",
    ["Introduction", "Model Inversion Attack", "HE Training Analysis", "Federated Learning Analysis"]
)


if page == "Introduction":
    intro()
elif page == "Model Inversion Attack":
    st.header("Model Inversion Attack - Deep Leakage from Gradients (DLG)")
    with st.container():
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            deep_leakage_attack()
elif page == "HE Training Analysis":
    st.header("🔐 Homomorphic Encryption Training Analysis")
    he_training_analysis()
elif page == "Federated Learning Analysis":
    st.header("Federated Learning Analysis")
    with st.container():
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            federated_learning_analysis()

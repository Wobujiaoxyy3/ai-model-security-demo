import streamlit as st
from page_units.intro import intro
from page_units.he_analysis import he_training_analysis
from page_units.dlg import deep_leakage_attack
from page_units.fl_analysis import federated_learning_analysis

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

    </style>
""", unsafe_allow_html=True)

# Create Sidebar
st.sidebar.title(" Security and Privacy in AI-driven digital banking")

page = st.sidebar.radio(
    "Placeholder",
    ["Introduction", "Model Inversion Attack", "HE Training Analysis", "Federated Learning Analysis"],
    label_visibility="collapsed"
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

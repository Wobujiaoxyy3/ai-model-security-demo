import streamlit as st
import pandas as pd
from PIL import Image

def federated_learning_analysis():

    st.write("""
    This page presents a fraud detection model trained for a digital bank, evaluated using two different approaches: **Centralized Learning (CL)** and **Federated Learning (FL)**. \
    The model's training results are compared and analyzed across both methods. The analysis includes the following aspects: 
    - **Federated Learning Training Diagram**
    - **Fraud Transaction Dataset Overview**
    - **CL vs FL Model Metrics**
    - **Training Curves(Loss & F1-score)**
    - **Analysis**
    """)

    # load FL training diagram
    diagram_path = "fl_analysis_results/Federated Learning Training Diagram.png"
    try:
        diagram = Image.open(diagram_path)
        st.image(diagram, caption="Federated Learning Training Diagram", use_container_width=True)
    except FileNotFoundError:
        st.warning("Diagram not found. Please check the file path.")

    # load fraud transaction dataset
    st.subheader("PaySim - Fraudulent Transaction Dataset")
    fradu_tx_csv_path = "data/fraud_tx_data.csv"
    fraud_tx_df = pd.read_csv(fradu_tx_csv_path)
    fraud_tx_df = fraud_tx_df.sample(n=100, random_state=42)
    st.dataframe(fraud_tx_df, height=250, use_container_width=True)

    # load classification reports
    cl_csv_path = "fl_analysis_results/cl_report.csv"
    fl_csv_path = "fl_analysis_results/fl_report.csv"

    try:
        # load CL report
        cl_report_df = pd.read_csv(cl_csv_path, index_col=0)
        cl_acc = cl_report_df.loc["accuracy", "f1-score"]
        cl_f1 = cl_report_df.loc["macro avg", "f1-score"]
        cl_report_df = cl_report_df.drop(index="accuracy")

        # load FL report
        fl_report_df = pd.read_csv(fl_csv_path, index_col=0)
        fl_acc = fl_report_df.loc["accuracy", "f1-score"]
        fl_f1 = fl_report_df.loc["macro avg", "f1-score"]
        fl_report_df = fl_report_df.drop(index="accuracy")

        # CL & FL Reports Comparison
        st.subheader("CL vs FL Classification Metrics")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h5 style='text-align: left;'>Centralized Learning Metrics</h5>", unsafe_allow_html=True)
            st.write(f"**Accuracy: {cl_acc:.4f} &emsp;&emsp; F1-score: {cl_f1:.4f}**", unsafe_allow_html=True)
            st.dataframe(cl_report_df, use_container_width=True)

        with col2:
            st.markdown("<h5 style='text-align: left;'>Federated Learning Metrics</h5>", unsafe_allow_html=True)
            st.write(f"**Accuracy: {fl_acc:.4f} &emsp;&emsp; F1-score: {fl_f1:.4f}**", unsafe_allow_html=True)
            st.dataframe(fl_report_df, use_container_width=True)

    except FileNotFoundError:
        st.warning("CSV files not found. Please check the file path.")

    # load training curves
    loss_curve_path = "fl_analysis_results/loss_curve.png"
    f1_score_curve_path = "fl_analysis_results/f1-score_curve.png"

    st.subheader("Training Curves")
    col3, col4 = st.columns(2)

    try:
        with col3:
            loss_curve = Image.open(loss_curve_path)
            st.image(loss_curve, caption="Loss Curve", use_container_width=True)

        with col4:
            f1_score_curve = Image.open(f1_score_curve_path)
            st.image(f1_score_curve, caption="F1-score Curve", use_container_width=True)

    except FileNotFoundError:
        st.warning("Training curves not found. Please check the file path.")
    
    
    st.subheader("Analysis")
    st.write("From the above metrics and training curves, we can observe that while federated learning allows different bank branches to collaboratively train a model without sharing their own data, \
        the final model trained through federated learning is less accurate and has a lower F1 score compared to the centrally trained model. In particular, the recall and precision for fraudulent transactions  \
        are significantly lower than those of the centrally trained model, highlighting certain disadvantages in training effectiveness compared to conventional methods.")
import streamlit as st

def intro():


    st.title("AI Security & Privacy Risks in Digital Banking")

    st.markdown("""
    This page explores the **security and privacy risks of AI-driven digital banking systems** and examines various strategies to enhance the security of financial technology applications.
    """)

    st.subheader("🔹 AI Models in Digital Banking")
    
    st.markdown("""
    In this study, we implemented two key AI models for the digital banking system:

    - **Facial Recognition Model** (for **KYC (Know Your Customer)** and **identity verification**)  
      Ensures the authenticity of user identities, preventing malicious account registration and unauthorized access.  

    - **Fraud Transaction Detection Model**  
      Leverages AI to identify suspicious transactions and protect digital banking users from financial fraud.
    """)

    st.subheader("📌 Three Key Demonstrations")

    st.markdown("""
    This page contains **three dedicated subpages**, which you can navigate through the **Sidebar** on the left.  \
    Each subpage demonstrates a different **AI security risk and its corresponding defense mechanisms**, including:
    """)

    # Use bullet points for better readability
    st.markdown("""
    1️⃣ **Model Inversion Attack**  
       Demonstrates how a facial recognition model can be exploited through gradient leakage attacks and explores potential defense strategies.  

    2️⃣ **Homomorphic Encryption Training Analysis**  
       Analyzes the impact of **homomorphic encryption** on model training time and gradient storage efficiency, showcasing privacy-preserving training techniques.  

    3️⃣ **Federated Learning Analysis**  
       Examines the training process of a fraud detection model using federated learning and compares its performance against a centrally trained model.
    """)



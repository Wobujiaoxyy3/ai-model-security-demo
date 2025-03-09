# AI Model Security Demo

This project provides a demonstration of **AI model security**, focusing on **data leakage risks** and **privacy-preserving training techniques**. The demo is divided into three main parts:

## Project Overview

1. **Deep Leakage from Gradients (DLG) Attack**  
   - Demonstrates how an attacker can reconstruct original training data from gradients using a **gradient inversion attack**.  
   - Users can select an image from the dataset and observe how easily an adversary can recover training data.  
   - The implementation follows the **Deep Leakage from Gradients (DLG) framework**.

2. **Homomorphic Encryption (HE) in Model Training**  
   - Explores the impact of **Homomorphic Encryption** on training efficiency and storage overhead.  
   - Compares training time and gradient storage with and without encryption.  
   - Helps users understand the trade-off between **data security and computational cost**.

3. **Federated Learning & Privacy Protection** 
   - Simulates **federated learning** and evaluates its security against gradient leakage attacks.  
   - Demonstrates how techniques like **differential privacy** can mitigate risks.

---

## Getting Started

### 1️⃣ Clone the Repo
```bash
git clone <repo-url>
cd <repo-name>
```

### 2️⃣ Create a Virtual Environment (Recommended)
It's recommended to use conda to  manage dependencies:
```bash
conda create -n ai_security_demo python=3.9
conda activate ai_security_demo
```

### 3️⃣ Install Dependencies
``` bash
pip install -r requirements.txt
```

### Run the Streamlit App
``` bash
streamlit run app.py
```
This will launch the web interface for interacting with the demo.

---

## Reference
The implementation of the **DLG Attack** is inspired by [DLG repo](https://github.com/mit-han-lab/dlg?tab=readme-ov-file), implemented by [MIT Han Lab](https://github.com/mit-han-lab). 

You can also refer to the paper **Deep Leakage from Gradients**: [[arXiv]](https://arxiv.org/abs/1906.08935)  [[Website]](https://dlg.mit.edu)  

The implementation of **Federated Learning** refers to [Federated-Learning-PyTorch](https://github.com/AshwinRJ/Federated-Learning-PyTorch).

## License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.
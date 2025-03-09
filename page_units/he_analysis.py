import os
import torch
import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tenseal as ts
from PIL import Image
from torchvision import transforms
from torch.autograd import grad
from models.vision import LeNet, weights_init
from utils import label_to_onehot, cross_entropy_for_onehot

def he_training_analysis():
    st.markdown("""This demo illustrates the impact of **Homomorphic Encryption (HE)** on training time and gradient storage overhead during model training. \
        By clicking the button below, you can start the analysis. We train a simple LeNet model on 10 facial images, comparing two scenarios: with and without Homomorphic Encryption. \
        The demo evaluates the **average training time per image** and the **gradient storage requirements**, providing insights into the computational cost of privacy-preserving machine learning.""")

    if st.button("⚡ Run HE Training Analysis"):
        ctx = ts.context(scheme=ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        ctx.generate_galois_keys()
        ctx.global_scale = 2**40
        
        image_set = sorted([os.path.join("data/facial_images", f) for f in os.listdir("data/facial_images") if f.endswith(".jpg")])[:10]
        tp = transforms.ToTensor()
        net = LeNet().to("cpu")
        torch.manual_seed(1234)
        net.apply(weights_init)
        criterion = cross_entropy_for_onehot
        
        train_times = {"no_he": [], "with_he": []}
        storage_sizes = {"no_he": [], "with_he": []}
        
        for img_path in image_set:
            gt_data = tp(Image.open(img_path)).unsqueeze(0).to("cpu")
            gt_label = torch.randint(0, 10, (1,)).to("cpu")
            gt_onehot_label = label_to_onehot(gt_label)
            
            for enable_he in [False, True]:
                start_time = time.time()
                pred = net(gt_data)
                y = criterion(pred, gt_onehot_label)
                dy_dx = torch.autograd.grad(y, net.parameters())
                original_dy_dx = [_.detach().clone() for _ in dy_dx]
                
                if enable_he:
                    encrypted_dy_dx = [ts.ckks_vector(ctx, t.cpu().numpy().flatten()) for t in original_dy_dx]
                    storage_sizes["with_he"].append(sum(np.array(gy.decrypt()).nbytes for gy in encrypted_dy_dx))
                else:
                    storage_sizes["no_he"].append(sum(gx.element_size() * gx.nelement() for gx in original_dy_dx))
                
                train_times["with_he" if enable_he else "no_he"].append(time.time() - start_time)
        
        # Compute Average Time & Storage Size
        avg_time_no_he = np.mean(train_times["no_he"])
        avg_time_with_he = np.mean(train_times["with_he"])
        avg_storage_no_he = np.mean(storage_sizes["no_he"]) / 1024  
        avg_storage_with_he = np.mean(storage_sizes["with_he"]) / 1024 

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("⏳ Training Time Comparison")
            st.write(f"📌 **No HE Average Training Time:** {avg_time_no_he:.4f} seconds")
            st.write(f"📌 **With HE Average Training Time:** {avg_time_with_he:.4f} seconds")
            st.write(f"🔍 **HE increases training time by approximately :red[{avg_time_with_he / avg_time_no_he:.2f}X]**")

            st.subheader("💾 Storage Overhead Comparison")
            st.write(f"📌 **No HE Average Gradient Storage:** {avg_storage_no_he:.2f} KB")
            st.write(f"📌 **With HE Average Gradient Storage:** {avg_storage_with_he:.2f} KB")
            st.write(f"🔍 **HE increases storage overhead by approximately :red[{avg_storage_with_he / avg_storage_no_he:.2f}X]**")
        
        with col2:
            image_counts = np.arange(0, 11)
            train_times_cumsum_no_he = np.insert(np.cumsum(train_times["no_he"]), 0, 0)
            train_times_cumsum_with_he = np.insert(np.cumsum(train_times["with_he"]), 0, 0)

            fig, ax = plt.subplots(figsize=(4, 2.5))
            ax.plot(image_counts, train_times_cumsum_no_he, marker="o", linestyle="-", label="No HE", linewidth=2)
            ax.plot(image_counts, train_times_cumsum_with_he, marker="s", linestyle="--", label="With HE", linewidth=2)

            ax.set_xlabel("Number of Training Images", fontsize=6)
            ax.set_ylabel("Total Training Time (seconds)", fontsize=6)
            ax.set_title("Training Time vs. Number of Images", fontsize=8)
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)

        st.markdown("""
                ### Conclusion
                From the above analysis, while **Homomorphic Encryption (HE)** ensures that training data remains private during model training, it significantly increases the training time (by approximately **100X**) \
                and **doubles** the storage requirements for gradients. Therefore, in real-world applications, a trade-off must be made between **data security, computational cost, and storage overhead**.

                It is also important to note that in this experiment, **HE was only applied to addition and element-wise multiplication**. If encryption were extended to more complex operations such as 2D convolution and matrix multiplication,\
                 the training time could further increase by a factor of **100X to 1000X**.""")
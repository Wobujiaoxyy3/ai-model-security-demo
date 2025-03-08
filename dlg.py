import streamlit as st
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torchvision import transforms
from models.vision import LeNet, weights_init
from utils import label_to_onehot, cross_entropy_for_onehot

def deep_leakage_attack():
    data_dir = "data"
    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".jpg")])
    if not image_files:
        st.error("❌ No images found, please check the data directory!")
        return
    
    st.write("The following demo demonstrates a scenario where a hacker uses a gradient inversion attack to reconstruct the original training data. \
        By leveraging the gradients generated during model training, the hacker can recover an image from the training dataset using a randomly generated image.")

    st.write('''You can select a random image from the dataset and click **Run** to launch the attack. \
        You may also enable **Gradient Differential Privacy** to add noise to the gradients. \
        Then, run the attack again to check if this protection strategy effectively safeguards the training data.''')

    st.write("**Select image index (1-500)**")
    with st.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            img_index = st.number_input("", min_value=1, max_value=500, value=66, step=1, label_visibility="collapsed")
        with col2:
            run_button = st.button("▶ Run", use_container_width=True)
        with col3:
            dp_button = st.toggle("🔏 **Gradient Differential Privacy**")
    
    if run_button:
        image_path = os.path.join(data_dir, f"{img_index}.jpg")
        tp = transforms.ToTensor()
        tt = transforms.ToPILImage()
        gt_data = tp(Image.open(image_path).convert("RGB")).unsqueeze(0).to("cpu")
        gt_label = torch.randint(0, 100, (1,)).long().to("cpu")
        gt_onehot_label = label_to_onehot(gt_label)

        net = LeNet().to("cpu")
        torch.manual_seed(1234)
        net.apply(weights_init)
        criterion = cross_entropy_for_onehot
        
        pred = net(gt_data)
        y = criterion(pred, gt_onehot_label)
        dy_dx = torch.autograd.grad(y, net.parameters())
        original_dy_dx = [_.detach().clone() for _ in dy_dx]
        
        noise_scale = 0.01
        noisy_dy_dx = [grad + noise_scale * torch.randn_like(grad) for grad in original_dy_dx]
        
        if dp_button:
            st.markdown("**:red[Gradient noise added!!]**")
        else:
            st.write("**Original gradient**")

        dummy_data = torch.randn(gt_data.size()).to("cpu").requires_grad_(True)
        dummy_label = torch.randn(gt_onehot_label.size()).to("cpu").requires_grad_(True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
        
        st.markdown("<h3 style='text-align: center;'>Recovery Progress</h3>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(6, 2))

        axes[0].imshow(tt(dummy_data[0].cpu()))
        axes[0].set_title("Random Init")
        axes[0].axis("off")

        progress_bar = st.progress(0)
        loss_placeholder = st.empty()
        img_placeholder = st.empty()
        
        for iters in range(300):
            def closure():
                optimizer.zero_grad()
                dummy_pred = net(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = criterion(dummy_pred, dummy_onehot_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

                grad_diff = 0
                for gx, gy_noisy, gy_clean in zip(dummy_dy_dx, noisy_dy_dx, original_dy_dx):
                    if dp_button:
                        grad_diff += ((gx - gy_noisy) ** 2).sum()
                    else:
                        grad_diff += ((gx - gy_clean) ** 2).sum()
                
                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)

            if iters % 10 == 0:
                current_loss = closure()
                l2_loss = ((dummy_data - gt_data) ** 2).sum().item()
                progress_bar.progress((iters + 1) / 300)

                axes[1].imshow(tt(dummy_data[0].cpu()))
                axes[1].set_title("Recovered")
                axes[1].axis("off")

                axes[2].imshow(tt(gt_data[0].cpu()))
                axes[2].set_title("Ground Truth")
                axes[2].axis("off")

                img_placeholder.pyplot(fig)
                
                loss_placeholder.markdown(
                    f"<h4 style='text-align: center;'>GLoss: {current_loss.item():.4f} &nbsp;&nbsp; | &nbsp;&nbsp; L2Loss: {l2_loss:.4f}</h4>",
                    unsafe_allow_html=True
                )
        if current_loss.item() < 0.01:
            st.success("✅ Recovery Successful!")
        else:
            st.error("❌ Recovery Failed!")
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Skin Cancer Detection",
    layout="centered"
)

st.title("Skin Cancer Detection")

st.markdown(
    """
**โครงงานนี้เป็นส่วนหนึ่งของรายวิชา ว31104 เทคโนโลยี 2 (วิทยาการคำนวณ)**
ภาคเรียนที่ 2 ปีการศึกษา 2568
โรงเรียนสาธิตมหาวิทยาลัยศรีนครินทรวิโรฒ ปทุมวัน
"""
)

# ===============================
# CONFIG
# ===============================
IMG_SIZE = 224
THRESHOLD = 0.4   # เน้น recall (medical-safe)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# MODEL DEFINITION
# ===============================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class SkinCancerCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            SEBlock(32)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            SEBlock(64)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            SEBlock(128)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        # ⭐ MUST MATCH TRAINING ARCHITECTURE
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)  # logits (NO sigmoid)


# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    model = SkinCancerCNN().to(DEVICE)
    state = torch.load("skin_cancer_model.pth", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


model = load_model()

# ===============================
# IMAGE PREPROCESSING
# ===============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image: Image.Image):
    img = transform(image).unsqueeze(0)
    return img.to(DEVICE)


# ===============================
# FILE UPLOADER
# ===============================
uploaded_file = st.file_uploader(
    "อัปโหลดภาพรอยโรคผิวหนัง (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="ภาพที่อัปโหลด", use_column_width=True)

    x = preprocess_image(image)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    st.subheader("ผลการทำนาย")

    if prob >= THRESHOLD:
        st.error("**Malignant (มีความเสี่ยงเป็นมะเร็งผิวหนัง)**")
    else:
        st.success("**Benign (ไม่พบความเสี่ยงสูง)**")

    st.write(f"**ความน่าจะเป็นที่เป็นมะเร็ง:** `{prob:.2f}`")

    st.info(
        "โมเดลนี้ใช้สำหรับการคัดกรองเบื้องต้นเท่านั้น "
        "ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้ "
        "หากมีความกังวล กรุณาปรึกษาแพทย์ผู้เชี่ยวชาญ"
    )

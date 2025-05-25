# 🦷 Mesiodens Detection from Periapical Radiographs

This project provides a web-based application for detecting **mesiodens** (supernumerary teeth) from **periapical dental radiographs** using a custom-trained **YOLOv5/YOLOv8** model. The interface is built with **Streamlit**, offering an easy and fast way to test dental X-ray images and get automated detection results.

---

## 🔬 Project Overview

Mesiodens are extra teeth found in the midline between the two central incisors, often causing misalignment or eruption issues. Manual detection in radiographs can be time-consuming and subjective. This tool uses deep learning to assist dentists and radiologists in identifying mesiodens automatically.

---

## 🚀 Features

- 📷 Upload periapical X-ray images (`.jpg`, `.jpeg`, `.png`)
- 🤖 Automatic detection of mesiodens using a YOLO-based deep learning model
- 📊 View bounding boxes and detection confidence
- 🧾 Outputs detection details (coordinates, confidence score)

---

## 🧠 Model Details

- Architecture: **YOLOv5 / YOLOv8**
- File format: PyTorch `.pt` (e.g., `best.pt`)
- Trained on: Labeled dataset of periapical radiographs with mesiodens annotations
- Output: Bounding boxes with class and confidence scores

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mesiodens-detection.git
cd mesiodens-detection
```
### 2. Set Up Virtual Environment (Recommended)

```
python -m venv venv
```

#### On Windows:

```
venv\Scripts\activate
```
#### On macOS/Linux:

```
source venv/bin/activate
```

### 3. Install Requirements

```
pip install -r requirements.txt
```
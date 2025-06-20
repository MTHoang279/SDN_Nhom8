# DDoS Detection and Mitigation with Ryu + Mininet

## Description

This project uses a machine learning model trained on the 2018 IDS dataset (Kaggle) to detect and mitigate DDoS attacks in an SDN environment powered by **Ryu controller** and **Mininet**. The repository provides two modes of operation:

- **Detection only**: `decection.py`  
- **Detection & Mitigation**: `mitigation.py`

---

## Prerequisites

### 1. Install Ryu and Mininet

```bash
sudo apt update
sudo apt install mininet
pip install ryu pandas numpy scikit-learn joblib requests
```

Alternatively, you can use the official Mininet VM:  
https://github.com/mininet/mininet

---

## Data Preparation

### 1. Download the dataset

Download the CSV file from Kaggle:  
https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv?select=02-21-2018.csv

### 2. Preprocess the data

Run the feature-extraction script:

```bash
python3 /path_to/features.py
```

This will produce `processed_data.csv`, which contains the cleaned features and labels required for training the Random Forest model.

---

## Usage

### Mode 1: DDoS **Detection Only**

1. **Start the Ryu controller**:  
   ```bash
   ryu-manager /path_to/detection.py
   ```
2. **Launch Mininet topology** (in a second terminal):  
   ```bash
   sudo python3 /path_to/generate.py
   ```
3. In the automatically opened xterm for host **h2**, launch the attack script:  
   ```bash
   sudo python3 /path_to/hoic.py
   ```
4. Watch the first terminal (running `detection.py`) to see DDoS detection logs.

---

### Mode 2: DDoS **Detection & Mitigation**

1. **Start the Ryu controller** with mitigation logic:  
   ```bash
   ryu-manager /path_to/mitigation.py
   ```
2. **Launch Mininet topology** (in a second terminal):  
   ```bash
   sudo python3 /path_to/generate.py
   ```
3. In the xterm for **h2**, run the attack:  
   ```bash
   sudo python3 /path_to/hoic.py
   ```
4. The `mitigation.py` controller will log detected attacks and automatically apply rate-limiting to the victim’s IP.

---

## Requirements

- **Python 3.9 or lower**  
- **Ryu SDN Framework**  
- **Mininet**  
- Python libraries: `pandas`, `numpy`, `scikit-learn`, `joblib`, `requests`

Install missing libraries with:

```bash
pip install pandas numpy scikit-learn joblib requests
```

---

## Notes

- If `best_model.pkl` is missing, the controller will automatically train a new model from `processed_data.csv`.
- Ensure all `.py` files reside in the same directory.
- Update file paths in `features.py`, `detection.py`, and `mitigation.py` if your environment differs.

---

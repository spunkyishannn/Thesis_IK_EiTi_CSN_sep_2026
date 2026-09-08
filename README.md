# Network Intrusion Detection Using Machine Learning (MSc Thesis)

* **Author:** Ishan Kaushik
* **Student ID:** 303029
* **University:** Warsaw University of Technology (Politechnika Warszawska)
* **Faculty:** Faculty of Electronics and Information Technology (EiTi)
* **Specialization:** Computer Systems and Networks (CSN)
* **Supervisor:** dr inż. Joanna Żarnecka
* **Date:** September 2026

---

> [!WARNING]
> ### Dataset Size and Storage Notice (~400 GB)
> 
> ![Dataset Size](https://img.shields.io/badge/Dataset%20Size-~400%20GB-orange) ![Storage](https://img.shields.io/badge/Storage-Offline%20Archive-blue)
> 
> The complete dataset (raw PCAP captures, network flows, and processed files) takes around **400 GB** in total. Because of GitHub's file size limits, the large dataset files are stored offline and not uploaded here. The `data/` folder structure is kept in the repository so you can place the files locally if you want to run the code.

---

## What This Project Is About

This repository has the source code and configuration files for my Master's thesis.

The goal of the thesis is to see how well machine learning models can detect network attacks when tested on network setups different from what they were trained on:
* Models usually score very high when tested on the same network they learned from.
* But when you test them on a different network or an external dataset like CICIDS2017, the score drops significantly.
* This project tests 5 different ML models across 5 test stages to measure this performance drop and see what causes it.

---

## Folder Structure

* `lab/`: Docker setup (`docker-compose.yml`) used to create the testbed network, traffic generators, and labeling scripts.
* `configs/`: Feature list (`feature_dictionary_v2.yaml`), split settings, and tuned parameters for models.
* `src/features/`: Scripts that check for correlated features and measure how traffic distributions shift between setups.
* `src/models/`: Python scripts to split the data, train models, tune hyperparameters, run cross-domain tests, and generate plots.
* `reports/`: Saved test results and scores in JSON format.
* `data/`: Empty folder where local dataset files go.

---

## Attacks Tested

* **DoS and DDoS:** HTTP Hulk, Slowloris, TCP SYN flood, UDP flood, ICMP flood
* **Brute Force:** SSH brute force, FTP brute force
* **Scans:** TCP port scan (Nmap), version scan
* **Normal (Benign) Traffic:** Regular web browsing (HTTP/HTTPS), DNS lookups, SSH, and FTP file transfers

---

## Machine Learning Models Used

* Random Forest
* XGBoost
* LightGBM
* Logistic Regression
* Linear SVC
* Soft-voting ensemble (combining the predictions of multiple models)

---

## Software and Tools

* **Python 3.11** (scikit-learn, xgboost, lightgbm, optuna, pandas, numpy)
* **Network & Lab:** Docker, Docker Compose, Linux traffic control (`tc`), Scapy, hping3, Nmap, Hydra
* **Flow Extraction:** CICFlowMeter
* **Plotting:** Matplotlib, Seaborn

---

## How to Run

1. **Set up the Python environment:**
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

2. **Feature selection:**
```bash
python src/features/phase1_feature_analysis.py
```

3. **Split the dataset by session:**
```bash
python src/models/build_primary_split.py
```

4. **Train and evaluate across domains:**
```bash
python src/models/run_lodo_evaluation.py
```

5. **Test on external benchmark (CICIDS2017):**
```bash
python src/models/evaluate_cicids2017.py
```

6. **Generate figures for the thesis:**
```bash
python src/models/generate_thesis_figures.py
```
All generated charts are saved to `reports/figures/`.

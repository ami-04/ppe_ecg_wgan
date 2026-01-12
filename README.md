# PPG-to-ECG Reconstruction Using Deep Learning

This repository contains code for reconstructing **ECG signals from PPG signals** using deep learning models. The project compares a **Generative Adversarial Network (WGAN-GP)** with a **Bidirectional LSTM (BiLSTM)** to study their performance in signal reconstruction tasks relevant to **wearable health monitoring devices**.

---

## 📌 Project Overview

Electrocardiogram (ECG) monitoring is critical for cardiovascular health, but continuous ECG measurement can be cumbersome. Photoplethysmography (PPG), commonly available in wearable devices, offers an indirect signal. This project explores **reconstructing ECG signals from PPG** using two approaches:

1. **WGAN-GP** – focuses on generating ECG signals that look realistic and preserve morphology.
2. **BiLSTM** – deterministic regression model that optimizes pointwise accuracy.

The aim is to find a model suitable for **real-time, wearable applications**, balancing **accuracy, computational efficiency, and waveform fidelity**.

---

## 🧰 Features

- ECG reconstruction from PPG signals
- Comparison of **WGAN-GP** and **BiLSTM** models
- MSE evaluation and visual signal comparison
- Support for training and evaluation pipelines
- Modular data handling for PPG, ECG, and ECG peaks

---

## 📂 Repository Structure


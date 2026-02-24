# 🛠️ The PRESTO Processing Pipeline

This folder contains the custom Python wrapper designed to automate the **PRESTO** software suite. It transforms raw radio telescope data (`.sf` format) into visual candidate plots ready for analysis.

## 🚀 Overview
Standard pulsar searching is a manual, CLI-heavy process. This pipeline automates the workflow into a seamless Python script, handling everything from RFI mitigation to final folding.

## ⚙️ The 5-Step Logic
The pipeline executes the following stages sequentially:

1.  **RFI Mitigation (`rfifind`):**
    *   Scans data chunks for man-made interference (satellites, radar).
    *   *Innovation:* Implements **Adaptive Masking** (automatically relaxing thresholds for ultra-bright sources like J0437 to prevent signal deletion).
2.  **De-dispersion (`DDplan` + `prepsubband`):**
    *   Corrects for the time delay caused by interstellar electrons (Dispersion Measure).
    *   Generates thousands of time-series `.dat` files at different DMs.
3.  **Universal Grid Search (`accelsearch`):**
    *   Instead of guessing the pulsar type, we run a **3-Pronged Grid Search** on *every* file (see below).
4.  **Sifting (`ACCEL_sift.py`):**
    *   Filters out harmonics and candidates peaking at DM 0 (Earth noise).
    *   *Source Code Hack:* We modified the source code to lower the `min_DM` threshold, allowing detection of nearby pulsars (like PSR J0437 at DM 2.64) that standard pipelines ignore.
5.  **Folding (`prepfold`):**
    *   Stacks the data based on the best period/DM to produce the final diagnostic plots.

## 🎯 The "Universal Grid" Strategy
To ensure this pipeline works for both slow, isolated pulsars and fast, binary millisecond pulsars (MSPs), we process every file using three concurrent strategies:

| Strategy | ZMAX (Accel) | Harmonics | Target |
| :--- | :--- | :--- | :--- |
| **Iso_Slow** | 0 | 8 | Normal, isolated pulsars. |
| **Binary** | 100 | 8 | Standard binary systems. |
| **MSP_Fast** | 200 | 16 | Fast MSPs & "Black Widow" binaries. |

## 📦 Requirements
*   **OS:** Linux (Ubuntu 20.04+) or WSL2.
*   **Software:** [PRESTO v3.0](https://github.com/scottransom/presto).
*   **Environment:** `PRESTO` and `LD_LIBRARY_PATH` variables must be set.

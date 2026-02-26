# Automated Pulsar Discovery & AI Classification

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fgeosh0%2FPulsar-Detection-and-AI-Classification&title=Views)](https://hits.seeyoufarm.com)

An end-to-end pipeline for detecting and classifying pulsars, moving from raw radio telescope voltage data to Deep Learning classification.

## 🚀 Introduction to the project

  In the field of Data Science, it is common to start projects with "clean," pre-processed datasets (like the famous HTRU-2 CSV found on UCI or Kaggle). While convenient, this skips the most challenging and scientifically significant part of the process: The raw data processing so instead of using existing feature tables, raw binary search files (.sf) were sourced directly from the CSIRO Data Access Portal. Next step was building the processing pipeline to handle the raw radio data, up to generating the candidates profiles and the classification of those candidates using AI. 
  
## 🌌  The Science: Finding Needles in the Cosmic Haystack

 ### What is a Pulsar?
 Pulsars are the rapidly spinning, highly magnetized cores of dead massive stars. They emit beams of radio waves that sweep across the earth like a cosmic lighthouse.

### The Challenge
Radio telescopes produce terabytes of noisy data. The signal of interest is often buried orders of magnitude below the noise floor, obscured by:

* Terrestrial RFI: Satellites, lightning, and power grids.
* Interstellar Medium (ISM): Free electrons in space delay lower frequencies, causing signal "smearing" (Dispersion).

To recover the signal, we must apply precise mathematical transformations to align frequencies across time (De-dispersion) and stack thousands of pulses to build a visible profile (Folding).

## 🔭  Data Acquisition & Navigation
Data was utilized from the Parkes Radio Telescope in Australia, specifically the HTRU Survey (High Time Resolution Universe - Project P630). 
Navigating the CSIRO archives is non-trivial and raw search data is massive (often >1GB per 9-minute observation). Due to these computational and storage constraints, the entire HTRU survey was not processed rather there was a focus on specific semesters of the P630 Project. A targeted search strategy was applied with selecting and processing two distinct raw data files (.sf) not only to calibrate the Presto pipeline but also to represent the differences of the pulsar population:

* PSR J0437-4715 (The Millisecond Pulsar): We approximately located the raw data file containing the brightest MSP in the southern sky to test our pipeline's ability to handle millisecond-level periods, binary orbital acceleration, and low dispersion measures.
* PSR J1048-5832 (The Normal Pulsar): We located a target deep in the Galactic Plane to validate the pipeline's performance against High Dispersion Measure (DM) smearing and significant galactic background noise.

NOTE: Detailed documentation on how to locate these specific files is available in docs/How_to_Search_in_CSIRO.txt.

## 🛠️ The Presto Processing Pipeline
We implemented a Python-wrapped automation of the PRESTO software (the industry-standard in pulsar astronomy). The pipeline consists of five stages, starts with the raw .sf voltage data and transforms them into the candidate plots:
* RFI Mitigation (rfifind): Statistical analysis of time/frequency blocks to mask man-made interference.
* De-dispersion Plan (DDplan & prepsubband): Calculating optimal Dispersion Measure (DM) steps to correct for interstellar delays without losing time resolution.
* Acceleration Search (accelsearch): FFT-based search in the Fourier domain, optimizing for binary orbital acceleration (zmax) and millisecond spin periods (numharm).
* Candidate Sifting (ACCEL_sift): Heuristic filtering to remove candidates that peak at DM=0 (Earth-based signals).
* Folding (prepfold): Generating the candidate profiles from which we focus on the diagnostic plots (Pulse Profile, Time vs. Phase, Frequency vs. Phase) for AI consumption.

### Key Changes in our Pipeline:
A core challenge in pulsar astronomy is that different types of pulsars (Slow vs. Fast) usually require different search parameters. While we selected files containing known targets, we designed the pipeline to treat every file as a "Blind Search." We did not tune the search parameters specifically for J0437 or J1048. Instead, we implemented a universal architecture that runs exhaustively on every file.

* Universal Grid Search: Instead of guessing parameters based on metadata, our pipeline runs three concurrent search strategies on every file:
  * Iso_Slow: Optimized for isolated, slow pulsars.
  * Binary: Optimized for standard binary systems.
  * MSP_Fast: Optimized for fast millisecond pulsars (High Z-max, High Harmonics).
* Source Code Modification: Standard sifting algorithms filter out candidates with DM < 2.0. Since our target (PSR J0437) has a DM of 2.64, we modified the ACCEL_sift.py source code to lower the threshold, successfully recovering the pulsar where standard tools failed.
* Adaptive RFI Masking: Implemented logic to relax RFI masking thresholds for ultra-bright sources, preventing the pipeline from mistaking strong pulsar signals for interference.

## 🧠 The Deep Learning Classifier
Once the raw data was processed into candidate plots, the Computer Vision pipeline was build to classify them.
### Dataset Construction
A Hybrid Dataset was contructed to ensure robustness and prevent overfitting:
* Internal Data: Candidates generated by our pipeline.
* External Data: We integrated candidate profiles of discovred pulsars from the PALFA (Arecibo) and GBNCC (Green Bank) surveys to increase the diversity of positive samples.

### The "Hydra" Architecture
We avoided standard "Black Box" CNNs. Instead, we implemented a Multi-Stream CNN (inspired by the PICS architecture). The model analyzes four distinct visual components of a candidate simultaneously:
1. Pulse Profile: (1D Shape analysis).
2. Time-Phase Plot: (Continuity analysis).
3. Frequency-Phase Plot: (Broadband signal analysis).
4. DM Curve: (Interstellar dispersion analysis).
These four "Heads" concatenate into a final dense layer to output a probability score.

## 📚 Bibliography & References

This project builds upon the foundational work of the radio astronomy and data science communities.

### The Survey & Data
*   **HTRU Survey:** Keith, M. J., et al. ["The High Time Resolution Universe Pulsar Survey - I. System configuration and initial discoveries"](https://academic.oup.com/mnras/article/409/2/619/1037409). *Monthly Notices of the Royal Astronomical Society* (2010).
    *   *Context:* Defines the specifications of the Parkes Multibeam Receiver and the survey parameters used for our pipeline tuning.
*   **Data Source:** [CSIRO Data Access Portal](https://data.csiro.au).
    *   *Context:* The official repository for the raw `.sf` search mode files used in this project.

### The Software
This project relies on the open-source software package **PRESTO**.
*   **Core Algorithms:** Ransom, S. M. ["New search techniques for binary pulsars"](https://ui.adsabs.harvard.edu/abs/2001PhDT.......123R/abstract). *Ph.D. Thesis, Harvard University* (2001).
*   **Source Code:** [PRESTO on GitHub](https://github.com/scottransom/presto).

### Machine Learning Methodology
*   **PICS (Image-Based Classification):** Zhu, W. W., et al. ["Searching for pulsars using image pattern recognition"](https://iopscience.iop.org/article/10.1088/0004-637X/781/2/117). *The Astrophysical Journal* (2014).
    *   *Context:* The primary inspiration for our Computer Vision approach (treating folded plots as images).
*   **Class Imbalance Strategy:** Lyon, R. J., et al. ["Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach"](https://academic.oup.com/mnras/article/459/1/746/2609075). *Monthly Notices of the Royal Astronomical Society* (2016).

### External Datasets (Training Data)
To solve the "Small Data" problem inherent in single-user processing, we augmented our training set with candidate profiles from established surveys:
*   **PALFA Survey:** The ALFA Pulsar survey at the Arecibo Observatory.
    *   [PALFA Survey Portal](https://palfa.nanograv.org/)
*   **GBNCC Survey:** The Green Bank North Celestial Cap survey.
    *   [GBNCC Discovered Pulsars](https://astro.phys.wvu.edu/GBNCC/)


## 🚧 Note from the Author:
This repository is the result of a research project created for educational purposes and the joy of discovery. While we successfully managed to hunt down pulsars like J0437 and J1048, I am not a tenured radio astronomer. There are likely edge cases, parameter tunings, or theoretical nuances that I have missed. If you find a mistake in the physics or the code, please open an Issue or a Pull Request—I am here to learn!  

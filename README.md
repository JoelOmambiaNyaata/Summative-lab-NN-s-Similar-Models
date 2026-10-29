# EcoSort: Integrated Waste Management Assistant
## Overview
This repository contains the complete implementation for the **Summative Lab: NNs and Similar Models**. The project builds an intelligent, multi-modal waste management assistant for "EcoSort" designed to help residents correctly classify waste items and receive accurate municipal recycling instructions.

---

## Project Structure & Architecture

The system integrates three core AI components:
1. **CNN Image Classifier (Part 2)**: Utilizes transfer learning via **MobileNetV2** to classify uploaded images of waste items into 8 specific categories (paper, cardboard, biological, metal, plastic, glass types).
2. **Text Classification Pipeline (Part 3)**: Implements text preprocessing and a robust machine learning classifier (Random Forest with TF-IDF features) to categorize textual descriptions of waste.
3. **RAG Retrieval Engine (Part 4)**: Implements a Retrieval-Augmented Generation (RAG) system using cosine similarity over vectorized municipal policy documents to source accurate disposal regulations.
4. **Integrated Assistant (Part 5)**: A unified wrapper function (`waste_management_assistant`) that routes either image or text inputs through the respective classification pipelines and returns structured recycling guidelines.

---

## Dataset Requirements
Ensure the following files are placed in your working directory before running the notebook:
* `RealWaste.zip` (Automatically extracted by the notebook code into the `RealWaste/` directory containing image subfolders).
* Here is the link to the .zip file ; https://drive.google.com/file/d/12s_lRQgXtoZeUXfVVuEhTAfAEjtL_CSq/view?usp=sharing
* `waste_descriptions.csv`.
* `waste_policy_documents.csv` (or `.json`).

---

## How to Run
1. Open `waste_management_summative.ipynb` in Jupyter Notebook or VS Code.
2. Ensure required dependencies are installed (`tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`).
3. Select **Kernel > Restart & Run All** to execute the pipeline sequentially from data exploration to the final integrated assistant evaluation.

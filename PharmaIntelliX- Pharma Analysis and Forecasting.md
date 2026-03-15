## Project Overview

PharmaIntelliX is a Business Intelligence (BI) framework designed to forecast pharmaceutical sales and optimize strategic decisions. Leveraging a heterogeneous ensemble of six predictive models (LightGBM, Random Forest, Ridge, Prophet, SARIMAX, and two baselines), the system provides robust and explainable insights into sales performance.

Beyond prediction, PharmaIntelliX includes a "What-If" simulation engine to analyze hypothetical business scenarios and a prescriptive "Strategy Optimizer" using a "Bang-for-Buck" (BFB) algorithm to recommend optimal resource allocation. The entire system is demonstrated through a comprehensive Jupyter notebook and an interactive Streamlit BI dashboard.

## Key Features

* **Multi-Model Ensemble Forecasting:** Combines diverse models for accuracy (achieved **25% MAE improvement** over baseline).
* **Time-Series Disaggregation:** Intelligent integration of aggregate time-series insights into store-level ML predictions.
* **"What-If" Simulation:** Quantifies the impact of business interventions (e.g., promotions, ad spend).
* **Prescriptive Strategy Optimization:** Recommends high-ROI actions to maximize sales uplift .
* **Explainable XAI:** Uses Permutation Importance to identify key sales drivers.
* **Interactive BI Dashboard:** User-friendly Streamlit app with real-time API integrations (weather, holidays) and AI-generated forecast summaries.

## Prepare the Data
This project utilizes the Rossmann Store Sales dataset from Kaggle. You MUST download train.csv , test.csv and store.csv from the Kaggle competition page.

## Dataset Link:
https://www.kaggle.com/competitions/rossmann-store-sales/data

## Folder Structure
PharmaIntelliX/
├── pharmaintellix.ipynb
├── app.py
├── README.md
├── requirements.txt
├── train.csv
├── test.csv
└── store.csv

## Requirements - install following libraries
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
lightgbm>=3.3.0
statsmodels>=0.13.0
prophet>=1.1.0 # Facebook Prophet
streamlit>=1.12.0
openmeteo-requests>=1.0.0 # For live weather data in Streamlit
requests-cache>=1.0.0 # Often used with openmeteo-requests for caching
retry-requests>=1.0.0 # Often used with openmeteo-requests for retries
holidays>=0.23 # For live holiday data in Streamlit
jupyterlab>=3.0.0 # For running the .ipynb notebook
ipykernel>=6.0.0 # Kernel for Jupyter

## Runnig the Project
1.Run the Jupyter Notebook
2.Open pharmaintellix.ipynb: Navigate to the notebook file and open it.
3.Install the mentioned libraries or packages 
4.Run the notebook cells sequentially to execute the data preparation, model training, forecasting, and analysis steps.
5.Explore the results and visualizations provided in the notebook.

6.Launch the Streamlit Dashboard
7.Open a terminal or command prompt.
8.Navigate to the directory where app.py is located.
9.Run the command: streamlit run app.py
10.This will start the Streamlit server and open the dashboard in your default web browser.
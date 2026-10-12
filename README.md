# AI-Powered Predictive Maintenance for IoT Devices

<div align="center">

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/ML-Random%20Forest-FF6F61" alt="Machine Learning" />
  <img src="https://img.shields.io/badge/IoT-Sensor%20Analytics-00B8D9" alt="IoT Analytics" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B" alt="Streamlit" />
</p>

<h1>AI-Powered Predictive Maintenance for IoT Devices</h1>

<p>
  <strong>Predict machine failures before they happen.</strong>
</p>
<p>
  A practical AI and industrial analytics project that uses synthetic IoT sensor data to detect failure risk, reduce downtime, and support proactive maintenance strategies.
</p>
    
</div>

## Overview
This project is a professional, beginner-friendly predictive maintenance system designed to simulate real industry use cases in smart manufacturing and industrial automation. Instead of using physical machines, the project generates realistic synthetic IoT sensor data and builds a machine learning model that predicts whether a machine is likely to fail soon.

The goal is to show practical AI and data science skills in a way that is easy to understand, strong for GitHub portfolios, and relevant to real industrial operations.

## Why this project is important
In modern factories and plants, unexpected equipment failure creates downtime, increases maintenance cost, delays production, and harms efficiency. Predictive maintenance helps companies move from reactive maintenance to proactive maintenance.

### Business value
- Predict machine failures before they occur
- Reduce downtime and production loss
- Lower emergency maintenance costs
- Improve asset reliability and uptime
- Support better maintenance planning and resource allocation

### Real-world industries
- Manufacturing plants
- Power generation systems
- Automotive production lines
- Aviation and aircraft maintenance
- Oil and gas operations
- Smart logistics and warehouse machinery

## Solution approach
The system uses sensor readings such as temperature, vibration, pressure, RPM, current, coolant temperature, humidity, and power draw to identify abnormal machine behavior. These inputs are processed using a supervised machine learning model that predicts failure risk and highlights maintenance warnings.

## Project workflow
1. Simulated IoT sensor data is generated.
2. Data is cleaned and checked for quality issues.
3. Domain-based engineered features are added.
4. A machine learning model is trained on labeled examples.
5. The model predicts the likelihood of failure.
6. Results are evaluated using standard metrics.
7. A dashboard and plots provide visual output for decision-making.

## Tech stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook
- Joblib
- Ollama (optional, local LLM assistant)

## Architecture
```text
Sensor Data --> Data Cleaning --> Feature Engineering --> Model Training --> Failure Prediction --> Alert & Visualization
```

### Main modules
- `src/data_generator.py` – Generates synthetic industrial sensor data
- `src/preprocessing.py` – Cleans and prepares the dataset
- `src/model_building.py` – Trains and evaluates the predictive model
- `src/visualization.py` – Produces confusion matrix and feature importance charts
- `src/prediction.py` – Performs live risk prediction and alert generation
- `main.py` – Executes the full workflow end-to-end
- `app.py` – Streamlit dashboard for interactive monitoring
- `notebooks/predictive_maintenance_analysis.ipynb` – Jupyter notebook walkthrough

## Folder structure
```text
AI-Predictive-Maintenance-IoT/
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── industrial_iot_failure_data.csv
├── docs/
│   ├── architecture.md
│   ├── phase_plan.md
│   └── portfolio_ready_summary.md
├── images/
├── models/
├── notebooks/
│   └── predictive_maintenance_analysis.ipynb
├── outputs/
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── model_building.py
│   ├── visualization.py
│   └── prediction.py
└── .venv/
```

## Installation
### Python version
Use Python 3.10+ or Python 3.13 for best compatibility in this setup.

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the project
### Command-line workflow
```bash
python main.py
```

### Streamlit dashboard
```bash
streamlit run app.py
```

### Optional local LLM assistant
The dashboard can ask an Ollama model to explain a prediction and suggest practical inspection steps. The classifier remains the source of the prediction; the LLM only explains it. The default endpoint is local (`http://localhost:11434`); the app sends current form readings only to the Ollama endpoint you configure and does not call a third-party LLM API itself.

1. Install and start Ollama from [ollama.com](https://ollama.com/).
2. Download a model, for example:
   ```bash
   ollama pull qwen2.5:0.5b
   ```
3. Start the dashboard and use **Generate maintenance explanation** below the risk prediction:
   ```bash
   streamlit run app.py
   ```

By default, the app uses `http://localhost:11434` and `qwen2.5:0.5b`. Configure a different Ollama endpoint or model with `OLLAMA_BASE_URL` and `OLLAMA_MODEL`. A local model requires sufficient memory and may respond slowly on CPU-only systems. If Ollama is not running or the model is missing, the dashboard displays a setup error.

#### Run app and Ollama with Docker Compose
Docker Compose starts both the dashboard and Ollama. After starting the services, download the model into the persistent Ollama volume:
```bash
docker compose up --build -d
docker compose exec ollama ollama pull qwen2.5:0.5b
```
Open `http://localhost:8501`. Model download is a one-time operation and requires several gigabytes of disk space.

### Jupyter notebook
```bash
jupyter notebook notebooks/predictive_maintenance_analysis.ipynb
```

## Expected outputs
After the project runs successfully, you will generate:
- `data/industrial_iot_failure_data.csv`
- `data/cleaned_iot_data.csv`
- `models/predictive_maintenance_model.pkl`
- `images/confusion_matrix.png`
- `images/feature_importance.png`
- `images/failure_probability.png`
- `outputs/latest_prediction.json`
- `outputs/maintenance_summary.txt`

## Local LLM behavior and limitations
- The LLM receives only the current sensor form values, classifier result, probability, and existing alert.
- The request is sent to `OLLAMA_BASE_URL` (default `http://localhost:11434`). Only point this setting to a server you trust.
- It does not train the model, alter classifier predictions, or independently detect faults.
- Recommendations are advisory; verify them with equipment documentation and a qualified maintenance professional.
- This project uses synthetic data and is a learning/portfolio prototype, not a certified safety or industrial control system.

## Model evaluation
This project evaluates the ML model using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

These metrics help measure how well the model predicts machine failure while balancing false alarms and missed detections.

## Example result
A representative result from this project may look like:
```text
Accuracy: 0.895
Precision: 0.724
Recall: 0.767
F1 Score: 0.745
```

## Professional value for GitHub and placements
This project is designed to be portfolio-ready. It shows a realistic engineering workflow and demonstrates value to industrial teams. It is strong evidence of:
- Python programming
- Machine learning fundamentals
- Data preprocessing and feature engineering
- Industrial analytics thinking
- Model selection and evaluation
- Dashboard creation and project communication

## GitHub deployment checklist
1. Create a repository on GitHub.
2. Upload the project files.
3. Add the README and screenshots.
4. Add the dashboard and notebook entries to the repo.
5. Commit with a meaningful message such as:
   ```bash
   git add .
   git commit -m "Add predictive maintenance project with dashboard and notebook"
   git push origin main
   ```

## Resume / LinkedIn-ready project description
AI/ML engineer with hands-on experience designing a predictive maintenance solution for IoT-based industrial equipment monitoring. Built a synthetic sensor-data pipeline, trained and evaluated a failure prediction model, engineered health indicators from machine telemetry, and delivered an interactive dashboard to visualize maintenance risk and equipment health.

## Final note
This project is intentionally practical, realistic, and beginner-friendly while still feeling professional and industry-aligned. It is a strong example of how data science and AI can be applied to real industrial problems even when physical machines are not available.

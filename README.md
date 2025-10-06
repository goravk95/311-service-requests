# NYC 311 Service Requests: DOHMH Ticket Forecasting

## Background
The problem this repo aims to solve is: how can we help Agency X allocate its resources?
- How should the FDNY allocate firefighters?
- How should the NYPD allocate policemen?

This repository aims to provide tools to the **NYC Department of Health and Mental Hygiene (DOHMH)** that aid them in:
- Identifying problematic areas of NYC where they frequently receive tickets
- Forecasting future (1-week ahead) ticket flow
- Comparing predicted patterns with historical data

DOHMH was selected for several reasons:

- **Interesting problem domain:** The DOHMH dataset contains compelling issues such as rodent complaints and food safety inspections.
- **Manageable data size:** With approximately 1M records, it's large enough to be meaningful but digestible enough to work with efficiently without requiring big data infrastructure.

## Technical Problem Statement

**Core question:** How many tickets will be opened in the next week?

## Technical Deliverables

1. **Predictive Models** - Machine learning models to forecast ticket volume
2. **Interactive Dashboard** - Visualization tool to explore predictions and run historical scenario analyses

## Getting Started

### Prerequisites

- Python 3.13 (version used during development)
- NYC Open Data API credentials

### Setup Instructions

1. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/nyc-311-service-requests.git
   cd nyc-311-service-requests
   ```

2. **Set up environment variables:**
   
   The NYC Open Data API (Socrata) requires authentication credentials. Create a `.env` file in the project root directory with the following variables:
   ```
   SOCRATA_APP_TOKEN=your_app_token
   SOCRATA_API_KEY_ID=your_api_key_id
   SOCRATA_API_KEY_SECRET=your_api_key_secret
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Navigating the Project

Start with the **Product Overview** notebook for a high-level overview of the problem and the two main deliverables.

Then, proceed through the numbered notebooks in order to understand the development process:

- **Step 1 - Fetch Data:** Data acquisition from NYC Open Data API and external sources
- **Step 2 - Basic EDA:** Exploratory data analysis and initial insights
- **Step 3 - Data Cleanup:** Data quality improvements and preprocessing
- **Step 4 - Feature Engineering:** Creation of predictive features including temporal, spatial, and weather data
- **Step 5a - Tuning:** Hyperparameter optimization
- **Step 5b - Modeling:** Final model training and evaluation

> Each notebook documents the intermediate steps and decision-making process throughout the project. The core feature engineering and modeling decision are made in Steps 4, 5a and 5b.

## Project Structure

```
nyc-311-service-requests/
├── data/                       # Raw and processed data
│   └── landing/               # Partitioned parquet files and external data
├── models/                     # Trained model artifacts
├── notebooks/                  # Analysis and development notebooks
├── src/                        # Source code modules
│   ├── config.py              # Configuration settings
│   ├── features.py            # Feature engineering functions
│   ├── fetch.py               # Data fetching utilities
│   ├── forecast.py            # Forecasting model code
│   ├── preprocessing.py       # Data preprocessing pipeline
│   ├── train.py               # Model training scripts
│   └── resources/             # Reference data and mappings
├── streamlit_app/             # Interactive dashboard
└── requirements.txt           # Python dependencies
```

## References

- [NOAA NCLIMGRID Daily Climate Data](https://noaa-nclimgrid-daily-pds.s3.amazonaws.com/index.html#EpiNOAA/v1-0-0/parquet/cty/YEAR=2025/STATUS=scaled/) - Weather data source
- [Research Paper on Data Quality of 311 Service Requests](https://arxiv.org/pdf/2502.08649) - Literature Review

## Disclaimer

Cursor code assist was used for many sections, especially visualization-heavy sections.

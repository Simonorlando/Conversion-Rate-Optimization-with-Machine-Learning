# Conversion Rate Optimization with XGBoost, Genetic Algorithms, and SALICON

This project aims to improve **Conversion Rate Optimization (CRO)** for an e-commerce platform using advanced machine learning techniques. The pipeline includes **XGBoost** for predicting key metrics, **genetic algorithms** for optimization, and **SALICON** for eye-tracking analysis, helping identify and improve user interaction with page elements.

## Project Structure

- **cro_ml.ipynb**: Notebook detailing the entire CRO analysis using machine learning techniques such as XGBoost and genetic algorithms to optimize conversion rates.

- **salicon_tuning.py**: Python script for tuning the SALICON model using a dataset of thousands of e-commerce page screenshots. This model helps to analyze which areas of a webpage attract the most user attention.

- **eye_tracking.py**: Script that executes the SALICON model and generates a tracking map, providing insights into which parts of the page are most engaging to users.

- **eye_tracking_color_weighting.py**: Script for generating a saliency-weighted tracking map, emphasizing the most significant areas of the page based on visual saliency.

## Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required Python libraries listed in `requirements.txt`

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your_username/CRO_ML_Project.git
   cd CRO_ML_Project

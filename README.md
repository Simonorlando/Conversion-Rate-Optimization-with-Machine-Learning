# Conversion Rate Optimization with XGBoost, Genetic Algorithms, and SALICON

This project aims to improve **Conversion Rate Optimization (CRO)** for an e-commerce platform using advanced machine learning techniques. The pipeline includes **XGBoost** for predicting key metrics, **genetic algorithms** for optimization, and **SALICON** for eye-tracking analysis, helping identify and improve user interaction with page elements.

## Project Overview

In today's competitive e-commerce landscape, optimizing conversion rates is crucial for ensuring high returns on marketing investments. This project provides a comprehensive solution to boost conversion rates by employing advanced machine learning and visual analysis techniques:

1. **XGBoost for Prediction**:
   - Used to predict key metrics such as conversion probability based on features like time on site, pages visited, user demographics, and interactions.
   - XGBoost's ability to handle non-linear relationships makes it highly effective in understanding complex user behavior and interactions on the site.

2. **Genetic Algorithms for Optimization**:
   - Applied to identify the optimal combination of features that maximize conversion rates.
   - Genetic algorithms simulate the process of natural selection, providing an efficient way to explore a large space of possible feature combinations, ensuring maximum ROI without necessarily increasing ad spend.

3. **SALICON for Eye-Tracking Analysis**:
   - Eye-tracking helps in understanding which areas of a webpage attract the most user attention.
   - **SALICON**, a model for saliency prediction, was used to generate heatmaps that guide visual changes to optimize page layout and improve user interaction.

## Project Structure

- **cro_ml.ipynb**: 
  - This notebook details the entire CRO analysis using machine learning techniques such as **XGBoost** and **genetic algorithms**.
  - The steps include data preprocessing, feature engineering, model training, and the application of genetic algorithms for optimization.
  - Insights derived from the model help improve the conversion rate through feature adjustments.

- **salicon_tuning.py**:
  - Python script for **tuning the SALICON model** using a dataset of thousands of e-commerce page screenshots.
  - The tuning process improves the accuracy of saliency predictions, allowing for more effective analysis of user attention on web elements.

- **eye_tracking.py**:
  - Script that **executes the SALICON model** and generates a tracking map.
  - The output tracking maps provide visual insights into which parts of the webpage are most engaging to users, guiding decisions on where to place important elements like calls to action (CTAs).

- **eye_tracking_color_weighting.py**:
  - Script for generating a **saliency-weighted tracking map**, emphasizing the most significant areas of the page based on visual saliency.
  - These insights are used to adjust page design and maximize user interaction with crucial elements, thus enhancing the conversion rate.

## Key Techniques and Tools

- **XGBoost**: Used for predicting the likelihood of conversion based on user behavior and site metrics. It effectively handles large datasets and complex relationships.
- **Genetic Algorithms**: Applied to optimize the key features identified by the model to improve overall conversion rates.
- **SALICON**: A model used for predicting visual saliency, helping to understand user attention and optimize webpage layout.
- **Heatmaps and Saliency Maps**: These tools visualize which areas of a webpage draw the most attention, guiding UI/UX improvements.

## Project Phases

### 1. Data Collection and Preprocessing
- **Data Collection**: Gathered user interaction data, such as time on site, pages visited, device type, and user demographics, as well as e-commerce conversion data.
- **Data Cleaning**: Handled missing values, removed outliers, and normalized the data to ensure that it was ready for modeling.
- **Feature Engineering**: Created additional features such as average time spent per page, bounce rate, and new user metrics to enhance the model's predictive capabilities.

### 2. Predictive Modeling with XGBoost
- **Model Training**: Used **XGBoost** to predict the likelihood of conversion based on preprocessed data. The model's ability to handle complex non-linear relationships was key to understanding user behavior.
- **Hyperparameter Tuning**: Conducted hyperparameter tuning to optimize model performance, ensuring high accuracy in predicting user conversion.

### 3. Optimization with Genetic Algorithms
- **Feature Optimization**: Leveraged genetic algorithms to explore various combinations of features to determine which combinations maximize conversion rates.
- **Constraints Handling**: Implemented constraints like fixed ad spend and maximum bounce rate allowed to ensure the optimization remained realistic and feasible.

### 4. Eye-Tracking Analysis with SALICON
- **Model Tuning**: Fine-tuned the **SALICON** model on a dataset of thousands of e-commerce page screenshots to improve its accuracy in predicting visual saliency.
- **Heatmap Generation**: Generated **heatmaps** to visualize user attention on different parts of the webpage. This helped identify which page elements were effective in drawing user focus.

### 5. Saliency-Weighted Design Changes
- **Saliency Maps Generation**: Used **eye_tracking_color_weighting.py** to create saliency-weighted tracking maps.
- **Design Recommendations**: Provided recommendations for UI/UX changes based on saliency analysis, optimizing the placement of CTAs, product descriptions, and other crucial elements to boost conversion.

### 6. Evaluation and Reporting
- **Conversion Rate Evaluation**: Evaluated conversion improvements by comparing pre- and post-optimization metrics.
- **Business Insights**: Generated a report summarizing the effectiveness of optimization techniques, with a focus on ROI and user engagement improvements.

## Getting Started

### Prerequisites

- **Python 3.7+**
- **Jupyter Notebook** for running `.ipynb` files
- Required Python libraries listed in **`requirements.txt`**

### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Simonorlando/CRO_ML_Project.git
   cd CRO_ML_Project

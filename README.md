# Gene-Expression-Based-Cancer-Prediction
Prediction

# Gene Expression Cancer Classifier

This project analyzes gene expression data to detect the presence of cancer using various machine learning models. It includes exploratory data analysis, visualizations, model training, accuracy comparison, and real-time prediction using user input.

![image](https://github.com/user-attachments/assets/da12b6e4-1e80-4191-b539-2b93b57cabf8)


## Features

- Loads and cleans gene expression data
- Visualizes data trends using scatter plots, histograms, heatmaps, and pair plots
- Computes correlations and evaluates missing data
- Trains and compares four ML models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Random Forest
- Saves the best model for later use
- Accepts user input to predict cancer presence in a new sample
- Visualizes the new input on existing data

## How to Use

### 1. Clone this repository

```bash
git clone https://github.com/your-username/gene-expression-cancer-detector.git
cd gene-expression-cancer-detector
```

### 2. Install Dependencies

Make sure you have Python 3.7+ installed. Then run:

```bash
pip install -r requirements.txt
```

### 3. Add Your Dataset

Place your `gene_expression.csv` file in the root directory of this project. Make sure it contains at least these three columns:
- `Gene One`
- `Gene Two`
- `Cancer Present`

### 4. Run the Analysis and Model Training

Run the main Python script (e.g., `main.py` or `gene_classifier.py`) to:
- Load and visualize the data
- Train models and compare performance
- Save the trained SVM model and scaler

```bash
python gene_classifier.py
```

### 5. Make a Prediction

You will be prompted to enter two numeric values for gene expressions:

```
Enter Gene One expression value:
Enter Gene Two expression value:
```

The model will predict whether cancer is present and visualize the input point on a scatter plot.

## File Overview

| File | Description |
|------|-------------|
| `gene_classifier.py` | Main script: loads data, trains models, saves SVM, and handles prediction |
| `svm_model.pkl` | Saved SVM model |
| `scaler.pkl` | Saved scaler used for feature normalization |
| `gene_expression.csv` | CSV file containing gene expression data |

## Requirements

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- joblib  

Install them via:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib

Result:

The visualization titled "Gene Expression vs Cancer Prediction" presents a clear and concise summary of how gene expression data is being used to predict cancer presence. Each point on the scatter plot represents a sample, with the x-axis showing "Gene One Expression" and the y-axis showing "Gene Two Expression". The dataset consists of both cancer-positive and cancer-negative samples, which are color-coded for clarity—blue indicates samples where cancer is not present (label 0), and red indicates those where cancer is present (label 1).

A key feature of this plot is the inclusion of a new patient, represented by a large black marker. This patient's gene expression values were manually entered, and their position in the plot reflects the scaled values of those two genes. An annotation next to the black point clearly displays the model’s prediction: "Cancer Present".

The new sample lies in a region surrounded by other red (positive) cases, which aligns with the model's output. This prediction was made using a Support Vector Machine (SVM) classifier that had been trained on historical data. Given that SVM is sensitive to feature scaling and performs well in high-dimensional spaces, it was well-suited for this binary classification task.

Overall, the visualization not only confirms the effectiveness of the model but also enhances interpretability. It allows us to see how the new patient compares to existing data and understand why the model made its prediction. This approach is particularly valuable in healthcare-related applications where transparency and visual validation are important.









![image](https://github.com/user-attachments/assets/e2d0c4ea-02d9-467c-959c-27ceba98ace3)










```

## Notes

- Scaling is essential for SVM and KNN; this is handled automatically before training and prediction.
- If using your own data, make sure it follows the structure of the original dataset.

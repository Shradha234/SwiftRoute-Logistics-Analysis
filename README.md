# SwiftRoute Logistics Analytics: End-to-End Data Analysis, Predictive Modeling & Optimization

**Case study:** SwiftRoute Logistics Pvt. Ltd. (hypothetical)  
**Project scope:** Four-week portfolio / academic case study  
**Domain:** Logistics, supply chain analytics, business intelligence and machine learning  
**Tools:** Python, Pandas, NumPy, Matplotlib, scikit-learn, Google Colab / Jupyter

> **Disclosure:** SwiftRoute is a hypothetical case study. The included 500-row shipment dataset is simulated. It is not real company operational data, client work, or proof of employment with SwiftRoute. Present this as a portfolio or academic project, not as a verified internship unless a separate formal internship program confirms that status.

## Project overview

This repository demonstrates a four-week analytics workflow for a simulated logistics scenario. It covers business problem framing, KPI selection, data quality checks, exploratory data analysis, visualizations, delivery-time regression, model evaluation, cross-validation, hyperparameter tuning, and operational recommendations.

## Four-week workflow

| Week | Focus | Deliverables |
|---|---|---|
| 1 | Strategic planning and data exploration | Scenario, objectives, logistics KPIs, research and analytical roadmap |
| 2 | Data collection, cleaning and preprocessing | Dataset inspection, data dictionary, quality checks and preparation |
| 3 | Advanced analysis and visualization | Descriptive statistics, shipment KPIs, zone analysis, correlation analysis and charts |
| 4 | Predictive modeling and optimization | Linear Regression, Decision Tree, five-fold GridSearchCV, evaluation and recommendations |

## Dataset

The supplied CSV contains **500 shipment records and 9 columns**:
`Shipment_ID`, `Order_Date`, `Distance_KM`, `Shipment_Weight_KG`, `Shipment_Volume`, `Delivery_Time_Days`, `Delivery_Zone`, `Delivery_Status`, and `Transportation_Cost`.

The dataset has no missing values and no duplicate rows. It was generated for educational analysis. The delivery-time values are random, which is important when interpreting the predictive modeling results.

## Selected findings from the supplied dataset

- 500 total shipment records.
- 372 Delivered, 100 Delayed, and 28 Failed.
- Delivered share: 74.40%; delay rate: 20.00%.
- Average delivery time: 4.48 days.
- Average transportation cost: 1,500.78 dataset units.
- Distance and transportation cost correlation: 0.99. Transportation cost was generated using a formula that includes distance, so this correlation is a feature of the simulation, not evidence of causality in real logistics operations.

## Predictive modeling

**Target:** `Delivery_Time_Days`  
**Features:** distance, shipment weight, shipment volume, transportation cost, and delivery zone.  
**Train/test split:** 80/20 with `random_state=42`.  
**Categorical encoding:** one-hot encoding for delivery zone.  
**Tuning:** five-fold GridSearchCV for Decision Tree Regression.

| Model | MAE (days) | RMSE (days) | R² |
|---|---:|---:|---:|
| Linear Regression | 1.876 | 2.170 | -0.069 |
| Decision Tree | 1.890 | 2.227 | -0.126 |
| Tuned Decision Tree | 1.921 | 2.266 | -0.166 |

Best tuned tree parameters: `max_depth=2`, `min_samples_leaf=10`. Best cross-validation MAE: approximately 1.985 days.

**Interpretation:** Linear Regression was the best of the tested configurations, but all test R² values were negative. The randomly generated delivery-time target does not provide a reliable learnable relationship. The models did not outperform a simple mean-based baseline on the held-out test set and must not be described as operationally useful forecasts.

## Optimization recommendations

- Analyze route efficiency using actual origin, destination, road distance and travel-time data.
- Monitor delay rate and average delivery time by zone.
- Track transportation cost per shipment and investigate cost drivers.
- Evaluate shipment consolidation using real weight, volume and vehicle-capacity data.
- Simulate proposed route or dispatch changes before implementation.
- Validate any predictive model on real historical shipment records before operational use.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── SwiftRoute_Logistics_Dataset.csv
│   └── README.md
├── notebooks/
│   ├── Week3_EDA_and_Visualization.ipynb
│   ├── Week4_Predictive_Modeling.ipynb
│   └── original_colab/
├── src/
│   └── week4_modeling.py
├── reports/
│   ├── source_docx/
│   └── pdf/
├── visualizations/
└── results/
    ├── model_comparison.csv
    └── cross_validation_and_tuning.txt
```

## Run the project

1. Clone or download this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Open notebooks in Jupyter or Google Colab. The cleaned notebooks use repository-relative data paths.
4. Run Week 3 EDA, then Week 4 modeling.
5. Alternatively, run the Week 4 script from the repository root: `python src/week4_modeling.py`.

## Skills demonstrated

Python; Pandas; NumPy; data quality checks; data preprocessing; exploratory data analysis; descriptive statistics; KPI analysis; data visualization; regression; scikit-learn pipelines; categorical encoding; train-test split; cross-validation; GridSearchCV; MAE; RMSE; R²; business interpretation; operational recommendations.

## Limitations and next steps

This is a simulated case study. The model target was randomly generated, so predictive performance is weak. A real deployment would require actual delivery timestamps, planned delivery windows, route and traffic information, vehicle capacity, fuel consumption, and recorded delay reasons. No real-world cost savings or operational improvements are claimed.

## Author

**Shradhalini Sahoo**  
B.Tech, Computer Science (AI & ML)  
Bhubaneswar, Odisha, India

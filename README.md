# ✈️ Flight Delays Prediction Using Machine Learning

Predicting whether a flight will be delayed before it even takes off — using data-driven insights and machine learning.

# 🧩 Introduction

Air travel is one of the most critical pillars of modern transportation — connecting people, goods, and businesses across the world. Yet, one recurring challenge continues to impact both airlines and passengers alike: flight delays.

Even a delay of a few minutes can trigger a chain reaction — disrupting connecting flights, increasing operational costs, and leading to passenger dissatisfaction. According to the Federal Aviation Administration (FAA), a flight is considered delayed if it departs 15 minutes or more later than its scheduled time.

In this project, we built a machine learning pipeline to predict whether a flight will be delayed or on-time, based on its characteristics, timing, and operational conditions.

## Our goal:

To identify the key attributes that influence delays.

To build robust predictive models capable of classifying flight outcomes.

To evaluate different ML algorithms and handle imbalanced data effectively.

# 🎯 Motivation

Flight delays are more than just an inconvenience — they represent billions of dollars in lost revenue, fuel wastage, and operational inefficiency each year.

For passengers, delays can mean missed connections, increased travel anxiety, and time lost. For airlines, delays lead to cascading disruptions, higher crew costs, and negative customer perception.

By developing a predictive model, we aim to help airlines:

Anticipate possible delays.

Optimize scheduling and turnaround times.

Improve customer satisfaction and operational reliability.

# 🧠 Problem Statement

Can we predict whether a flight will be delayed based on its operational and environmental factors?

To answer this, we framed the problem as a binary classification task:

Class 0 → On-Time Flight

Class 1 → Delayed Flight (> 15 minutes)

The challenge?
The dataset is heavily imbalanced, as most flights operate on time. This required special preprocessing and resampling techniques to ensure balanced model learning.

# 📊 Dataset

Source: Kaggle – Airline Delay and Cancellation Data (2009–2018)

Provider: U.S. Department of Transportation, Bureau of Transportation Statistics
Year Used: 2018 subset
Size: 7,213,446 records | 27 features

Each record contains flight-level information including:

Carrier and Flight Number

Origin and Destination Airports

Scheduled/Actual Departure and Arrival Times

Taxi-In and Taxi-Out Times

Distance and Elapsed Time

Delay Type Codes

# 🧹 Data Preprocessing

All preprocessing steps were executed in Python using Pandas and NumPy.

## Data Filtering:

Selected flights from the busiest U.S. airports for higher data quality.

Removed irrelevant attributes such as canceled/diverted flight indicators.

## Data Cleaning:

Dropped columns with >50% missing data.

Standardized carrier and airport codes using IATA abbreviations.

Removed records with minor missing values (<1% of total).

## Feature Engineering:

Extracted Month, Day, and Day of Week from flight date.

Derived Wheels-On, Wheels-Off, and Elapsed Time durations.

Created a binary feature FlightDelay (1 = Delayed, 0 = On-Time).

## Encoding:

Converted categorical attributes (e.g., Airport, Carrier, Day of Week) using One-Hot Encoding.

# 🔍 Feature Selection

Feature importance was evaluated using a Random Forest Classifier to identify the most impactful predictors.

Key Observations:

Highly correlated features (Elapsed Time, Distance, Airtime) were compared to avoid redundancy.

Features like Departure Delay, Taxi-Out, Distance, and Wheels-Off Time had the strongest predictive power.

Temporal attributes such as Month and Day had minimal impact.

Only the top 9 features with the highest importance scores were retained for modeling.

# ⚙️ Model Development

The modeling phase was performed using Weka and Python.

Models Tested:

Naïve Bayes (NB)

Logistic Regression (LR)

Decision Tree (DT)

Random Forest (RF)

AdaBoost (with Decision Tree base)

Handling Imbalanced Data:

Since delayed flights formed the minority class, the dataset was resampled using:

SMOTE (Synthetic Minority Oversampling Technique) — generated synthetic samples to balance classes.

Random Undersampling — reduced majority samples to equalize class distribution.

After resampling, recall and F1-scores improved significantly, especially for Decision Tree and Random Forest models.

# 📈 Results & Evaluation
Model	Accuracy	Recall	F1-Score	Remarks
Logistic Regression	Moderate	Low	Moderate	Struggled with imbalance
Decision Tree	High	Improved	High	Sensitive to depth
Random Forest	High	Good	Very High	Balanced and interpretable
AdaBoost (DT)	Best	Highest	Top Performer	Best F1 and recall tradeoff

## 🏆 Best Model: AdaBoost with Decision Tree
Key Features: Departure Delay, Taxi-Out, Wheels-Off, Distance

# 💡 Insights

Departure Delay is the single strongest indicator of future delays.

Taxi-Out and Wheels-Off times capture operational congestion effectively.

Weather-related and temporal factors contributed less to prediction accuracy.

Balanced data improved recall by over 20%, indicating fairer model performance across both classes.

# 🔮 Future Work

Extend modeling using Neural Networks and Ensemble Deep Learning.

Explore advanced oversampling techniques like ADASYN to minimize noise.

Implement Cost-Sensitive Learning to penalize misclassification of minority classes.

Deploy the best-performing model as an API-based flight delay prediction service.

# 📚 References

Bureau of Transportation Statistics (2021). Understanding the reporting of causes of flight delays and cancellations.

National Academies of Sciences, Engineering and Medicine (2014). Defining and Measuring Aircraft Delay and Capacity Thresholds.

NEXTOR (2010). Total Delay Impact Study: Comprehensive Assessment of the Costs and Impacts of Flight Delay in the U.S.

## 🧾 Keywords

Machine Learning | Flight Delay Prediction | Data Preprocessing | Feature Engineering | Imbalanced Data | SMOTE | Random Forest | AdaBoost | Classification | Python

# 📊 Tools & Libraries

Python, Pandas, NumPy, Scikit-learn, Weka, Matplotlib, Seaborn, Jupyter Notebook

## 👨‍💻 Contributors

This project was developed as part of the Machine Learning coursework in the Master of Data Science program (Group Project).

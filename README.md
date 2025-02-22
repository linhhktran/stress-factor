# Analyzing Student Stress Factors for Improved Mental Health Support
![image](https://img.freepik.com/premium-photo/student-stress-concept-illustration-jpg_1072857-4797.jpg)
## **Purpose and Outcome:**
- **Purpose:** To identify key factors contributing to student stress and develop actionable strategies for improving mental health support in academic institutions.
- **Outcome:** Create a model that predicts stress levels, helping schools and mental health professionals understand the main reasons students feel stressed. This can guide changes or provide support where it is needed most.

---

## **Dataset Information:**
- **Number of Observations (Rows)**: 1,100.
- **Number of Features (Columns)**: 20, grouped into five categories:  
  - **Psychological Factors** → 'anxiety_level', 'self_esteem', 'mental_health_history', 'depression'  
  - **Physiological Factors** → 'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem’  
  - **Environmental Factors** → 'noise_level', 'living_conditions', 'safety', 'basic_needs'  
  - **Academic Factors** → 'academic_performance', 'study_load', 'teacher_student_relationship', 'future_career_concerns'  
  - **Social Factors** → 'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying'

### **Scale Explanation:**
- **‘anxiety_level' (0-21)**: Generalized Anxiety Disorder Scale (GAD-7)
  - 0–4: Minimal anxiety
  - 5–9: Mild anxiety
  - 10–14: Moderate anxiety
  - 15–21: Severe anxiety

- **‘self_esteem' (0-30)**: Rosenberg Self-Esteem Scale
  - 0–14: Low self-esteem
  - 15–24: Moderate self-esteem
  - 25–30: High self-esteem

- **‘depression' (0-27)**: Patient Health Questionnaire (PHQ-9)
  - 0–4: Minimal depression
  - 5–9: Mild depression
  - 10–14: Moderate depression
  - 15–19: Moderately severe depression
  - 20–27: Severe depression

- **mental_health_history (0/1)**: Yes/No

- **‘blood_pressure' (1-3)**:
  - 1: Normal blood pressure
  - 2: Elevated blood pressure (pre-hypertension)
  - 3: High blood pressure (hypertension)

- **The remaining columns (0-5)** can be adjusted depending on which factors:
  - 0: Very Poor
  - 1: Poor
  - 2: Below Average
  - 3: Average
  - 4: Good
  - 5: Very Good

---

## **Step 1: Data Download**

Download the dataset from Kaggle using `opendatasets`. This dataset will be used to analyze the stress factors and predict stress levels.

---

## **Step 2: Import Libraries and Process Data**
Libraries used in this project: 
- pandas: Data manipulation and analysis.
- numpy: Numerical operations.
- matplotlib.pyplot: Plotting and data visualization.
- seaborn: Enhanced data visualization.
- statsmodels.api: Statistical modeling and hypothesis testing.
- sklearn.model_selection: Splitting data and cross-validation.
- sklearn.linear_model: Machine learning models (logistic regression, etc.).
- scipy.stats: Statistical functions and tests.
- sklearn.metrics: Model evaluation and performance metrics.

- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `scipy and `sklearn`.
- The dataset is loaded, and the `stress_level` column is transformed into a binary format for easier classification.

---

## **Step 3: Exploratory Data Analysis (EDA)**

- **Visualize data distributions**: Visualizations like count plots help in understanding how each variable is distributed.
  
- **Analyze correlations**: We check the relationship between the factors (such as anxiety, academic performance, etc.) and stress level using chi-square test.

---

## **Step 4: Chi-Square Test**

- **Chi-Square Test** for independence between each categorical feature and stress level. This test helps assess whether a feature (e.g., anxiety level, self-esteem, etc.) is significantly related to the **stress_level** or not.

---

## **Step 5: Logistic Regression Model**

- **Build Logistic Regression Model** using **statsmodels**.
- The model is evaluated using **classification report**, **confusion matrix**, and **accuracy** to assess how well it predicts stress levels based on the given factors.

---

## **Step 6: Model Training and Evaluation**

- **Train-Test Split**: Data is split into training and testing sets to evaluate the model on unseen data.
- **Classification Report**: The model’s precision, recall, F1-score, and accuracy are evaluated.
- **Confusion Matrix**: This visualizes the model's performance, showing how many true positives, true negatives, false positives, and false negatives it produced.

---

## **Step 7: Feature Importance**

- **Extract Feature Coefficients**: The model's coefficients are examined to determine which features most significantly influence the prediction of stress levels.
- Features with higher absolute values in the coefficients are considered **more important**.

---

## **Step 8: Prediction on Test Data**

- The trained model is used to make predictions on a new set of test data. Predictions are made based on a threshold of 0.5.

---

## **Step 9: Recommendations**

- **Focus on managing anxiety** and **depression**.
- Improve **sleep quality** and address **headaches**.
- Provide **academic support** for students with poor performance and high study load.
- **Strengthen social support networks** and reduce **peer pressure**.
- Address **bullying** and enhance **safety measures**.

---

## **Project Files:**
- **`stress_factor.ipynb`**: Jupyter Notebook containing the data analysis, model building, and evaluation or you can access to my Google Colab Notebook: https://colab.research.google.com/drive/1ybgIMuqS-9EyatZXOjEBDfpGt2RgZ5CW?usp=sharing

---

### **Conclusion:**

The logistic regression model successfully predicts **stress levels** based on psychological, physiological, academic, and social factors. By improving areas such as **anxiety management**, **academic support**, and **sleep quality**, stress levels can be significantly reduced. Further model optimization and additional data collection can improve predictions.

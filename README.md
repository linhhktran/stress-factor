# Analyzing Student Stress Factors for Improved Mental Health Support
![image](https://img.freepik.com/premium-photo/student-stress-concept-illustration-jpg_1072857-4797.jpg)
## **Introduction**
Student stress is a growing issue in academic institutions, significantly affecting students' mental and physical well-being. This project aims to analyze various psychological, physiological, academic, and social factors contributing to student stress levels, using data analysis and machine learning techniques to identify key stressors and predict stress levels. By understanding these factors, the project seeks to provide valuable insights for improving mental health support and well-being programs in schools and universities, helping educators and health professionals implement effective strategies to manage and reduce stress, ultimately enhancing students' academic and personal experiences.


## **Purpose and Outcome**
- **Purpose:** To identify key factors contributing to student stress and develop actionable strategies for improving mental health support in academic institutions.
- **Outcome:** Create a model that predicts stress levels, helping schools and mental health professionals understand the main reasons students feel stressed. This can guide changes or provide support where it is needed most.

---
## **Data source**
The dataset is obtained from Kaggle: https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis/data 

## **Dataset Information:**
- **Number of Observations (Rows)**: 1,100.
- **Number of Features (Columns)**: 20, grouped into five categories:  
  - **Psychological Factors** → 'anxiety_level', 'self_esteem', 'mental_health_history', 'depression'  
  - **Physiological Factors** → 'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem’  
  - **Environmental Factors** → 'noise_level', 'living_conditions', 'safety', 'basic_needs'  
  - **Academic Factors** → 'academic_performance', 'study_load', 'teacher_student_relationship', 'future_career_concerns'  
  - **Social Factors** → 'social_support', 'peer_pressure', 'extracurricular_activities', 'bullying'

## **Scale Explanation:**
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

## Tools and techniques applied
### 1. Tools
Google Colabs: using Python to load data, clean data, build Ma predictive models, and do feature analysis.
### 2. Techniques

In this project, I will use the Logistic Regression Model to predict the stress level (Yes/No) of students and use train_test_split, classification_report, and confusion_matrix to evaluate the model.

---

## **Step 1: Data Download**

Download the dataset from Kaggle using `opendatasets`. This dataset will be used to analyze the stress factors and predict stress levels.
```python
!pip install opendatasets
```

Since I do not want to download the dataset manually, I am using this library to automatically download it by entering the Kaggle API Key:
```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis?utm_medium=social&utm_campaign=kaggle-dataset-share&utm_source=facebook&fbclid=IwY2xjawIbRtpleHRuA2FlbQIxMQABHRdRNGmQAKlzUZb-JqmCkOyh_hwJ-NBBiiMx_HEVgykntNr09IPuZSJXWQ_aem_L-4C_GcPU3m4ROoQy4Dd9Q")
```

**How to get the Kaggle API Key to download dataset on Kaggle:**

- Go to Your Kaggle Account Settings
- On the page shown in the image, click on your profile icon in the top right corner.
- From the dropdown menu, select "Settings". Scroll Down to the API Section
- In the API section, you’ll find an option to Create New API Token.
- Download kaggle.json:
- After clicking "Create New API Token", a file named kaggle.json will be downloaded to your computer.
- Open kaggle.json for your information and type in username and key.

Then the dataset should be imported to Google Colab Notebook File.

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

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
```

Let's take a quick look over 5 first rows of the dataset (In Google Colab Notebook)
```python
df = pd.read_csv("/content/student-stress-factors-a-comprehensive-analysis/StressLevelDataset.csv")
df.head()
```

Let's take a quick look through data info to see if any missing values (See in Google Colab Notebook)
```python
df.info()
```

Luckily, the dataset does not have any missing values so we can skip handling missing values step.

- The dataset is loaded, and the `stress_level` column is transformed into a binary format for easier classification.

```python
df['stress_level'] = df['stress_level'].map({0:0, 1:1, 2:1})
```
---

## **Step 3: Exploratory Data Analysis (EDA)**

- **Visualize data distributions**: Visualizations like count plots help in understanding how each variable is distributed.
  
- **Analyze correlations**: We check the relationship between the factors (such as anxiety, academic performance, etc.) and stress level using chi-square test.

Let's plot bar charts showing the total counts in each factor in order to see general distribution of the dataset.
```python
import math

# Define number of columns (3 per row) and calculate rows dynamically
num_columns = 3
num_plots = len(df.columns)
num_rows = math.ceil(num_plots / num_columns)  # Calculate required rows

# Adjust figure size dynamically based on the number of rows
plt.figure(figsize=(19, num_rows * 5.5))

for i, column in enumerate(df.columns, 1):
    plt.subplot(num_rows, num_columns, i)  # Organize plots into grid layout (rows x 3 columns)
    ax = sns.countplot(x=column, data=df, palette='mako')

    # Add data labels on top of bars
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}',  # Convert count to integer
            (p.get_x() + p.get_width() / 2, p.get_height()),  # Position text at the top center of the bar
            ha='center',  # Horizontally align text at center
            va='bottom',  # Position text slightly above the bar
            fontsize=10,
            color='black',
            xytext=(0, 3),  # Offset to avoid overlap with the bar
            textcoords="offset points"
        )

    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.title(column)  # Set title for each subplot

plt.tight_layout()  # Optimize layout to prevent overlap
plt.show()
```

***image***

### **Insights**

**Psychological Factors:**

*Anxiety Level:*

*   Most responses concentrated in moderate anxiety levels (10-14), which is ~27% of the total


*Self-Esteem:*


*   Most responses concentrated in high self-esteem levels (25-30), which is ~33% of the total.

*Mental Health History:*

*   Individuals with and without mental health history are nearly equally distributed (50/50)

*Depression:*

*   Most responses concentrated in moderate depression levels (10-14), which is ~27% of the total.

**Physiological Factors:**


*Headache:*


*   Many individuals experience headaches - they are mostly at level 1 and 3, which are very mild and modeate level respectively.


*Blood Pressure:*

*   Most students experience high blood pressure (level 2-3), which takes up ~73% of the total.

*Sleep Quality:*

*   Majority report poor sleep quality at level 1.


*Breathing Problem:*

*   Most have breathing problems, especially at mild level (level 2) and severe level (level 4)


**Environmental Factors:**

*Noise Level:*

*  Many report low to moderate levels of noise (level 2-3).

*Living Conditions:*

*  Most report below average to average living conditions (level 2-3), with few extremes.


*Safety:*

*  Most report fair safety level, especially at level 2.

*Basic Needs:*

*  Most report fair access to basic needs, especially at level 2.


**Academic Factors:**

*Academic Performance:*

*  Majority report below average academic performance (level 2).

*Study Load:*

*  Many report moderate study loads (level 2-3),

*Teacher-Student Relationship:*

*  Majority report fairly positive relationships (level 2).

*Future Career Concerns:*

*  Many are concerned about their careers, but just a light level (level 1).


**Social Factors:**

*Social Support:*

*  Majority report adequate social support - mostly in level 1 and 3.

*Peer Pressure:*

*  Many report fair peer pressure level (level 2).

*Extracurricular Activities:*

*  Many participate in extracurricular activities, but not overwhelmingly, mostly level 2.

*Bullying:*

*  Majority report a significant number experience bullying at level 1.




---

## **Step 4: Perform Chi-Square Test**

- **Chi-Square Test** for independence between each categorical feature and stress level. This test helps assess whether a feature (e.g., anxiety level, self-esteem, etc.) is significantly related to the **stress_level** or not.


Since all of the above variables are categorical, Pearson’s correlation cannot be used to measure the strength and direction of their linear relationship. In this case, Chi-square test is performed to see if they are significantly related to stress_level or not.

```python
# List of numeric categorical variables
categorical_columns = df.drop('stress_level', axis=1)
categorical_variables = categorical_columns.columns

# Store p-values for each variable
p_values = {}

# Loop through each numeric categorical variable and perform Chi-square test with stress_level
for var in categorical_variables:
    # Create a contingency table for each variable and the binary stress_level
    contingency_table = pd.crosstab(df[var], df['stress_level'])

    # Perform the Chi-square test of independence
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Store the p-value
    p_values[var] = p_value

# Print the p-values for each variable
p_values
```
{'anxiety_level': 1.1938362077342143e-141,
 'self_esteem': 2.411923932968084e-133,
 'mental_health_history': 1.7536764103203237e-76,
 'depression': 1.089618001608702e-135,
 'headache': 2.9046832125102588e-167,
 'blood_pressure': 3.4964796648828596e-179,
 'sleep_quality': 8.299021953291136e-151,
 'breathing_problem': 1.551118764502076e-91,
 'noise_level': 1.1642815766321163e-104,
 'living_conditions': 9.220630012635033e-107,
 'safety': 5.2216565809793016e-164,
 'basic_needs': 3.119078598356659e-162,
 'academic_performance': 1.2865044180425957e-163,
 'study_load': 3.193137200822339e-98,
 'teacher_student_relationship': 3.223795854151013e-169,
 'future_career_concerns': 5.498593724561714e-170,
 'social_support': 1.086203098107412e-79,
 'peer_pressure': 4.949628553186511e-88,
 'extracurricular_activities': 1.1071653825402377e-93,
 'bullying': 1.185223781907082e-164}

All factors are related to stress_level, all are significantly associated with stress levels. --> use all of them in logistic regression.

---

## **Step 5: Build Logistic Regression Model**

- **Build Logistic Regression Model** using **statsmodels**.
- The model is evaluated using **classification report**, **confusion matrix**, and **accuracy** to assess how well it predicts stress levels based on the given factors.

```python
X = df.drop('stress_level', axis=1)
y = df['stress_level']
X = sm.add_constant(X)
model = sm.Logit(y, X).fit()
model.summary()
```

***image***

### **Results:**
Pseudo R-squared: $0.7775$

>In this case, a value of 0.78 indicates a reasonably good fit.


LLR p-value: $1.629e-129$

>The p-value is extremely low, indicating that the model can significantly predict the target variable (stress_level).

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

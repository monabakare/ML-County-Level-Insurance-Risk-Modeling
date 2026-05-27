### Introduction / Background
Health insurance coverage is a critical determinant of healthcare access, preventive service utilization, and long-term health outcomes in the United States. Counties with high uninsured rates often experience poorer health outcomes, increased emergency room utilization, and greater financial strain on public health systems. Prior research has demonstrated strong associations between insurance coverage and both mortality and access to care [1], while socioeconomic determinants such as income, employment, and education significantly influence coverage disparities [2]. Additionally, geographic disparities in healthcare access remain persistent despite federal policy efforts such as the Affordable Care Act (ACA) [3].

This project leverages county-level data from Data Commons, a publicly accessible knowledge graph aggregating U.S. Census, CDC, and other federal data sources. The primary dataset explores “Which counties in the US have the highest rates of uninsured?”

Dataset link: [Data Commons – Uninsured Rates by County](https://datacommons.org/explore#client=ui_landing&q=Which+counties+in+the+US+have+the+highest+rates+of+uninsured)

The dataset provides county-level measures of health insurance coverage and its socioeconomic, demographic, and structural determinants. Specifically, the dataset captures:

Insurance Coverage Metrics: Total uninsured rate and coverage breakdowns by age group.

Demographic Composition: Population size, age distribution, racial and ethnic composition, and gender distribution.

Socioeconomic Indicators: Median household income, poverty rate, unemployment rate, labor force participation, and educational attainment levels.

Health and Access Proxies: Indicators related to healthcare access.

Geographic and Structural Characteristics: Urban–rural classification, regional location, and population density.

These features enable both descriptive and predictive modeling of uninsured rates across U.S. counties. Because the dataset integrates socioeconomic and demographic indicators, it is well-suited for supervised and unsupervised machine learning approaches to uncover structural patterns and predictive relationships.

Existing literature has primarily focused on national or state-level trends [1][3], with fewer predictive modeling studies at the county level. Machine learning offers an opportunity to move beyond correlation toward identifying nonlinear relationships and clustering counties with similar risk profiles.

### Problem Definition
**Problem:**

Can we use county-level socioeconomic and demographic features to predict and identify high-risk counties with elevated uninsured rates, and uncover structural clusters of counties with similar insurance vulnerability profiles?

We aim to predict uninsured rate as a continuous variable, classify counties into high-risk vs. low-risk categories, and identify latent clusters of counties with similar socioeconomic patterns.

### Motivation:

Healthcare access inequities remain a pressing national concern. Counties with persistently high uninsured rates often overlap with economically disadvantaged or rural regions. However, policymakers typically rely on descriptive statistics rather than predictive tools.

A machine learning framework could enable early identification of at-risk counties, support data-driven policy allocation of healthcare resources, and reveal nonlinear interactions between poverty, employment, education, and insurance coverage.

Beyond predictive performance, this project contributes to sustainability and ethical governance by promoting equitable healthcare access.

### 3. Methods

#### Data Processing:

We collected county-level socioeconomic indicators from Data Commons, including total population, number of uninsured households, median household income, and unemployment rate. From these variables, we constructed a target variable, `UninsuredRate`, defined as the ratio of uninsured households to total population. Rows with missing target values were removed, while remaining feature-level missing values were handled through preprocessing pipelines to preserve as much data as possible.

To prevent data leakage, the dataset was split into training (80%) and testing (20%) sets prior to preprocessing. Feature preprocessing was performed using a pipeline-based approach. Numerical features were imputed using the median to handle missing values and then standardized using z-score normalization (`StandardScaler`) to ensure consistent feature scaling. Categorical features, when present, were imputed using the most frequent value and encoded using one-hot encoding. A `ColumnTransformer` was used to apply these transformations in parallel, ensuring a clean, reproducible, and leakage-resistant preprocessing workflow.

Additional engineered features, including log-transformed income, log-transformed population density, bachelor-degree rate, and an economic distress index, were created to better capture nonlinear socioeconomic relationships.

#### Model and Implementation:

To address the problem of predicting uninsured rates and identifying high-risk counties, we implemented both supervised and unsupervised learning approaches.

**ElasticNetCV (Baseline Regression)**

As a baseline regression model, we implemented ElasticNetCV. ElasticNet was selected because it provides a balance between feature selection and coefficient shrinkage, helping mitigate overfitting while maintaining interpretability. Additionally, ElasticNetCV performs internal cross-validation to automatically tune hyperparameters such as regularization strength, reducing the need for manual tuning.

The model was integrated into a unified preprocessing pipeline, ensuring transformations were applied consistently during both training and evaluation. Model performance was evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².

This approach established a strong and interpretable linear baseline while maintaining robustness against overfitting. Although ElasticNet is limited in capturing complex nonlinear relationships, it provides an important benchmark against which more advanced models can be compared.

**Random Forest Regressor**

To capture nonlinear relationships and interactions between socioeconomic variables, `RandomForestRegressor` was implemented. Unlike ElasticNet, Random Forest does not assume linear relationships and can model more complex dependencies between features such as income, population density, unemployment, and age.

We constrained hyperparameters such as tree depth and minimum samples per leaf to reduce overfitting and improve generalization. Random Forest Regression served as a nonlinear benchmark, allowing us to evaluate whether increased model flexibility improved predictive performance and better captured variation in uninsured rates.

**Logistic Regression**

Logistic Regression was used as a baseline classification model due to its interpretability and efficiency. The model predicts the probability of a county being classified as high-risk using a linear decision boundary, making it well-suited for understanding how socioeconomic variables contribute to classification outcomes.

Logistic Regression provides clear insight into feature importance and directionality, which is valuable for policy-focused applications where interpretability is critical. It serves as a baseline classifier, allowing comparison against more flexible nonlinear models.

**Random Forest Classifier**

To capture nonlinear decision boundaries and feature interactions, `RandomForestClassifier` was implemented. Unlike Logistic Regression, this model can learn more complex relationships between socioeconomic variables and county-level uninsured risk.

The ensemble nature of Random Forest allows it to handle feature interactions and variability in the data more effectively. It serves as a nonlinear classification benchmark, enabling comparison against Logistic Regression to determine whether increased model complexity improves classification performance.

**KMeans Clustering (Unsupervised Learning)**

In addition to predictive modeling, we applied KMeans clustering to identify groups of counties with similar geographic and socioeconomic characteristics. This unsupervised approach enables discovery of latent structural patterns that may not be fully captured by supervised learning models.

![ElbowMethod](figures/Elbow_method.png)

The number of clusters (`k = 4`) was selected using the elbow method. Principal Component Analysis (PCA) was additionally used for dimensionality reduction and visualization. KMeans complements the regression and classification models by helping explain why certain counties exhibit similar insurance-vulnerability profiles, rather than only predicting outcomes.

#### Evaluation Pipeline

Model evaluation was centralized through a unified metrics pipeline that performed holdout evaluation, 5-fold cross-validation, and automated visualization generation for regression, classification, and clustering workflows. This ensured consistent evaluation procedures across all models while improving reproducibility and comparability.

#### Rationale

This combination of models and preprocessing techniques allowed us to approach the problem from multiple perspectives while maintaining proper data preparation and evaluation practices. ElasticNet provided an interpretable linear baseline, while Random Forest Regression captured nonlinear interactions and more complex feature relationships. Classification models, including Logistic Regression and Random Forest Classifier, provided a decision-oriented perspective by identifying counties at high risk for elevated uninsured rates. This enables more actionable insights for resource allocation and policy intervention.

KMeans clustering revealed broader structural groupings and socioeconomic patterns across counties. Together, these approaches support both prediction (estimating uninsured risk) and interpretation (understanding why patterns exist), aligning closely with the project goal of identifying high-risk counties and uncovering underlying structural disparities.

### Introduction / Background
Health insurance coverage is a critical determinant of healthcare access, preventive service utilization, and long-term health outcomes in the United States. Counties with high uninsured rates often experience poorer health outcomes, increased emergency room utilization, and greater financial strain on public health systems. Prior research has demonstrated strong associations between insurance coverage and both mortality and access to care [1], while socioeconomic determinants such as income, employment, and education significantly influence coverage disparities [2]. Additionally, geographic disparities in healthcare access remain persistent despite federal policy efforts such as the Affordable Care Act (ACA) [3].

This project leverages county-level data from Data Commons, a publicly accessible knowledge graph aggregating U.S. Census, CDC, and other federal data sources. The primary dataset explores “Which counties in the US have the highest rates of uninsured?”

Dataset link: Data Commons – Uninsured Rates by County

The dataset provides county-level measures of health insurance coverage and its socioeconomic, demographic, and structural determinants. Specifically, the dataset captures:

Insurance Coverage Metrics: Total uninsured rate and coverage breakdowns by age group.

Demographic Composition: Population size, age distribution, racial and ethnic composition, and gender distribution.

Socioeconomic Indicators: Median household income, poverty rate, unemployment rate, labor force participation, and educational attainment levels.

Health and Access Proxies: Indicators related to healthcare access.

Geographic and Structural Characteristics: Urban–rural classification, regional location, and population density.

These features enable both descriptive and predictive modeling of uninsured rates across U.S. counties. Because the dataset integrates socioeconomic and demographic indicators, it is well-suited for supervised and unsupervised machine learning approaches to uncover structural patterns and predictive relationships.

Existing literature has primarily focused on national or state-level trends [1][3], with fewer predictive modeling studies at the county level. Machine learning offers an opportunity to move beyond correlation toward identifying nonlinear relationships and clustering counties with similar risk profiles.

2. Problem Definition
Problem:

Can we use county-level socioeconomic and demographic features to predict and identify high-risk counties with elevated uninsured rates, and uncover structural clusters of counties with similar insurance vulnerability profiles?

We aim to predict uninsured rate as a continuous variable, classify counties into high-risk vs. low-risk categories, and identify latent clusters of counties with similar socioeconomic patterns.

Motivation:

Healthcare access inequities remain a pressing national concern. Counties with persistently high uninsured rates often overlap with economically disadvantaged or rural regions. However, policymakers typically rely on descriptive statistics rather than predictive tools.

A machine learning framework could enable early identification of at-risk counties, support data-driven policy allocation of healthcare resources, and reveal nonlinear interactions between poverty, employment, education, and insurance coverage.

Beyond predictive performance, this project contributes to sustainability and ethical governance by promoting equitable healthcare access.

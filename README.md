# No-Hunger Project

## Overview
The No-Hunger Project is a data-driven initiative aimed at addressing global hunger by analyzing key indicators such as Global Hunger Index (GHI), child mortality, stunting, wasting, and undernourishment. This project combines data from various sources to explore trends, patterns, and actionable insights that contribute to alleviating hunger and promoting food security.

## Dataset Details
The project leverages data from multiple datasets:
- **GHI Data**: Global Hunger Index values from 2000, 2007, 2014, and 2022, including metrics on the absolute and percentage change in GHI scores since 2014.
- **Mortality Data**: Child mortality rates from 2000, 2007, 2014, and 2022.
- **Stunting Data**: Stunting prevalence in children for the years 2000, 2007, 2014, and 2022.
- **Undernourished Data**: Data on the proportion of the undernourished population from 2000 to 2022.
- **Wasting Data**: Prevalence of wasting in children over the same time period.

Each dataset includes information per country to enable cross-country comparisons and temporal analysis.

## Project Goals
The primary objectives of this project are:
1. **Data Analysis**: Conduct exploratory data analysis (EDA) to identify trends and correlations among different hunger-related indicators.
2. **Visualization**: Create meaningful visualizations to illustrate the relationships between indicators, such as undernourishment and stunting, and how they impact GHI scores.
3. **Predictive Modeling**: Build predictive models to anticipate future hunger-related trends based on historical data.
4. **Insight Generation**: Provide actionable insights that can inform policy decisions and targeted interventions to reduce hunger and malnutrition.

## Data Preprocessing
The datasets contain some missing values. Our preprocessing steps include:
- Handling missing values in critical columns such as GHI scores, stunting, wasting, and undernourishment indicators.
- Normalizing and merging data across datasets to create a comprehensive dataset suitable for modeling and analysis.

## Installation and Usage
### Prerequisites
- Python 3.8+
- Required libraries:
  ```bash
  pip install numpy pandas matplotlib seaborn 

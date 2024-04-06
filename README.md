# Data Science Salary Estimator: Project Overview
* Created a tool that estimates data science salaries (MAE ~ $ 22K) to help data scientists negotiate their income when they get a job.
* Scraped about 900 job descriptions from glassdoor using python and selenium
* Engineered features from the text of each job description to quantify the value companies put on python, excel, aws, and spark.
* Optimized Linear Regression, Decision Tree, Lasso Regression, Gradient Boosting, XGBoost, and Random Forest Regressors using GridsearchCV to reach the best model.

## Code and Resources Used
* Python version: 3.12
* Packages: pandas, numpy, sklearn, matplotlib, seaborn, selenium
*  Scraper Github: https://github.com/arapfaik/scraping-glassdoor-selenium
* Scraper Article: https://mersakarya.medium.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905
* YouTube Project Walk-Through: https://www.youtube.com/playlist?list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

## Web Scraping
Tweaked the web scraper github repo (above) to scrape 1000 job postings from glassdoor.com. With each job, we got the following:
*       Job title
*       Salary Estimate
*       Job Description
*       Rating
*       Company
*       Location
*       Company Size
*       Company Founded Date
*       Type of Ownership
*       Industry
*       Sector
*       Revenue

## Data Cleaning
After scraping the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:
*        Parsed numeric data out of salary
*        Made column for hourly wages
*        Removed rows without salary
*        Made a new column for company state and city
*        Transformed founded date into age of company
*        Made columns for different skills that were listed in the job description:
    * Python
    * R
    * Excel
    * AWS
    * Spark
    * java
    * Power BI
    * Tableau
    * SAS
    * SQL
*        Made columns for different study fields that were listed in the job description:
    * Statistics
    * Mathematics
    * Computer Science
    * Economics
*        Made columns for different education level that were listed in the job description:
    * PhD
    * Masters
    * Bachelors
Column for simplified job title and Seniority
Column for description length
Column for remote jobs

## EDA
## Model Building
First, I transformed the categorical variables into dummy variables. I also split the data into train and tests sets with a test size of 20%.

I tried six different models and evaluated them using Mean Absolute Error. I chose MAE because it is relatively easy to interpret and outliers aren’t particularly bad in for this type of model.

I tried six different models:
*  Multiple Linear Regression – Baseline for the model
*  Lasso Regression – Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
*  Random Forest – Again, with the sparsity associated with the data, I thought that this would be a good fit.
*  Decision Trees - Provide a clear and intuitive representation of decision-making
*  Gradient Boosting - Is robust to outliers and missing data, making it suitable for real-world datasets
*  XGBoost - Because of consistently delivers accurate predictions

## Model performance
Based on the test set performance, the Lasso Regression model achieved the lowest MAE (22.69), making it the best-performing model among the ones tested
* Lasso Regression: MAE = 22.69
* Random Forest : MAE = 23.10
* XGBoost: MAE = 23.58
* Decision Tree : MAE = 24.23
* Gradient Boosting: MAE = 24.37
* Linear Regression: MAE = 31.53


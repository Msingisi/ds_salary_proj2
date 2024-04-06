# Data Science Salary Estimator: Project Overview
* Created a tool that estimates data science salaries (MAE ~ $ 22K) to help data scientists negotiate their income when they get a job.
* Scraped about 900 job descriptions from glassdoor using python and selenium
* Engineered features from the text of each job description to quantify the value companies put on python, excel, aws, and spark.
* Optimized Linear Regression, Decision Tree, Lasso Regression, Gradient Boosting, XGBoost, and Random Forest Regressors using GridsearchCV to reach the best model.

## Code and Resources Used
** Python version:** 3.12
** Packages:** pandas, numpy, sklearn, matplotlib, seaborn, selenium
** Scraper Github:** https://github.com/arapfaik/scraping-glassdoor-selenium
** Scraper Article:** https://mersakarya.medium.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905
** YouTube Project Walk-Through:** https://www.youtube.com/playlist?list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t

##Web Scraping
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

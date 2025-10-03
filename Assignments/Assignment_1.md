---
title: "Assignment 1"
author: "Dan Lizotte and Daniel Fuller"
output:
  html_document:
    keep_md: true
  pdf_document: default
---



## Supervised Learning

1. [ISLR CH2 p.52 Q2] Explain whether each of the following scenarios is a classification or regression problem, and indicate whether we are most interested in inference or prediction. Finally, provide $n$ and $p$.

(a) We collect a set of data on the top 500 firms in the US. For each firm we record profit, number of employees, industry and the CEO salary. We are interested in understanding which factors affect CEO salary.

```{}
Answer
```

(b) We are considering launching a new product and wish to know whether it will be a success or a failure. We collect data on 20 similar products that were previously launched. For each product we have recorded whether it was a success or failure, price charged for the product, marketing budget, competition price, and ten other variables.

```{}
Answer
```

(c) We are interested in predicting the % change in the USD/Euro exchange rate in relation to the weekly changes in the world stock markets. Hence we collect weekly data for all of 2012. For each week we record the % change in the USD/Euro, the % change in the US market, the % change in the British market, and the % change in the German market.

```{}
Answer
```

## Causation

2. [ISLR Ch4 p.191 Q6] Suppose we collect data for a group of students in a statistics class with variables X1 = hours studied, X2 = undergrad GPA, and Y=
receive an A. We fit a logistic regression and produce estimated coefficients as follows:

$β_0 =−6$, $β_1 = 0.05$, $β_2 = 1$.

(a) According to this model, what is the estimated probability that a student who studies for 40 h and has an undergrad GPA of 3.5 gets an A in the class?

```{}
Answer
```

(b) How many hours (approximately) would the student in part (a) need to study to have a 50 % chance of getting an A in the class?

```{}
Answer
```

3. Reflect on question 2(b) above.

a) Clearly describe the causal assumption implicit in the question.

```{}
Answer
```

b) Re-write the question in a way that does not rely on assuming causal relationships between inputs and outputs.

```{}
Answer
```

## Model Fit

4. This question should be answered using the `canpath_data.csv` available on the Canvas page. An example logistic regression is presented in the [Data Analysis](https://github.com/ai4ph-hrtp/foundations_ai_ml_course/blob/main/Data%20Analysis/logistic_regression.md) page of the Github repo for the course. You can use that as an example to run your regression. 

(a) Use the full data set to perform a logistic regression with `diabetes` as the response and the following variables as predictors
        * Age = SDC_AGE_CALC
        * Education level = SDC_EDU_LEVEL
        * Self reported general health = HS_GEN_HEALTH
        * Physical activity = PA_TOTAL_SHORT
        * Sleep Apnea = DIS_RESP_SLEEP_APNEA_EVER

Use the summary function to print the results.


``` r
# Your R code here
```

(b) Compute the confusion matrix and overall fraction of correct predictions. Explain what the confusion matrix is telling you about the types of mistakes made by logistic regression.


``` r
# Your R code here
```

```{}
Written answer.
```

(c) Now fit the logistic regression model using a 70/30 data split with 70% of the data for training and 30% for testing. Use the same 5 predictor variables you did in question 4(a). Compute the confusion matrix and the overall fraction of correct predictions for the held out data.


``` r
# Your R code here
```

(d) Repeat (c) using a hyperparameter optimization method of your choice (ie., ridge, lasso, or elastic net). Or, if you want more of a challenge, develop a new model using a different supervised ML approach (eg., Random Forest).


``` r
# Your R code here
```

## Model Generalization

5. In a few sentences, explain what your results from Question 4 tell you about the ability of the models you constructed to make accurate predictions on future data.

```{}
Answer
```

6. Suppose a colleagues suggests that you use cross validation to evaluate the models in Question 4. In a short paragraph, state whether or not you think this is a good idea and explain why or why not.

```{}
Answer
```

7. Using the same model developed in question 4, fit a logistic regression using a cross-validation approach. 

(a) Interpret the results of the regression with a cross-validation approach. 


``` r
# Your R code here
```

```{}
Your comments here
```

(b) Discuss the differences and similarities in the results of the logistic regression developed in question 4(a), question 4(c), and question 7(a). 


``` r
# Your R code here
```

```{}
Your comments here
```

8. Conduct the same analysis as question 4 using the Random Forest method. There is an example implementation in the [Data Analysis](https://github.com/ai4ph-hrtp/foundations_ai_ml_course/blob/main/Data%20Analysis/random_forest.md) section on Github. For an extra challenge include cross-validation and hyperparameter tuning in your analysis. 


``` r
# Your R code here.
```

```{}
Your comments here
```

9. Based on the results, what is your opinion about the relative merits of the logistic regression model and the Random Forest model for this problem?

```{}
Your answer
```

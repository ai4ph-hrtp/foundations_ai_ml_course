---
title: "AI4PH 25FA: Assignment 2"
output:
  html_document:
    keep_md: true
  pdf_document: default
date: "Due Date: 2025-11-14"
author: "YOUR NAME HERE" 
---




Please complete this notebook and upload it to the course website in **PDF format**.  
Ensure that the knitted PDF has all necessary code and text output. 

The assignment is set up to use data available through R packages. You are welcome to instead use the CANPATH dataset for some or all of the coding questions, ensuring that the core modelling tasks remain the same. In this case please additionally add a brief description of your approach (e.g., types of variables being used and research question you want to answer) and whether you needed to change any key modelling decisions due to the different data source (e.g., distance metric to match data type).



## Written Questions 


### Question 1: Supervised vs Unsupervised Learning

Below are three scenarios. For each, indicate whether it is a supervised or unsupervised learning problem and briefly explain your reasoning.

*1.1 A hospital wants to predict whether a patient will be readmitted within 30 days based on past admission data.*

> **Answer:**  
> YOUR TEXT HERE  


*1.2 A researcher wants to find similar disease progression patterns from patient health records.*


> **Answer:**  
> YOUR TEXT HERE  


*1.3 A public health department categorizes neighborhoods based on socioeconomic and health indicators without prior knowledge of groups.*


> **Answer:**  
> YOUR TEXT HERE  


### Question 2: Clustering vs. Topic Modelling  

*2.1 List two differences between clustering and topic modelling.*  

> **Answer:**  
> YOUR TEXT HERE  

*2.2 Provide one example public health application where clustering would be more appropriate than topic modelling. Briefly explain why.*   

> **Answer:**  
> YOUR TEXT HERE  


*2.3 Provide one example public health application where topic modelling would be more appropriate than clustering. Briefly explain why.*  


> **Answer:**  
> YOUR TEXT HERE  


### Question 3: Distance Metrics 

Given the following points in a 2D space:

Point A: (2,3)

Point B: (5,7)

Point C: (1,2)

*Calculate the Euclidean and Manhattan distances between these points. Then briefly explain how these distance metrics differ in how they measure similarity.*


> **Answer:**   
> Point A and B:    
> Point A and C:     
> Point B and C:  

> YOUR EXPLANATION HERE  


## Coding Questions

### Question 4: k-Means Clustering

*Use the `USArrests` dataset from the `ISLR2` package to perform k-means clustering. Information on the dataset is available through R documentation.*    

Steps:

1. Load the dataset. You may want to do some exploratory analyses to get comfortable with the data. You do not need to show us this code. 


``` r
# Load required packages
library(tidyverse)
library(ISLR2)
library(cluster)
library(factoextra)

# Example load of the dataset 
data("USArrests")
```

2. Scale the variables to mean=0, sd=1. K-means is distance based so if the variables are not scale the variables with larger range will dominate the clustering. 


``` r
# YOUR CODE HERE 
```

3. Apply k-means clustering for 2 to 6 clusters.  


``` r
# YOUR CODE HERE 
```


4. Select one set of clusters to keep. Justify your choice. 

> **Answer:**   
> Final number of clusters selected:    
> Rationale:    


5. Visualize the clusters in your final model. Consider using the `fviz_cluster` function. 


``` r
# YOUR CODE HERE 
```


6. Interpret the results. 

> **Answer / Interpretation notes:**   
> Cluster 1 (red):   
> ...  
> Cluster n (colour):     


### Question 5: Hierarchical Clustering

*Use the same `USArrests` dataset from the `ISLR2` package as above to perform hierarchical clustering.*  

Steps:

1. What is the default distance metric used by `hclust`?   

> **Answer:**  
> YOUR ANSWER HERE 


2. Use `hclust` to perform agglomerative hierarchical clustering using complete linkage criteria and using single linkage criteria. Plot both versions. 


``` r
# load additional required packages
library(stats)

# load the data
data("USArrests")

# YOUR CODE HERE 
```


3. Compare the resulting dendrograms and briefly describe how the choice of linkage affects clustering. 


> **Answer:**   
> YOUR TEXT HERE  


5. Select one of the dendrograms and decide the number of clusters to keep. Justify this decision. 

> **Answer:**   
> YOUR TEXT HERE  


### Question 6: Topic Modeling with Latent Dirichlet Allocation (LDA)

*Use the `AssociatedPress` dataset from the `topicmodels` package. This dataset comes preprocessed. You can read more about the dataset in the R documentation.*   

Steps:

1. Load the dataset. 


``` r
# Load additional required packages
library(topicmodels)
library(tm)

# Example way to load the preprocessed dataset
data("AssociatedPress", package = "topicmodels")  
```

What is the automatically loaded data structure?   

> **Answer:**  
> YOUR TEXT HERE   



2. Fit LDA models with 2, 4, and 6 topics. Use the seed 123 to allow for reproducibility of results. 


``` r
# set seed 
set.seed(123)

# YOUR CODE HERE 
```

3. Select the best number of topics to keep. Describe how you made your selection. 

> **Answer:**  
> Number of topics selected:    
> Rationale:   


4. Extract the most important features for each topic.


``` r
# YOUR CODE HERE 
```


5. Provide a label for each topic based on your interpretation of its content.  

> **Answer:**  
> Topic 1: 
> .... 
> Topic n: 

 


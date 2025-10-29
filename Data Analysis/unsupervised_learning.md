---
title: "Unsupervised Learning"
output:
      html_document:
        keep_md: true
---


``` r
knitr::opts_chunk$set(echo = TRUE)
library(tidymodels)
```

```
## ── Attaching packages ────────────────────────────────────── tidymodels 1.4.1 ──
```

```
## ✔ broom        1.0.10     ✔ recipes      1.3.1 
## ✔ dials        1.4.2      ✔ rsample      1.3.1 
## ✔ dplyr        1.1.4      ✔ tailor       0.1.0 
## ✔ ggplot2      4.0.0      ✔ tidyr        1.3.1 
## ✔ infer        1.0.9      ✔ tune         2.0.0 
## ✔ modeldata    1.5.1      ✔ workflows    1.3.0 
## ✔ parsnip      1.3.3      ✔ workflowsets 1.1.1 
## ✔ purrr        1.1.0      ✔ yardstick    1.3.2
```

```
## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
## ✖ purrr::discard() masks scales::discard()
## ✖ dplyr::filter()  masks stats::filter()
## ✖ dplyr::lag()     masks stats::lag()
## ✖ recipes::step()  masks stats::step()
```

``` r
library(tidyclust)
```

```
## 
## Attaching package: 'tidyclust'
```

```
## The following objects are masked from 'package:parsnip':
## 
##     knit_engine_docs, list_md_problems
```

``` r
library(ClusterR)
library(klaR)
```

```
## Loading required package: MASS
```

```
## 
## Attaching package: 'MASS'
```

```
## The following object is masked from 'package:dplyr':
## 
##     select
```

``` r
library(GGally)
library(knitr)
library(gtsummary)
```

```
## 
## Attaching package: 'gtsummary'
```

```
## The following object is masked from 'package:MASS':
## 
##     select
```

``` r
library(devtools)
```

```
## Loading required package: usethis
```

```
## 
## Attaching package: 'devtools'
```

```
## The following object is masked from 'package:recipes':
## 
##     check
```

``` r
library(ggmosaic)
```

```
## 
## Attaching package: 'ggmosaic'
```

```
## The following object is masked from 'package:GGally':
## 
##     happy
```

``` r
library(tune)
library(recipes)
library(yardstick)
library(parsnip)
library(tidyverse)
```

```
## ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
## ✔ forcats   1.0.0     ✔ stringr   1.5.2
## ✔ lubridate 1.9.4     ✔ tibble    3.3.0
## ✔ readr     2.1.5
```

```
## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
## ✖ readr::col_factor() masks scales::col_factor()
## ✖ purrr::discard()    masks scales::discard()
## ✖ dplyr::filter()     masks stats::filter()
## ✖ stringr::fixed()    masks recipes::fixed()
## ✖ dplyr::lag()        masks stats::lag()
## ✖ gtsummary::select() masks MASS::select(), dplyr::select()
## ✖ readr::spec()       masks yardstick::spec()
## ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors
```

# Unsupervised Learning

## Research question and data

We are using a version of the CanPath student dataset [https://canpath.ca/student-dataset/](https://canpath.ca/student-dataset/). The nice thing about this dataset is that it's pretty big in terms of sample size, has lots of variables, and we can use it for free. We have a large number of variables about people's disease status and we want to develop an indicator of multimorbity using unsupervised learning. 

Our research question is:  

- **Can we develop an indicator of multimorbidity and identify grouping of co-occuring disease in the sample**

### Reading in data

Here are reading in data and getting organized to run our models. 


``` r
data <- read_csv("canpath_data.csv")
```

```
## Rows: 41187 Columns: 93
## ── Column specification ────────────────────────────────────────────────────────
## Delimiter: ","
## chr  (1): ID
## dbl (92): ADM_STUDY_ID, SDC_GENDER, SDC_AGE_CALC, SDC_MARITAL_STATUS, SDC_ED...
## 
## ℹ Use `spec()` to retrieve the full column specification for this data.
## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

``` r
data <- data %>% mutate_at(3, factor)
data <- data %>% mutate_at(5:6, factor)
data <- data %>% mutate_at(8:12, factor)
data <- data %>% mutate_at(15:81, factor)
data <- data %>% mutate_at(83:93, factor)

data$ID <- NULL
data$ADM_STUDY_ID <- NULL
```

### Disease Status

We have a number of variables with the labels `DIS_` we are going to use those to develop our unsupervised learning model. These variables are all categorical and are coded as the following 

* DIS_*_EVER	0	Never had disease
* DIS_*_EVER	1	Ever had disease
* DIS_*_EVER	2	Presumed - Never had disease

We are going to recode these to 

* 0 Never had disease
* 1 Presumed - Never had disease
* 2 Ever had disease

Let's filter the data so we only have disease for the individuals. There are also variables that have family history that have the `_FAM_` or `_SIB_` or `_CHILD_` flags in the variable name. We don't really want those. 


``` r
data <- data %>% dplyr::select(!c(contains("_FAM_") | contains("_SIB_") | contains("_CHILD_")))
```

Let's create a subset of the data with only the disease variables


``` r
data_disease <- data %>% dplyr::select(contains("DIS_"))
glimpse(data_disease)
```

```
## Rows: 41,187
## Columns: 27
## $ DIS_HBP_EVER              <fct> 0, 0, 1, 0, 1, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0,…
## $ DIS_MI_EVER               <fct> 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_STROKE_EVER           <fct> 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_ASTHMA_EVER           <fct> 0, 0, 2, 0, 1, 2, 0, 1, 0, 0, 1, 0, 0, 0, 0,…
## $ DIS_COPD_EVER             <fct> 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_DEP_EVER              <fct> 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,…
## $ DIS_DIAB_EVER             <fct> 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_LC_EVER               <fct> 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_CH_EVER               <fct> 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_CROHN_EVER            <fct> 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_UC_EVER               <fct> 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_IBS_EVER              <fct> 1, 0, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_ECZEMA_EVER           <fct> 0, 0, 2, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,…
## $ DIS_SLE_EVER              <fct> 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_PS_EVER               <fct> 0, 0, 2, 1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_MS_EVER               <fct> 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_OP_EVER               <fct> 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_ARTHRITIS_EVER        <fct> 0, 1, 1, 1, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0,…
## $ DIS_CANCER_EVER           <fct> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_CANCER_F_EVER         <fct> 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1,…
## $ DIS_CANCER_M_EVER         <fct> 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0,…
## $ DIS_ENDO_HB_CHOL_EVER     <fct> 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1,…
## $ DIS_CARDIO_HD_EVER        <fct> 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_RESP_SLEEP_APNEA_EVER <fct> 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_MH_ANXIETY_EVER       <fct> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_MH_ADDICTION_EVER     <fct> 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,…
## $ DIS_NEURO_MIGRAINE_EVER   <fct> 1, 0, 2, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,…
```

``` r
data_disease <- data_disease %>% 
     mutate_all(list(rec = ~ recode(., 
                                    "0" = "0", 
                                    "2" = "1", 
                                    "1" = "2", 
                                    .default = NA_character_)))

table(data_disease$DIS_HBP_EVER, data_disease$DIS_HBP_EVER_rec)
```

```
##    
##         0     2     1
##   0 29994     0     0
##   1     0  9922     0
##   2     0     0  1271
```

``` r
data_disease <- data_disease %>% select(28:54)

data_disease <- data_disease %>% mutate_at(1:27, as.integer)
```

We have 27 disease status variables in the data and we want to create clusters of commonly groups diseases. 

We are going to use K-means cluster here to understand which disease tend to go together. K-means is very common method for clustering (read: unsupervised learning) but it only considers continuous predictor variables. Here the data are actually categorical and we could use a method like k-medians or k-modes, which allow categorical predictors but it's much harder to visualize and understand what is happening with other methods.... so we are just considering our categorical variables as continuous. We are again using `Tidymodels` for this so the analysis will have a similar set. 

In the `Tidymodels` framework we setup a workflow. A workflow must includ the following

* A Recipe `add_recipe` which is how we tell the workflow to process the data.
* A Model `add_model` which specifies the model paramaters

We will also set seed to ensure reproducibility. 

### Model 

There are different ways to fit this model, and the method of estimation is chosen by setting the model engine. The engine-specific pages for this model are listed below.

* stats: Classical K-means
* ClusterR: Classical K-means
* klaR: K-Modes
* clustMixType: K-prototypes

We are using the `stats` package. 


``` r
set.seed(10)

### Model
kmeans_model <- k_means(num_clusters = 3) %>%
                  set_engine("stats")
kmeans_model
```

```
## K Means Cluster Specification (partition)
## 
## Main Arguments:
##   num_clusters = 3
## 
## Computational engine: stats
```

### Recipe 

The recipe is just the model we are going to run. In the logsitic regression example we say more things happening in the recipe but this is straightforward because all of the data are the same units and we are just doing listwise deletion on the missing. 


``` r
### Recipe 
kmeans_recipe <- recipe( ~., data = data_disease)
```

### Workflow

Once we have the recipe we can run the workflow to have and object that will let us fit the model. This will come in handy later when we want to run a few different iterations of the model to try and understand which number clusters is the best. 


``` r
set.seed(10)

### Workflow
kmeans_workflow <- workflow() %>%
    add_recipe(kmeans_recipe) %>%
    add_model(kmeans_model)
```

### Fit the model

Now we fit the model. 


``` r
set.seed(10)

kmeans_fit <- kmeans_workflow %>% fit(data=data_disease)

kmeans_fit
```

```
## ══ Workflow [trained] ══════════════════════════════════════════════════════════
## Preprocessor: Recipe
## Model: k_means()
## 
## ── Preprocessor ────────────────────────────────────────────────────────────────
## 0 Recipe Steps
## 
## ── Model ───────────────────────────────────────────────────────────────────────
## K-means clustering with 3 clusters of sizes 25528, 1861, 13798
## 
## Cluster means:
##   DIS_HBP_EVER_rec DIS_MI_EVER_rec DIS_STROKE_EVER_rec DIS_ASTHMA_EVER_rec
## 1         1.187950        1.025306            1.020213            1.132639
## 3         2.183235        2.498657            2.619559            2.418592
## 2         1.395999        1.109726            1.104725            1.169807
##   DIS_COPD_EVER_rec DIS_DEP_EVER_rec DIS_DIAB_EVER_rec DIS_LC_EVER_rec
## 1          1.012927         1.119555          1.049593        1.006346
## 3          2.467491         2.186459          2.272972        2.702848
## 2          1.077113         1.172924          1.159226        1.035730
##   DIS_CH_EVER_rec DIS_CROHN_EVER_rec DIS_UC_EVER_rec DIS_IBS_EVER_rec
## 1        1.008814           1.009166        1.016022         1.085553
## 3        2.691026           2.832886        2.831811         2.699624
## 2        1.039716           1.048340        1.055080         1.124946
##   DIS_ECZEMA_EVER_rec DIS_SLE_EVER_rec DIS_PS_EVER_rec DIS_MS_EVER_rec
## 1            1.121083         1.007247        1.051160        1.012104
## 3            2.326169         2.549704        2.427190        2.529823
## 2            1.147485         1.054573        1.104218        1.068778
##   DIS_OP_EVER_rec DIS_ARTHRITIS_EVER_rec DIS_CANCER_EVER_rec
## 1        1.047281               1.210122            1.079521
## 3        2.564213               2.358947            1.198818
## 2        1.137121               1.339180            1.160386
##   DIS_CANCER_F_EVER_rec DIS_CANCER_M_EVER_rec DIS_ENDO_HB_CHOL_EVER_rec
## 1              1.286156              1.264376                  1.000000
## 3              1.372918              1.364320                  1.628694
## 2              1.338310              1.324032                  2.090593
##   DIS_CARDIO_HD_EVER_rec DIS_RESP_SLEEP_APNEA_EVER_rec DIS_MH_ANXIETY_EVER_rec
## 1               1.028753                      1.074663                1.114619
## 3               2.683503                      2.380441                1.479850
## 2               1.215321                      1.237208                1.211045
##   DIS_MH_ADDICTION_EVER_rec DIS_NEURO_MIGRAINE_EVER_rec
## 1                  1.029928                    1.176003
## 3                  2.491671                    2.340677
## 2                  1.215466                    1.477316
## 
## Clustering vector:
##     [1] 1 1 2 1 2 2 1 3 1 1 1 3 1 1 3 1 1 3 1 2 1 1 3 1 1 1 1 2 1 3 1 1 3 1 1 1
##    [37] 1 3 3 3 1 2 3 3 1 3 1 2 1 1 1 1 1 2 1 1 1 1 1 1 1 3 1 3 1 1 1 3 3 1 2 1
##    [73] 1 1 1 1 2 1 3 1 1 2 1 3 2 1 3 1 1 1 3 1 3 1 3 1 1 1 3 3 1 1 3 3 1 1 3 1
##   [109] 1 1 1 1 1 1 1 3 3 2 1 3 2 2 1 1 1 3 1 1 1 3 1 1 1 1 1 1 1 2 1 1 1 2 1 3
##   [145] 3 1 2 3 1 2 2 1 3 3 1 1 3 1 2 1 1 3 1 1 1 1 1 3 1 3 3 1 1 1 3 1 1 1 1 1
##   [181] 3 2 2 1 1 2 1 3 1 3 1 2 3 3 1 2 3 1 1 1 3 2 1 1 3 3 3 1 2 1 3 2 1 1 1 1
##   [217] 3 1 2 3 1 1 1 2 2 1 1 1 3 3 1 1 1 2 3 3 1 1 2 3 2 1 3 3 1 1 2 1 3 1 1 1
##   [253] 3 1 1 3 1 1 3 1 1 1 2 1 1 1 1 3 3 3 3 1 3 1 1 1 3 3 1 2 2 1 2 1 1 1 2 1
##   [289] 1 1 1 1 1 1 1 3 1 1 1 3 1 1 2 1 1 3 3 1 1 2 3 3 2 2 2 3 2 1 1 3 1 1 1 1
##   [325] 1 1 1 2 1 1 2 1 3 1 3 1 1 1 1 1 1 1 1 1 1 1 1 3 3 2 1 1 2 1 2 1 3 1 2 3
##   [361] 3 3 1 1 1 1 3 3 1 1 3 1 1 3 3 1 1 1 1 3 1 1 1 1 1 1 1 1 1 1 1 3 3 1 2 1
##   [397] 1 1 3 1 1 2 1 1 3 1 3 2 1 1 1 2 3 3 1 3 1 3 3 3 1 2 1 1 1 1 3 1 1 2 1 1
##   [433] 1 2 1 3 2 3 3 3 1 1 2 1 1 1 1 3 1 1 3 2 1 1 3 1 1 1 1 3 3 2 2 1 2 1 1 1
## 
## ...
## and 1141 more lines.
```

The output shows us that we have 3 clusters (we defined that at the begining) with sample sizes of 25528, 1861, 13798 in each cluster. 

We see the means for each cluster for each variable and which person is in which cluster. 

### Save the cluster variable

Now we want to save the cluster variable and put it back in our dataset. That we we can visualize what is happening with the clusters and the specific variables and try and make sense of which cluster variables tend to go together. 


``` r
clusters <- kmeans_fit %>% extract_cluster_assignment()
clusters <- as.data.frame(clusters)

names(clusters) <- c("cluster")
data$clusters <- clusters$cluster
```

Let's make a pairs plot to visualize how the clusters hang together with the actual variables. We will only do this for a few variables because otherwise it's a bit too much to visualize. 


``` r
data %>%
    dplyr::select(c("DIS_HBP_EVER", "DIS_MI_EVER", "DIS_STROKE_EVER", "DIS_ASTHMA_EVER", "DIS_COPD_EVER", "DIS_DEP_EVER", "clusters")) %>%
    ggpairs(aes(fill = clusters, color = clusters))
```

![](unsupervised_learning_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

We can also visualize the centroid of each cluster for each variable


``` r
centroids <- extract_centroids(kmeans_fit)

centroids_long <- centroids %>% pivot_longer(cols=c("DIS_HBP_EVER_rec", "DIS_MI_EVER_rec", "DIS_STROKE_EVER_rec", "DIS_ASTHMA_EVER_rec", "DIS_COPD_EVER_rec", "DIS_DEP_EVER_rec", "DIS_DIAB_EVER_rec", "DIS_LC_EVER_rec", "DIS_CH_EVER_rec", "DIS_CROHN_EVER_rec", "DIS_UC_EVER_rec", "DIS_IBS_EVER_rec", "DIS_ECZEMA_EVER_rec", "DIS_SLE_EVER_rec", "DIS_PS_EVER_rec", "DIS_MS_EVER_rec", "DIS_OP_EVER_rec", "DIS_ARTHRITIS_EVER_rec", "DIS_CANCER_EVER_rec", "DIS_CANCER_F_EVER_rec", "DIS_CANCER_M_EVER_rec", "DIS_ENDO_HB_CHOL_EVER_rec", "DIS_CARDIO_HD_EVER_rec", "DIS_RESP_SLEEP_APNEA_EVER_rec", "DIS_RESP_SLEEP_APNEA_EVER_rec", "DIS_MH_ANXIETY_EVER_rec", "DIS_MH_ADDICTION_EVER_rec", "DIS_NEURO_MIGRAINE_EVER_rec"))

ggplot(data = centroids_long, aes(x = name, y = value, group = .cluster, color = .cluster)) +
    geom_point() +
    geom_line() +
    labs(x="", y="Value at cluster center") + 
  theme(axis.text.x = element_text(angle=45, hjust = 1))
```

![](unsupervised_learning_files/figure-html/unnamed-chunk-10-1.png)<!-- -->

Here we can see that cluster 2, which is the least common with has a relatively high proportion of all disease. Cluster 3 has a high proportion of 3 diseases, `ENDO_HB_CHOL`, `NEURO_MIGRAINE`, and `SLEEP_APNEA`. 

### How many clusters? 

A key question with cluster analysis is how many clusters to select. We can `tune()` the number of clusters parameter to try and get to a better solution for this. We do this using the tune parameter on the num_clusters function in the model step of tidymodels. 


``` r
### Model with tuning of cluster #
kmeans_model_tune <- k_means(num_clusters = tune()) %>%
                  set_engine("stats")
```


``` r
### Workflow
kmodes_workflow_tune <- workflow() %>%
    add_recipe(kmeans_recipe) %>%
    add_model(kmeans_model_tune)

folds <- vfold_cv(data_disease, v = 2)
grid <- tibble(num_clusters=1:10)
```

## NOT RUN

The code below takes around 3 hours on a M4 Mac Mini so run with caution. 


``` r
tuned_model <- tune_cluster(kmodes_workflow_tune, 
                        resamples = folds, 
                        grid = grid,
                       metrics = cluster_metric_set(silhouette_avg), 
                       control = control_resamples(save_pred = TRUE, 
                                                  verbose = TRUE)
                       )
```

```
## i Fold1: preprocessor 1/1
```

```
## ✓ Fold1: preprocessor 1/1
```

```
## i Fold1: preprocessor 1/1, model 1/10
```

```
## ✓ Fold1: preprocessor 1/1, model 1/10
```

```
## i Fold1: preprocessor 1/1, model 1/10 (predictions)
```

```
## i Fold1: preprocessor 1/1, model 2/10
```

```
## ✓ Fold1: preprocessor 1/1, model 2/10
```

```
## i Fold1: preprocessor 1/1, model 2/10 (predictions)
```

```
## i Fold1: preprocessor 1/1, model 3/10
```

```
## ✓ Fold1: preprocessor 1/1, model 3/10
```

```
## i Fold1: preprocessor 1/1, model 3/10 (predictions)
```

```
## i Fold1: preprocessor 1/1, model 4/10
```

```
## ✓ Fold1: preprocessor 1/1, model 4/10
```

```
## i Fold1: preprocessor 1/1, model 4/10 (predictions)
```

```
## i Fold1: preprocessor 1/1, model 5/10
```

```
## ✓ Fold1: preprocessor 1/1, model 5/10
```

```
## i Fold1: preprocessor 1/1, model 5/10 (predictions)
```

```
## i Fold1: preprocessor 1/1, model 6/10
```

```
## ✓ Fold1: preprocessor 1/1, model 6/10
```

```
## i Fold1: preprocessor 1/1, model 6/10 (predictions)
```

```
## i Fold1: preprocessor 1/1, model 7/10
```

```
## ✓ Fold1: preprocessor 1/1, model 7/10
```

```
## i Fold1: preprocessor 1/1, model 7/10 (predictions)
```

```
## i Fold1: preprocessor 1/1, model 8/10
```

```
## ✓ Fold1: preprocessor 1/1, model 8/10
```

```
## i Fold1: preprocessor 1/1, model 8/10 (predictions)
```

```
## i Fold1: preprocessor 1/1, model 9/10
```

```
## ✓ Fold1: preprocessor 1/1, model 9/10
```

```
## i Fold1: preprocessor 1/1, model 9/10 (predictions)
```

```
## i Fold1: preprocessor 1/1, model 10/10
```

```
## ✓ Fold1: preprocessor 1/1, model 10/10
```

```
## i Fold1: preprocessor 1/1, model 10/10 (predictions)
```

```
## i Fold2: preprocessor 1/1
```

```
## ✓ Fold2: preprocessor 1/1
```

```
## i Fold2: preprocessor 1/1, model 1/10
```

```
## ✓ Fold2: preprocessor 1/1, model 1/10
```

```
## i Fold2: preprocessor 1/1, model 1/10 (predictions)
```

```
## i Fold2: preprocessor 1/1, model 2/10
```

```
## ✓ Fold2: preprocessor 1/1, model 2/10
```

```
## i Fold2: preprocessor 1/1, model 2/10 (predictions)
```

```
## i Fold2: preprocessor 1/1, model 3/10
```

```
## ✓ Fold2: preprocessor 1/1, model 3/10
```

```
## i Fold2: preprocessor 1/1, model 3/10 (predictions)
```

```
## i Fold2: preprocessor 1/1, model 4/10
```

```
## ✓ Fold2: preprocessor 1/1, model 4/10
```

```
## i Fold2: preprocessor 1/1, model 4/10 (predictions)
```

```
## i Fold2: preprocessor 1/1, model 5/10
```

```
## ✓ Fold2: preprocessor 1/1, model 5/10
```

```
## i Fold2: preprocessor 1/1, model 5/10 (predictions)
```

```
## i Fold2: preprocessor 1/1, model 6/10
```

```
## ✓ Fold2: preprocessor 1/1, model 6/10
```

```
## i Fold2: preprocessor 1/1, model 6/10 (predictions)
```

```
## i Fold2: preprocessor 1/1, model 7/10
```

```
## ✓ Fold2: preprocessor 1/1, model 7/10
```

```
## i Fold2: preprocessor 1/1, model 7/10 (predictions)
```

```
## i Fold2: preprocessor 1/1, model 8/10
```

```
## ✓ Fold2: preprocessor 1/1, model 8/10
```

```
## i Fold2: preprocessor 1/1, model 8/10 (predictions)
```

```
## i Fold2: preprocessor 1/1, model 9/10
```

```
## ✓ Fold2: preprocessor 1/1, model 9/10
```

```
## i Fold2: preprocessor 1/1, model 9/10 (predictions)
```

```
## i Fold2: preprocessor 1/1, model 10/10
```

```
## ✓ Fold2: preprocessor 1/1, model 10/10
```

```
## i Fold2: preprocessor 1/1, model 10/10 (predictions)
```

Here we see the results of all of the 20 models we ran. We can plot the silhouette score across each model to better understand which number of clusters is optimal. 


``` r
collect_metrics(tuned_model) %>% head()
```

```
## # A tibble: 6 × 7
##   num_clusters .metric        .estimator    mean     n  std_err .config         
##          <int> <chr>          <chr>        <dbl> <int>    <dbl> <chr>           
## 1            1 silhouette_avg standard   NaN         0 NA       Preprocessor1_M…
## 2            2 silhouette_avg standard     0.672     2  0.00606 Preprocessor1_M…
## 3            3 silhouette_avg standard     0.248     2  0.0543  Preprocessor1_M…
## 4            4 silhouette_avg standard     0.158     2  0.0243  Preprocessor1_M…
## 5            5 silhouette_avg standard     0.126     2  0.00142 Preprocessor1_M…
## 6            6 silhouette_avg standard     0.314     2  0.137   Preprocessor1_M…
```

``` r
autoplot(tuned_model)
```

![](unsupervised_learning_files/figure-html/unnamed-chunk-14-1.png)<!-- -->

The silhouette plot suggests that the 5 cluster solution probably optimal. 

## Ressources

* [https://gedeck.github.io/DS-6030/book/clustering.html](https://gedeck.github.io/DS-6030/book/clustering.html)
* [https://tidyclust.tidymodels.org/index.html](https://tidyclust.tidymodels.org/index.html)
* [https://cran.r-project.org/web/packages/ClusterR/ClusterR.pdf](https://cran.r-project.org/web/packages/ClusterR/ClusterR.pdf)


``` r
sessionInfo()
```

```
## R version 4.5.1 (2025-06-13)
## Platform: aarch64-apple-darwin20
## Running under: macOS Sequoia 15.7.1
## 
## Matrix products: default
## BLAS:   /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libRblas.0.dylib 
## LAPACK: /Library/Frameworks/R.framework/Versions/4.5-arm64/Resources/lib/libRlapack.dylib;  LAPACK version 3.12.1
## 
## locale:
## [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
## 
## time zone: America/Regina
## tzcode source: internal
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
##  [1] lubridate_1.9.4    forcats_1.0.0      stringr_1.5.2      readr_2.1.5       
##  [5] tibble_3.3.0       tidyverse_2.0.0    ggmosaic_0.4.0     devtools_2.4.6    
##  [9] usethis_3.2.1      gtsummary_2.4.0    knitr_1.50         GGally_2.4.0      
## [13] klaR_1.7-3         MASS_7.3-65        ClusterR_1.3.5     tidyclust_0.2.4   
## [17] yardstick_1.3.2    workflowsets_1.1.1 workflows_1.3.0    tune_2.0.0        
## [21] tidyr_1.3.1        tailor_0.1.0       rsample_1.3.1      recipes_1.3.1     
## [25] purrr_1.1.0        parsnip_1.3.3      modeldata_1.5.1    infer_1.0.9       
## [29] ggplot2_4.0.0      dplyr_1.1.4        dials_1.4.2        scales_1.4.0      
## [33] broom_1.0.10       tidymodels_1.4.1  
## 
## loaded via a namespace (and not attached):
##   [1] RColorBrewer_1.1-3  rstudioapi_0.17.1   jsonlite_2.0.0     
##   [4] magrittr_2.0.4      modeltools_0.2-24   farver_2.1.2       
##   [7] rmarkdown_2.29      fs_1.6.6            vctrs_0.6.5        
##  [10] memoise_2.0.1       sparsevctrs_0.3.4   htmltools_0.5.8.1  
##  [13] haven_2.5.5         sass_0.4.10         parallelly_1.45.1  
##  [16] bslib_0.9.0         htmlwidgets_1.6.4   plyr_1.8.9         
##  [19] plotly_4.11.0       cachem_1.1.0        iterators_1.0.14   
##  [22] mime_0.13           lifecycle_1.0.4     pkgconfig_2.0.3    
##  [25] Matrix_1.7-3        R6_2.6.1            fastmap_1.2.0      
##  [28] future_1.67.0       shiny_1.11.1        digest_0.6.37      
##  [31] furrr_0.3.1         pkgload_1.4.1       philentropy_0.9.0  
##  [34] labeling_0.4.3      productplots_0.1.2  timechange_0.3.0   
##  [37] httr_1.4.7          compiler_4.5.1      remotes_2.5.0      
##  [40] bit64_4.6.0-1       withr_3.0.2         S7_0.2.0           
##  [43] backports_1.5.0     ggstats_0.11.0      pkgbuild_1.4.8     
##  [46] highr_0.11          lava_1.8.1          sessioninfo_1.2.3  
##  [49] tools_4.5.1         otel_0.2.0          flexclust_1.5.0    
##  [52] httpuv_1.6.16       future.apply_1.20.0 nnet_7.3-20        
##  [55] glue_1.8.0          questionr_0.8.1     promises_1.4.0     
##  [58] grid_4.5.1          modelenv_0.2.0      cluster_2.1.8.1    
##  [61] generics_0.1.4      gtable_0.3.6        labelled_2.15.0    
##  [64] tzdb_0.5.0          class_7.3-23        data.table_1.17.8  
##  [67] hms_1.1.3           utf8_1.2.6          foreach_1.5.2      
##  [70] ggrepel_0.9.6       pillar_1.11.1       vroom_1.6.5        
##  [73] later_1.4.4         splines_4.5.1       lhs_1.2.0          
##  [76] lattice_0.22-7      bit_4.6.0           survival_3.8-3     
##  [79] gmp_0.7-5           tidyselect_1.2.1    miniUI_0.1.2       
##  [82] stats4_4.5.1        xfun_0.53           hardhat_1.4.2      
##  [85] timeDate_4041.110   stringi_1.8.7       DiceDesign_1.10    
##  [88] lazyeval_0.2.2      yaml_2.3.10         evaluate_1.0.5     
##  [91] codetools_0.2-20    cli_3.6.5           rpart_4.1.24       
##  [94] xtable_1.8-4        jquerylib_0.1.4     Rcpp_1.1.0         
##  [97] globals_0.18.0      parallel_4.5.1      ellipsis_0.3.2     
## [100] gower_1.0.2         GPfit_1.0-9         listenv_0.9.1      
## [103] viridisLite_0.4.2   ipred_0.9-15        prodlim_2025.04.28 
## [106] crayon_1.5.3        combinat_0.0-8      rlang_1.1.6
```


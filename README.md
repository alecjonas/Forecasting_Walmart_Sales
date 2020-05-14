# Forecasting Walmart Sales

## Question
How accurately can I predict Walmart sales down to the day?

## Introduction
In 2019, Walmart was the largest retailer in sales beating second place by more than three times (Amazon). Behind the scenes, a massive support operation is conducted in order to make sure customer demand is met with appropriate customer service and supply of product. 

The goal of my project is to use real Walmart data to build a forecasting model that can accurately predict sales for any given product in any given month. The application of this model will allow for better customer service and increased sales.

The complexities of this model are vast for a multitude of reasons. The future contains an element of unpredictability. However, elements can be extrapolated from historical data which can be used to make predictions about the future. For example, I analyzed the overall sales trend for Walmart in addition to seasonality of sales (i.e. do some products sell better in summer than winter). Additionally, I put a lot of thought into figuring out how to not only forecast, but also apply that prediction to a monthly distribution that can estimate daily product sales. My methodology, analysis, and results will be explored in the remainder of this report.

## Data
The data set comes from Kaggle.com. It contains three files: 1) The training set (to be discussed), 2) A Calendar, and 3) a list of products and their sales price.

The majority of my work was done in the training data set. This file contains a data set, which is 30490 rows by 1919 columns. This means that there are 30,490 individual products and 1,913 days of sales for each product. The remaining six columns include a product id, item id, department id, , category id, store id, and state id. The data set contains product information from 3 states (California, Texas, and Wisconsin), 3 categories of products (Food, Hobby, and Household), 7 departments (3 for the Food cateogry, and 2 each for Hobby and Household), and 10 stores (4 in CA, and 3 in TX/WI).

## Analysis
Because Walmarts sales are dependent on the departments and the stores (which are influenced by the categories, states, and overall Walmart trends), I spent a lot of time thinking about how I would A) generate a forecast and then B) apply that forecast to the individual level. I decided to forecast at a higher level almost immediately because I felt that it would be more volatile to forecast on the indiviudal product level since many items only sparesly sell during the month. My original blueprint was as follows:

1) Forecast Total Walmart Sales
2) Trickle down the forecast to the individual level by multiplying the means of each sub category until I got to the store and department level.
3) Use the store/department level forecast to figure out individual product sales for the month
4) Figure out how products typically sell through the course of the month to find the daily sales

Here is the Forecast for Total Walmart Sales:










#### References
* https://www.kaggle.com/c/m5-forecasting-accuracy
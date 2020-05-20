# Forecasting Walmart Sales

## Question
Given a data set with five and a half years of product sales, how accurately can I predict Walmart sales down to the day?

## Introduction
In 2019, Walmart was the largest retailer in sales beating second place by more than three times (Amazon). Behind the scenes, a massive support operation is conducted in order to make sure customer demand is met with appropriate customer service and supply of product. 

The goal of my project is to use real Walmart data to build a forecasting model that can accurately predict sales for any given product in any given month. The application of this model will allow for better customer service and increased sales. It also will allow for a better understanding of the trend and seasonality of Walmart's business. This information will be able to help staff departments in order to meet customer demand, and also make sure that the appropriate supply from vendors is maintained to meet that demand. The information to be gained from this model has many benefits and a wide breadth of application.

The complexities of this model are vast for a multitude of reasons. The future contains an element of unpredictability. However, elements can be extrapolated from historical data which can be used to make predictions about the future. My methodology, analysis, and results will be explored in the remainder of this report.

## Data
The data set comes from Kaggle.com. It contains three files: 1) The training set (to be discussed), 2) A Calendar, and 3) a list of products and their sales price. The data set covers sales from Jan 2011- April 2016.

The majority of my work was done in the training data set. This file contains a data set, which is 30,490 rows by 1,919 columns. This means that there are 30,490 individual products and 1,913 days of sales for each product. The remaining six columns include a product id, item id, department id, , category id, store id, and state id. The data set contains product information from 3 states (California, Texas, and Wisconsin), 3 categories of products (Food, Hobby, and Household), 7 departments (3 for the Food category, and 2 each for Hobby and Household), and 10 stores (4 in CA, and 3 in TX/WI). It's also important to note that Walmart anonymized their data so product IDs do not give much information about the actual product.

## Forecasting Methodology
For this project, I used time series analysis. While there are several methods to forecast using time series, I used a trend and seasonality approach. In a given category, I made a line of best fit and set that as the trend. I then extrapolated the seasonality, which remains constant over time, and added in that component to the trend depending on which month I was forecasting. I also decided to forecast on the monthly sales, as opposed to the daily sales, in order to smooth the trend and seasonality.

Here are two sets of images that were primarily used in my analysis:

Here is a plot of Walmart's total sales and a trendline:
![Image](images/big_all_lin_trend.png)

Here is a plot of the raw data, trend, seasonality, and residuals for Walmart's total sales:
![Image](images/big_all_tsr.png)

Here is a plot of the CA3 store and Foods3 department sales and a trendline:
![Image](images/ca3_foods3_lin_trend.png)

Here is a plot of the raw data, trend, seasonality, and residuals for the CA3 store and Foods3 department sales:
![Image](images/ca3_foods3_tsr.png)

The second and fourth graph is a trend, seasonal, and residual plot. I used the seasonality component from these plots and the blue trendline from the first and third graph to conduct my analysis.

The overarching forecasting was conducted this way:
* Forecast = Trend + Seasonality

Since the trendline is a straight line (blue line in graphs), it can be used to predict the future. And then the seasonality is added in to adjust for the month.

## Analysis
Because Walmarts sales are dependent on the departments and the stores (which are influenced by the categories, states, and overall Walmart trends), I spent a lot of time thinking about how I would A) generate a forecast and then B) apply that forecast to the individual level. I decided to forecast at a higher level almost immediately because I felt that it would be more volatile to forecast on the individual product level since many items only sparsely sell during the month. For all of my analysis, I tested my accuracy by predicting February and March 2016 since I had data for these months already. This allowed me to assess any error and make adjustments.

My original blueprint was as follows:

1) Forecast Total Walmart Sales
2) Trickle down the forecast to the individual level by multiplying the means of each sub category until I got to the store and department level.
3) Use the store/department level forecast to figure out individual product sales for the month
4) Figure out how products typically sell through the course of the month to find the daily sales

For this test, I decided to predict along the following pipeline:
1) Total Sales for February
2) Percent of total sales allocated to the Foods Category
3) Percent of total Foods sales allocated to the Foods 3 Department across all Stores and States.
4) Percent of total Foods 3 Department sales allocated to California.
5) Percent of total California sales allocated to the CA3 store.

This trickle down approach is a method for taking a high level forecast, multiplying by the means for various sub categories, until I arrive at a specific store and department. Below are visualizations for the distribution of sales among different categories. The dotted lines represent the mean for that specific category (the mean is what I multiply my forecast by). Notice that in some graphs the means represent the entire historical data, and in other graphs the mean starts in a different spot. This is because some categories had major swings at some point in the past and I wanted the means to be believable for predicting the future.

Visualization for breakdown between the major product categories (left) and the states (right):
![Image](images/cats_states.png)

Visualization for the breakdown within each category (for example, there are three food departments within the entire food category):
![Image](images/sub_cats.png)

Visualization for the breakdown within each store in each state:
![Image](images/sub_stores.png)

Here is an example workflow, which I used to conduct a test:

| Trickle Down Forecast | Path |
| ----------- | ----------- |
| Feb 2016 Forecast | 1,166,909 Units |
| Food Category (orange dotted line in 'Major Categories') | x 67% | 
| Food 3 Department (purple dotted line in 'Foods Category') | x 70% |
| California State (orange dotted line in 'States') | x 43% | 
| CA3 Store (purple dotted line in 'CA Stores') | x 39% |
| Forecast for CA3 store Foods 3 Dept | = 92,050 Units | 

After I made my forecast for the month of February, I got concerned that by trickling down the pipeline that there would be data loss (due to the assumptions made such as the means). I decided that I needed to forecast directly on the CA3 store and Foods3 Department and and then compare my results to the trickle down method. By comparing two forecasting methods, I increased the likelihood that I would be using the most accurate forecast. 

I compared my results from the trickle down forecast to the direct forcast and noticed that my accuracy was significantly improved forecasting directly on the store and department. I decided to purse this forecasting strategy as my primary method from this point forward.

Now that I had my forecast at the Store and Department level, it was now time to work on breaking that forecast down to the individual product level. The workflow that I used is as follows:

1) Look at the 2015 breakdown for what percentage of total sales were associated with a given item. At this location and in this department, there are about 800 products. I decided to look at 2015 only because not every product was sold every year.
2) Look at the daily distribution in a given month by analyzing how the product sold every day in the month and averaging that together (for example, average what percent of total sales for the product occured on Feb 1 from 2011-2015).
3) Multiply the forecast to the product percentage and then use the daily distribution to scatter the sales over the month.

(See the results for more information)

You'll see in the results section that without any rounding, I am predicting partial units to be sold. Of course this isn't possible, so I rounded the results. It turns out that the rounding can have a significant effect on the final units sold (for example, rounding can change 16.4 units to 12 units or 16.4 units to 24 units depending on how I round).

I discovered that even though the workflow worked as intended, there is still more room to conduct more tests to assess the accuracy of the model. For the sake of time, I was only able to test this total workflow on one sample item.


## Results

| Trickle Down Forecast for Total Walmart Sales Feb/2016 | Result |
| ----------- | ----------- |
| Forecast | 1,166,909 Units |
| Actual | 1,264,510 Units|
| Error | 7.8%|

| Trickle Down Forecast for CA3 Store Foods 3 Dept Feb/2016 | Result |
| ----------- | ----------- |
| Forecast | 92,050 Units |
| Actual  | 85,925 Units|
| Error | 7.1%|

| Direct Forecast On Store and Department Feb/2016 | Result |
| ----------- | ----------- |
| Forecast | 85,647 Units |
| Actual   | 85,925 Units|
| Error | .32%| 

The above results were specific for only one store and one department in that store. After deciding to pursue a direct trend & seasonality forecast on stores and departments directly, I decided to examine the errors accross all stores/department.

![Image](images/all_error_hist.png)

As you can see, the mean forecasting error is 10%, significantly higher than the .32% I presented above. The histogram above shows that the majority of stores and departments have fairly low errors, but the mean is being driven higher by a smaller group.

The series of histograms below show the forecasting errors across each state. The mean forecasting errors are roughly equal.

![Image](images/states_hist.png)

The following series of graphs show the mean forecasting error for each store, and then breaks down the forecasting error for each department in that store.

![Image](images/ca_stores_bar.png)

![Image](images/tx_store_bar.png)

![Image](images/WI_store_bar.png)

You can start to see that some departments are more difficult to forecast than others. For example, the Foods1, Foods2, and Hobbies2 appear to have higher errors across all stores.

The following series of graphs show the mean forecasting error for each department, and then breaks down the forecasting error for each store.

![Image](images/foods_bar.png)

![Image](images/hobbies_bar.png)

![Image](images/household_bar.png)

It appears some stores are more problematic than others, such as TX1 and CA2. These graphs will be useful to think about how to tailor forecasting approaches for each department and store. A one size fits all model may not be appropriate for this project.

My final goal is to be able to develop a prediction for sales for any given product in the future, however. I decided to do a deeper dive on the CA3 Store and Foods3 Department.

The image below shows how the daily distribution results vary depending on how the forecast is rounded (each row represents a day, and the bottom row represents the total). The forecast has to be rounded in some fashion because Walmart can only sell products in whole units.  The first column is the forecast, the second is rounding up at .5, the third column is rounding up at .4, and the fourth column contains the actual sales for this product in February 2016.

![Image](images/all_rounding.png)

## Conclusion
In conclusion, I accomplished my primary objective which was to build a model that can predict unit sales. I was able to build my model in a way where any individual product can be tested and reviewed. I learned that sales predictions for indivudal products can be difficult to predict, but the monthly level sales have more confidence.

To Do:
* Conduct more tests to establish the validity of the model
* Explore more complex forecasting methodology such as ARIMA
* Continue to adjust and review how the daily distribution compares to the final results
* Examine weekly patterns in order to improve the daily distribution
* Adjust daily distribution for holidays and events that may affect sales

This project was very exciting to work on and I look forward to continuing my analysis. The implications for this model have major strategic benefits to businesses.

#### References
* https://www.kaggle.com/c/m5-forecasting-accuracy
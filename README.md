# Credit Card Fraud Detection

### Project Description

For this project we will be gathering some credit card data, analyzing it for particular features, and use these features as drivers for fraud. We will plug these features into our machine learning models and make accurate predictions on what a fradulent purchase and what is a regular purchase.

### Project Goals

Discover drivers of quality
Use drivers to develop a machine learning model that accurately predicts wine quality
This information could be used on future datasets to help find high quality wine

### Initial Questions

- How much fraud is occuring in this dataset?
- Do Fraud and Online Ordering have a Relationship?
- Does The Use Of a Pin and Fraud Have a Relationship?
- Are Distances Related?
- Are Fraud and Ratio To Median Purchase Price correlated?

### The Plan

#### Acquire data
- Data aquired from Kaggle  
- Data frame containted 1,000,000 rows and 8 columns before cleaning  
- Each row represents a credit card transaction  
- Each column represents a feature associated with the transaction 
#### Prepare
- Data came pythonic and not much was needed in terms of readability
- I checked for nulls and none were present
- I checked for outliers and though there were some, I chose to keep as they were needed
- Split data into train, validate and test, stratifying on 'quality'
#### Explore data in search of drivers of quality and answer the following questions:
- Do Fraud and Online Ordering have a Relationship?
- Does The Use Of a Pin and Fraud Have a Relationship?
- Are Distances Related?
- Are Fraud and Ratio To Median Purchase Price correlated?

### Data Dictionary

| Name                 | Definition |
| -------------------- | ---------- |
| distance_from_home | The distance from home where the transaction happened |
| distance_from_last_transaction | The distance from last transaction |
| ratio_to_median_purchase_price | Ratio of purchased price transaction to median purchase price. |
| repeat_retailer      | Binary, specifies if the transaction happened from same retailer. |
| used_chip           | Binary, specifies if the transaction through chip (credit card). |
| used_pin_number  | Binary, specifies if the transaction happened by using PIN number. |
| online_order | Binary, specifies if the transaction is an online order. |
| fraud              | Binary, specifies if the transaction is fraudulent. |

### Steps to Reproduce

- Clone this repo.
- Acquire the data from https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud
- Put the data in the file containing the cloned repo.
- Run notebook.
### Takeaways and Conclusions

- Ratio to median purchase price had the most correlation to fraud
- Distance from home also had some correlatio to fraud, but no enough to move forward
- Used pin number proved to have significance with fraud
- Online ordering also proved to have a significant relationship with fraud
### Recommendations

- Adding a feature for mean purchases in a day can help with modeling
- Adding a feature for distance from work could also help as we spend most time at our homes and jobs
- Purchase amounts can be helpful as well
- If provided more time to work on this project I would try and use scaling with my modeling and I would also try and bin the distance and ratio to median purchase price to use for modeling.


```python

```

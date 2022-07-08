# Predicting-Rate

In this project, we will first scrape data from a dynamic website and then we will train it using different models like linear regression random forest, gradient boosting, and random forest regressor.

## 1. Web Scrapping

We are using this website: https://www.magicbricks.com/property-for-sale/residential-real-estate?bedroom=&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment&cityName=Pune

The information we are extracting are: project name, location, bhk, floor, total floor, carpet area, super area, property type, furnishing, possessed by, status and price.



### Main Problem

In a simple static website you can easily scrape data, because the data is already loaded in the website. But in case of dynamic websites we have to deal with the dynamically loaded content. To solve this problem, we are first scrolling the website and then scrapping the data.

### Dependencies

This code requires following packages:

Selenium - https://selenium-python.readthedocs.io/

BeautifulSoup - https://www.crummy.com/software/BeautifulSoup/bs4/doc/

Pandas - https://pandas.pydata.org/

time - https://docs.python.org/3/library/time.html


## 2. Training The models

First we have defined X and Y columns after inspection.
We took project name, location, total_floors, and Carpet Area in X and rate in Y.

Then, we have used label encoder to convert object type columns to integers.

Now, we can train different models using X and Y.

The models have accuracy of 70%

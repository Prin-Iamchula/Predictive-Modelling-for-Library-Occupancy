# Predictive-Modelling-for-Library-Occupancy
This is the dissertation of MSc Data Science at University of Sussex by Prin Iamchula. The purpose is to generate 
the predictive model for forecasting the future occupancy in the library building.


## Introduction
The Library, University of Sussex, would like to understand the patterns of occupancy in the Library building and 
predict the future volume of occupancy to enhance the building management, for instance, increasing more staff on 
service points when the use of Library is the peak, reducing energy consumption from electrical facilities 
when the demand is low, etc. Accordingly, the major goal of this research is to generate the predictive model for 
the future occupancy volume in the Library building. By the contribution of the machine learning techniques, 
time series forecasting algorithms and neuron network system, the research applied the KNN, SVR and Random forest to 
the multivariate model, and the ARIMA, SARIMA, ANN and LSTM to univariate model. 

## Data description
All implementations and analysis in the research experiment mainly utilized Timestamp data, which is transformed into 
hourly and daily time scales. Likewise, weather condition data and term dates period information was applied and fitted 
to both time scales of Timestamp data.

* Timestamp data
It contains the record of the transaction when the user enters and leaves the library in every single second together 
with the user information such as group of users, from 2018 to 2019. The dataset has 4,830,793 rows and eleven columns 
(features).

![timestamp](https://user-images.githubusercontent.com/66419715/99997856-5536fc80-2df0-11eb-9486-decce1061bf5.PNG)

* Weather data
By applying `Scapy` (web scraping program), 
weather data is scraped from: https://www.wunderground.com/history/daily/gb/gatwick-airport/EGKK/date/2018-1-1 , 
which presents and record both daily and hourly weather condition between Gatwick airport and Brighton area.
It has ten features consists of Datetime, Temperature, Dew Point, Humidity, Wind, Wind Speed, Wind Gust, Pressure, 
Precip. and Condition.
Houly weather condition
<br/><br/>![image](https://user-images.githubusercontent.com/66419715/100001067-22dbce00-2df5-11eb-867f-14e5c3f261bf.png)
Daily weather condition
<br/><br/>![image](https://user-images.githubusercontent.com/66419715/100001210-60405b80-2df5-11eb-847f-1eb9069652c1.png)

* Term dates information
Term dates information is the period of activities, events and assessment of the University of Sussex. Such information 
is referred to the term dates of 2018 and 2019, as shown below. 
<br/><br/>![term_date](https://user-images.githubusercontent.com/66419715/99999466-c7a8dc00-2df2-11eb-920a-43c4907f0945.PNG)

## Data preprocessing
**Timestamp data**
* Calculate total number of user entering (“I”) per hour, month and day, to create hourly, daily and monthly respectively.
* Also calculate total number of group and school which belong to user. 
<br/><br/>![image](https://user-images.githubusercontent.com/66419715/100002904-da71df80-2df7-11eb-940d-4fe6842a4381.png)
<br/><br/>![image](https://user-images.githubusercontent.com/66419715/100003019-068d6080-2df8-11eb-9847-ea367804dd1f.png)

**Term dates**
* Term dates were added belong to the date which is in the periods or events that the University has announced. 	
![image](https://user-images.githubusercontent.com/66419715/100003270-5f5cf900-2df8-11eb-8972-5b1f3159dc9c.png)

**Weather condition data**
* Group weather condition to dataset by date and time (for both hourly and daily data).
<br/><br/>![image](https://user-images.githubusercontent.com/66419715/100003986-6d5f4980-2df9-11eb-954a-8aced5be9e9f.png)
<br/><br/>![image](https://user-images.githubusercontent.com/66419715/100004019-78b27500-2df9-11eb-8a83-38540f26be96.png)

# cafe_upscaling_models

![alt text](images/logos/logo_full_small.png)

Project repositor for the café upscaling models team.

Our mission is to predict future curiosity for café menus.

Our customers (cafés and coffee and spice manufactuers) use this data to decide what to stock and which drinks become popular which time of the year and which drinks are rising in popularity and as such how they shoul modify their menus.

## The Team

Data Processing and Feature engineering: Maria Helmetsberger (<s2410929013@fhooe.at>)\
Model Selection, Tunning and Evaluation: Bettina Pölzleitner (<s2410929023@fhooe.at>)\
Visualisations and Artistic Assests: Andrey Mayev (<s2410929043@fhooe.at>)\
Team lead and Presenter: Jonas Gstöttenmayr (<s2410929009@fhooe.at>)

## Business Problem

Many cafè's each year fail because of a variety of reasons. A common one is a miss-match between what they serve and what customers want in the moment.

Altough the art of brewing coffee is quite old in recent times many new variety of drinks have emerged and cafè's have to have an ever more eccentric menu. Hower creating these eccentric drinks requires not just new spices but also training. As such it can quickly become a costly afair to prepare a new in-trend drink only for interest to be lost by the time a cafè is ready to serve it.

We are trying to solve this problem through predicting the curiosity of customers over current interests.

## The Data

For the detailed version of this summary chapter see: [The Data Documentation](/docs/data_ingestion_preprocessing.mkd)

Data is the basis and lifeblood of any (AI) prediction project and as such of paramount important.

Our data comes from a tool set up by the wikemedia foundation: [Page views](https://pageviews.wmcloud.org/), it is updated 24 h after a day showcasing how often certain wikipedia pages are viewed.

### Data Exploration

The data contains the daily wikipeida page views of the last 5 years for 10 different drinks.
It has no missing values, but it contains some clear outliers.

### Data Preparation

While quite pretty the data is not perfect, mostly because of what it represents, as such it requires certain preparatory steps - like the "removal" of outliers.

For further utilisation of the data it was also brought into long format format, rather than the orignal (one drink per column).

### Feature engineering

To extract the most information from this rather simple dataset our in-house Data engineer Ms. Helmetsberger modified the data with additional features.

They include the fourier analysis to better understand the underlying trend as well as the trend.

For external features we decided to inclulde holidays as it is only natural that certain holidays would increase/decrease the viewcount.

## Models

For the detailed version of this summary chapter see: [The Model Documentation](/docs/data_ingestion_preprocessing.mkd)

### Model training

The data was perfectly prepared to be used in combination with NIXTLA.

As it is proper we trained a veritable swarm of models, of all possible families and types to search for the one who rules them all (in the case of drinks here).

The training was done on a variety of feature sets to truly find the most accurate model.

### Model Selection

Of course we adhere to the rules of machine learning only training on a training set and testing on a validation set for our model selection.

Our Metrics are based on an untouched unbiased test set where only the final model was used to try and predict it.

Our final model was then trained on all available data, using the parameters of the previously selected model, to make a Prediction for the unknown month of January.

### Model evaluation

Our model was evaluated in a variety of ways.

- Visualy, the often best and most accurate way

- With metrics, well known metrics like r2 score, MAE....
- Against a baseline, the future is impossible to predict as such our Model need not be perfect but better than randomness to be of use.

## Final Predicion

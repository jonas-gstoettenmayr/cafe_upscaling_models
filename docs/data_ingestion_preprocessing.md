# Data Ingestion and Preprocessing
Done by: Maria Helmetsberger

### To long format

First step is to read the data in and to make it into a long format with polars unpivot. (The index is set to the Date, variable is the Drink and the value name is the Views.)
Afterwards the casting of the date to an actual date and not just to a string. After this point I saved the data as a parquet as my long dataframe. 

### Preprocessing the data

To preprocess the data you firstly need to figure out what you are working with, therefore I plotted the data to get a bit of an overview. (Plots are in the Notebook data_ingestion_preprocessing.ipynb)

The Bubble tea was a real outlier here, on the day where it got an imense view count there was a google bubble tea themed game: https://www.google.com/logos/2023/boba/rc3/boba.html?sdoodles=1
Some others also had some outliers that where quite high. 

There was also the Pumpkin Spice Latte which just gained Popularity in late 2025.

To make the outliers less impactfull to the data I decided to cap it, I made 3 functions one with the IQR one with Percentages and one with the ZScore, we ended up using the IQR for now, because the Pumpkin Spice latte did not have other outliers besides the new popularity in late 2025 we wanted to kind of keep this, so I made a function that extracts data based on its Drink Name and then later puts it back into the capped ones. It had some problems with the types so I just casted our old data to the type the capped ones have. Like this we were able to keep the new trend of the Pumpkin Spice Latte

Afterwards the plots also looked better in the distribution. 

As a next step I checked if there are any missing values in our data, there where none. So we where good to go. I sorted the Data again by Drink and Date and afterwards renamed the columns to make it work nicely with nixtla.

This I again saved as a file in our data. 

### Feature Engineering

Next up I decided on the forcasting horizon of 31 days (Jannuary), every day predictions and season length of weeks for now. 

Then I made a function to get our holiday features in a given range for now and for Jannuary (I also looked at when the last one was, when the next one is and if there is a holiday today. Then we count for everyday the number of holidays in a rolling window (for now 7 days))

Next up we made a split_train_test function that splits the data into a train, val and test set. 

The next step was to make feature dictionaries for our data. I used the nixtla utilsforecast for help. I made the features none(so just the baseline), holidays with my function, fourier, trend and one with all the features combined. 

Afterwards there is a function to save all the data nicely with the featue type and the split. 

Then we have our feature_engineering_pipeline, we firstly print out some basic information of our data, then we split it, afterwards we make our feature engineering and afterwards we save the data. Then I just run the function.

### Nixtla compatibility

In the other notebook I just look at the data to check if it is truly nixtla compatible, the only missing stuff is the y of the future data (which of course we do not have as of now so everything went good).
# Modelling - By Bettina Pölzleitner and Jonas Gstöttenmayr

## Model training

The data was prepared to be used in combination with NIXTLA. As such no more work was necessary.

To make the calculation load manageable I went through the models familiy by family, choosing models which could have good performance on a dataset as "random" as ours.

After defning all the models I trained on the training set.

## Model Selection

Adhereing to the rules of machine learning I compared all the models on a validation set to choose the best of the non-tuned models.

This allowed me to choose NHITS.

NHITS (Neural Hierarchical Interpolation for Time Series) is a deep learning model that breaks a time series into different temporal scales to capture both long-term trends and short-term fluctuations.
It downsamples input data at different rates across multiple stacks, forcing earlier parts of the model to focus on coarse, low-frequency patterns while later parts handle high-frequency details.
Hierarchical interpolation techniques are used to combine these multi-scale predictions into a smooth, accurate forecast.

As NHITS is a neural model I was quite thankfull for my GPU.

To get a proper predicion the default parameters are of course note enough.

### Parameter tunning

Nixtla offers Auto models for all its components but unlike with statsmodels the searchspace for neuralmodels is to large to accuratly search without instructions.
So I head to manually try some configurations until I got a sense of which models performed better and which worse.
Using this I constructed a search grid and tried over 50 different configurations with the **optuna** optimzer.

With this I was able to choose the best hyperparameters, in the limited time we have for this project.

With this new "best" model I bagan the evaluation.

## Model evaluation

Our model was evaluated in a variety of ways.

- Visualy, the often best and most accurate way to read more about the visualisation see [The Visualisation Documentation](/docs/visualisations_documentation.md)
- With metrics, well known metrics like MAE, MAPE, RMSE
- Against a baseline, the future is impossible to predict as such our Model need not be perfect but better than randomness to be of use.

The metrics for the test set:

|    | metric   |      NHITS |   SeasonalNaive |
|---:|:---------|-----------:|----------------:|
|  0 | mae      | 150.933    |      171.024    |
|  1 | mape     |   0.328873 |        0.406559 |
|  2 | rmse     | 170.175    |      194.209    |

The visualisations are in their respetive Notebooks

An here we can see that our model is roughly 25% better than the baseline SeasonalNaive.

This may not seem like much but it is actually quite the feat when looking at the different time series of the drinks, they are vastly varied and not very patterend without clear trends, as such simply repeating the last 7 days constantly leads to actually pretty good results - most of the time at least - and to beat that is an achievement.

What is also very important to mention here is that in our business case it is not too important to get the exact numbers correct, but rather the overal "trend" if the views will rise or fall overall.
Which the metrics cannot express very well but visually our model is very good at following the overall "trend" by increasing and decreasing with the model, the error comes from the "size" of the increases and decreases.

## Final Predicion

The Final prediction is done in the Scripts folder where we read the raw data and first messure the metrics of our predictor on a new test set, than we predict january and save it.

The final prediction is dependand on the drink once more, some look very dependable and quite believable, while others leave some space for improvement.

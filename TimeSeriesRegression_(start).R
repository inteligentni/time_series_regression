# (install and) load fpp2 package

# Read the data from the "data/yearly_eatout_expenditures.csv" file


# Transform this data frame into a ts object and print the results:


# Plot the time series:



# Load the data about total monthly expenditures for the same time period
# from the file "data/monthly_eatout_expenditures.RData"


# Transform the loaded data into a ts object and print time series data for the first two years:


# Plot the monthly time series:


# To take a closer look at the monthly pattern, we can take a slice of the time series
# (e.g. from the beginning of 2005 till the end of 2007)




#########################
# Time series components
#########################

# Trend 
# Another example of a time series with a rising trend:


# Season. 
# As an example, we can take a look at the data set with average monthly 
# temperatures in Nottingham, 1920–1939
# As the data is in degrees Fahrenheit, first, transform time series in Celsius
# (C = 5*(F-32)/9)


# Cycle. 
# An example of annual numbers of lynx trappings in Canada, 1821–1934 


# Season + Cycle 
# Example: the monthly sales (in millions USD) of new one-family houses in the USA (1973-1995)


# White noise - time series that show no trend nor seasonality
# It is a stationary ts (its properties do not depend on the time the data were observed)



################################
# Time Series Regression Models
################################

# Read the data from the "data/quarterly_eatout_expenditures.csv" file


# Create and plot a ts object:


# Check for the presence of trend and season in the time series
# (use decompose() f.)


#
# Splitting the time series data into training and test sets
#

# Create the test set using the last 8 observations (= 2 years) 


# The rest of time series is used for training


#
# Regression using trend and season as predictors
#

# Create a regression model using the tslm() function w/ trend and season as predictors


# Examine the elements of the tslm model:


# Plot the fitted against the actual time series:



# Create a scatter plot of actual (x-axis) and fitted values (y-axis)
# But, first load the 'util.R' script as it contains the auxiliary plotting function



#
# Making predictions (forecasts)
#

# Use the test set to make predictions


# Print predicted values


# Plot the predictions



# Create predictions using seasonal naive method, to serve as the baseline


# Re-plot the predictions, now with the baseline included:



# Compute evaluation measures using the accuracy() function


# Select a subset of measures (MAE, RMSE, MAPE, MASE) and use them to compare 
# the regression model against the baseline (on the test set)



#
# Checking assumptions of the linear model
#

# The checkresiduals() function conveniently gathers different methods for checking assumptions 
# (based on the residuals of the linear model):


# We can check the assumptions related to the mean and normal distribution, as follows:


###############################################
# Multivariate (time series) regression models
###############################################

# To build this model, we need a richer data set (uschange)


# Plot time series (separatelly for each variable)


#
# Splitting the data into training and test sets. 
#

# To decide on the split, examine the length of the time series, 
# the start and end time points



# Take the last 5 years for the test set



#
# Creating a model
#

# Create a tslm model with Income, Production, Savings, and Unemployment as predictors:



# Examine (visually) how well the fitted values match the data:



#
# Examining the assumptions for linear models
#


# Check the remaining assumption: that residuals are unrelated to the predictor variables. 
# To that end, plot each predictor variable against the residuals using the
# residuals_against_predictors_plot() function (from the util.R script)



#
# Evaluating the model on the test set 
#

# Make predictions on the test set:


# As baseline models, create a seasonal naive model and a drift model



# Compare the predictions of the linear model and the two baselines, first, by plotting them:  


# Use the accuracy() function to evaluate the models' performance using evaluation measures 
# (MAE, RMSE, MAPE, MASE):

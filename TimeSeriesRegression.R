# install.packages('fpp2')
library(fpp2)

# Read the data from the "data/yearly_eatout_expenditures.csv" file
yearly_exp <- read.csv("data/yearly_eatout_expenditures.csv")
str(yearly_exp)

# Transform this data frame into a ts object and print the results:
yearly_exp_ts <- ts(yearly_exp$tot_exp, start=2000)
yearly_exp_ts

# Plot the time series:
autoplot(yearly_exp_ts) + 
  xlab("Year") + 
  ylab("Expenditures in billion AUD") +
  ggtitle("Total yearly expenditure on eating out in Australia") +
  theme_light()

# Load the data about total monthly expenditures for the same time period
monthly_exp <- readRDS("data/monthly_eatout_expenditures.RData")
head(monthly_exp)

# Transform the loaded data into a ts object and print time series data for the first two years:
monthly_exp_ts <- ts(monthly_exp$expenditure, start = c(2000, 1), frequency = 12)
head(monthly_exp_ts, 2*12)

# Plot the monthly time series:
autoplot(monthly_exp_ts) + 
  xlab("Month-Year") + 
  ylab("Expenditures in bilion AUD") +
  ggtitle("Total monthly expenditure on eating out in Australia") +
  theme_light()

# Plot sub-series plot w/ monthly values and averages
ggsubseriesplot(monthly_exp_ts)

# To take a closer look at the monthly pattern, we can take a slice of the time series
autoplot(window(monthly_exp_ts, start=c(2005,1), end=c(2007,12))) +
  xlab("Month-Year") + ylab("Expenditures in bilion AUD")


#########################
# Time series components
#########################

# Trend. As another example of a time series with a rising trend:
autoplot(auscafe)

# Season. 
# As an example, we can take a look at the data set with average monthly 
# temperatures in Nottingham, 1920–1939
# As the data is in degrees Fahrenheit, first transform time series in Celsius
# (C = 5*(F-32)/9)
nottem_c <- (nottem-32)*5/9
autoplot(nottem_c)

# Cycle. An example of annual numbers of lynx trappings in Canada, 1821–1934 
autoplot(lynx) + xlab("Year") + ylab("Number of lynx trapped")

# Season + Cycle 
# Example: the monthly sales (in millions USD) of new one-family houses in the USA (1973-1995)
autoplot(hsales) +
  ylab("Monthly sales of houses (in millions USD)")

####################################
# Lagged values and autocorrelation
####################################

# Compute lagged values using the lag() function from the stats package:
monthly_exp_lag1 <- stats::lag(monthly_exp_ts, k=1)
head(monthly_exp_lag1)

# This can be better observed when plotted:
autoplot(monthly_exp_ts) +
  autolayer(stats::lag(monthly_exp_ts, k=1), series = "lag=1") +
  autolayer(stats::lag(monthly_exp_ts, k=12), series = "lag=12") +
  xlab("Month-Year") + ylab("Expenditures in bilion AUD") +
  guides(color=guide_legend(title = "Lagged values")) +
  theme_light() 

# Autocorrelation measures the extent of linear relationship between lagged values of a time series.
# A convenient way to examine autocorrelation is by plotting them:
ggAcf(monthly_exp_ts)

# White noise - time series that show no autocorrelation (no trend nor seasonality).
# It is a stationary ts (its properties do not depend on the time the data were observed).
set.seed(7)
rand_walk <- ts(rnorm(60))
autoplot(rand_walk)
ggAcf(rand_walk)

################################
# Time Series Regression Models
################################

# Read the data from the "data/quarterly_eatout_expenditures.csv" file
quarterly_exp <- read.csv("data/quarterly_eatout_expenditures.csv")
str(quarterly_exp)

# Create and plot a ts object:
quarterly_exp_ts <- ts(quarterly_exp$tot_exp, start = c(2000,1), frequency = 4)

autoplot(quarterly_exp_ts) +
  xlab("Year") + ylab("Total quarterly expenditures (in billion AUD)") + 
  ggtitle("Total quarterly expenditure on eating out in Australia") + theme_light()

# Check for the presence of trend and season in the time series
# (use stl() f.)
autoplot(stl(quarterly_exp_ts, s.window = "periodic"))


#
# Splitting the time series data into training and test sets
#

# Create the test set using the last 8 observations (= 2 years) 
exp_test <- tail(quarterly_exp_ts, 8)
exp_test

# The rest of time series is used for training
exp_train <- window(quarterly_exp_ts, end=c(2014, 4))
tail(exp_train, 4)

#
# Regression using trend and season as predictors
#

# Create a regression model using the tslm() function w/ trend and season as predictors
exp_tslm <- tslm(exp_train ~ trend + season)
summary(exp_tslm)

# Examine the elements of the tslm model:
names(exp_tslm)

# Plot the fitted against the actual time series:
autoplot(exp_train, series = "Data") +
  autolayer(fitted(exp_tslm), series = "Fitted") +
  guides(color = guide_legend(title = "")) +
  xlab("Year") + ylab("Total quarterly expenditures (in billion AUD)") + 
  ggtitle("True vs fitted values") +
  theme_light()

# Create a scatter plot of actual (x-axis) and fitted values (y-axis)
# But, first load the 'util.R' script as it contains the auxiliary plotting function
source('util.R')
true_vs_fitted_plot(true_vals = exp_train, fitted_vals = fitted(exp_tslm))

#
# Making predictions (forecasts)
#

# Use the test set to make predictions
pred_tslm <- forecast(exp_tslm, newdata = as.data.frame(exp_test))

# Print predicted values
pred_tslm$mean


# Plot the predictions
autoplot(quarterly_exp_ts) +
  autolayer(pred_tslm, series = "Predicted", PI = FALSE) +
  guides(color=guide_legend(title="")) +
  ylab("Quarterly expenditures in billion AUD") + 
  xlab("Year") +
  theme_light()

# Create predictions using seasonal naive method, to serve as the baseline
exp_snaive <- snaive(exp_train, h=length(exp_test))

# Re-plot the predictions, now with the baseline included:
autoplot(quarterly_exp_ts) +
  autolayer(pred_tslm, series = "tslm", PI = FALSE) +
  autolayer(exp_snaive, series = "seasonal naive", PI = FALSE) +
  guides(color=guide_legend(title="Predictions")) +
  ylab("Quarterly expenditures in billion AUD") + 
  xlab("Year") +
  theme_light()

# Compute evaluation measures using the accuracy() function
?accuracy
accuracy(pred_tslm, exp_test)

# Select a subset of measures (MAE, RMSE, MAPE, MASE) and use them to compare 
# the regression model against the baseline (on the test set)
eval_measures <- c('MAE', 'RMSE', 'MAPE', 'MASE')
tslm_eval <- accuracy(pred_tslm, exp_test)[2, eval_measures]
snaive_eval <- accuracy(exp_snaive, exp_test)[2, eval_measures]
rbind(tslm_eval, snaive_eval)

#
# Checking assumptions of the linear model
#

# The checkresiduals() function conveniently gathers different methods for checking assumptions 
# (based on the residuals of the linear model):
checkresiduals(exp_tslm)

# We can check the assumptions related to the mean and normal distribution, as follows:
mean(residuals(exp_tslm))
shapiro.test(residuals(exp_tslm))
qqnorm(residuals(exp_tslm))
qqline(residuals(exp_tslm))

###############################################
# Multivariate (time series) regression models
###############################################

# To build this model, we need a richer data set
?uschange
head(uschange)

# Plot time series (separatelly for each variable)
autoplot(uschange, facets = TRUE)

#
# Splitting the data into training and test sets. 
#

# To decide on the split, examine the length of the time series, 
# the start and end time points
dim(uschange)
head(uschange)
tail(uschange) 

# Take the last 5 years for the test set
uschange_test <- tail(uschange, 4*4+3)
head(uschange_test)
uschange_train <- window(uschange, end=c(2011, 4))

#
# Creating a model
#

# Create a tslm model with Income, Production, Savings, and Unemployment as predictors:
usch_tslm <- tslm(Consumption ~ Income + Production + Savings + Unemployment,
                  data = uschange_train)  
summary(usch_tslm)

# Examine (visually) how well the fitted values match the data:
autoplot(uschange_train[,'Consumption'], series = 'Data') +
  autolayer(fitted(usch_tslm), series = 'Fitted') +
  ylab("Percentage change in personal consumption") +
  xlab("Year") +
  guides(color = guide_legend(title = "")) +
  theme_light()

#
# Examining the assumptions for linear models
#
checkresiduals(usch_tslm)

# Check the remaining assumption (not captured by the previous function) 
# that residuals are unrelated to the predictor variables. 
# To that end plot each predictor variable against the residuals using the
# residuals_against_predictors_plot() function (from the util.R script)
uschange_train_df <- as.data.frame(uschange_train)
uschange_train_df$Residuals <- as.numeric(residuals(usch_tslm))
residuals_against_predictors_plot(df = uschange_train_df, 
                                  predictor_lbls = colnames(uschange_train_df)[2:5],
                                  residuals_lbl = 'Residuals')

#
# Evaluating the model on the test set 
#

# Make predictions on the test set:
usch_tslm_pred <- forecast(usch_tslm, newdata = data.frame(uschange_test))

# As baseline models, create a seasonal naive model and a drift model
pred_horizon <- nrow(uschange_test)
usch_snaive <- snaive(uschange_train[,'Consumption'], h=pred_horizon)
usch_drift <- rwf(uschange_train[,'Consumption'], h=pred_horizon, drift = TRUE)

# Compare the predictions of the linear model and the two baselines, first, by plotting them:  
autoplot(uschange[,'Consumption']) +
  autolayer(usch_tslm_pred, series = "tslm", PI = FALSE) +
  autolayer(usch_drift, series = 'drift', PI = FALSE) +
  autolayer(usch_snaive, series = 'seasonal naive', PI = FALSE) +
  ylab("Percentage change in personal consumption") +
  xlab("Year") +
  guides(color = guide_legend(title = "Predictions")) +
  theme_light()

# Use the accuracy() function to evaluate the models' performance using evaluation measures 
# (MAE, RMSE, MAPE, MASE):
tslm_eval <- accuracy(usch_tslm_pred, uschange_test[,"Consumption"])
tslm_eval <- tslm_eval[2, eval_measures]

snaive_eval <- accuracy(usch_snaive, uschange_test[,"Consumption"])
snaive_eval <- snaive_eval[2, eval_measures]

drift_eval <- accuracy(usch_drift, uschange_test[,"Consumption"])
drift_eval <- drift_eval[2, eval_measures]

rbind(tslm_eval, snaive_eval, drift_eval)

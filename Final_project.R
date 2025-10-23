setwd("E:/Okanagan college/OneDrive - Okanagan College/Desktop/443/project")

library("fpp2")
library("fma")
library("gridExtra")
library("ggplot2")
library("forecast")
library("ggfortify")
library("zoo")
library("lmtest")
library("tseries")

stock <- read.csv("stock.csv")
stock <- na.omit(stock)
head(stock)
str(stock)

#stock_ts <- is.numeric(stock)
#stock_ts <- zoo(stock_ts, order.by = time(stock_ts))
stock_ts <- ts(stock, start = c(1985,1), frequency = 12)


p1 <- autoplot(stock_ts)
p1
p2 <- ggsubseriesplot(stock_ts)
p2
p3 <- ggseasonplot(stock_ts)
p3


autoplot(stock_ts) + ggtitle('s&p 500') 

# box-cox transformation
lambda = BoxCox.lambda(stock_ts)
stock_box <- BoxCox(stock_ts, lambda)
autoplot(stock_box) + ggtitle('S&P 500 BOXCOX transformed')



stock_decomp <- decompose(stock_ts)
autoplot(stock_decomp) +
  labs(title = "Decomposition of Time Series", x = "Year", y = "Price") +
  theme_minimal()



#### Base Models ####

# Split into training and testing sets for base models
train_ts <- window(stock_box, end = c(2022,12))
test_ts <- window(stock_box, start = c(2023,1))


#Average Model
avg_model <- meanf(train_ts, h = length(test_ts))
avg_fc <- forecast(avg_model, h = length(test_ts))

#Naive Model
naive_model <- naive(train_ts, h = length(test_ts))
naive_fc <- forecast(naive_model, h = length(test_ts))

#Drift Model
drift_model <- rwf(train_ts, h = length(test_ts), drift = TRUE)
drift_fc <- forecast(drift_model, h = length(test_ts))


# Compare forecast accuracy
accuracy_table_base_models <- data.frame(
  Model = c("Mean", "Naive", "Drift"),
  RMSE = c(
    Metrics::rmse(test_ts, avg_fc$mean),
    Metrics::rmse(test_ts, naive_fc$mean),
    Metrics::rmse(test_ts, drift_fc$mean)
  ),
  MAE = c(
    Metrics::mae(test_ts, avg_fc$mean),
    Metrics::mae(test_ts, naive_fc$mean),
    Metrics::mae(test_ts, drift_fc$mean)
  ),
  MAPE = c(
    Metrics::mape(test_ts, avg_fc$mean),
    Metrics::mape(test_ts, naive_fc$mean),
    Metrics::mape(test_ts, drift_fc$mean)
  )
)

print(accuracy_table_base_models)

# here we can see our Drift model is performing the best


# Checking residuals to see if there's 
checkresiduals(drift_model)
# Ljung-Box Test (for autocorrelation)
Box.test(drift_model$residuals, type = "Ljung")
# Breusch-Godfrey test
bgtest(drift_model$residuals ~ 1, order = length(test_ts))


# Make sure to fully convert all ts objects to data frames
# Training set
df_train <- data.frame(
  date = as.numeric(time(train_ts)),
  value = as.numeric(train_ts),
  type = "Training"
)

# Test set
df_test <- data.frame(
  date = as.numeric(time(test_ts)),
  value = as.numeric(test_ts),
  type = "Actual"
)

# AVG forecast
df_mean <- data.frame(
  date = as.numeric(time(avg_fc$mean)),
  value = as.numeric(avg_fc$mean),
  type = "AVG"
)

# Naive forecast
df_naive <- data.frame(
  date = as.numeric(time(naive_fc$mean)),
  value = as.numeric(naive_fc$mean),
  type = "Naive"
)

# Drift forecast
df_drift <- data.frame(
  date = as.numeric(time(drift_fc$mean)),
  value = as.numeric(drift_fc$mean),
  type = "Drift"
)


# Combine safely
df_all <- rbind(df_train, df_test, df_mean, df_naive, df_drift)

# Plot
library(ggplot2)

ggplot(df_all, aes(x = date, y = value, color = type)) +
  geom_line(size = .5) +
  labs(
    title = "Model Validation on Test Set",
    x = "Year",
    y = "Value",
    color = "Series"
  ) +
  theme_minimal()




############################# ETS (Exponential Smoothing) ###############################

# Split into training and testing sets for base models
train_ts_ets <- window(stock_ts, end = c(2022,12))
test_ts_ets <- window(stock_ts, start = c(2023,1))


# ETS Model1(Auto)
ets_model_1 <- ets(train_ts_ets)
ets_fc_1 <- forecast(ets_model_1, h = length(test_ts_ets))

# ETS Model2(Manual)
ets_model_2 <- ets(train_ts_ets, model = 'MMM')
ets_fc_2 <- forecast(ets_model_2, h = length(test_ts_ets))

# ETS Model3(Manual)
ets_model_3 <- ets(train_ts_ets, model = 'MAM')
ets_fc_3 <- forecast(ets_model_3, h = length(test_ts_ets))

# Compare forecast accuracy
accuracy_table_ets_models <- data.frame(
  Model = c("Auto_ETS","ETS2","ETS3"),
  RMSE = c(
    Metrics::rmse(test_ts_ets, ets_fc_1$mean),
    Metrics::rmse(test_ts_ets, ets_fc_2$mean),
    Metrics::rmse(test_ts_ets, ets_fc_3$mean)
  ),
  MAE = c(
    Metrics::mae(test_ts_ets, ets_fc_1$mean),
    Metrics::mae(test_ts_ets, ets_fc_2$mean),
    Metrics::mae(test_ts_ets, ets_fc_3$mean)
  ),
  MAPE = c(
    Metrics::mape(test_ts_ets, ets_fc_1$mean),
    Metrics::mape(test_ts_ets, ets_fc_2$mean),
    Metrics::mape(test_ts_ets, ets_fc_3$mean)
  ),
  AIC = c(
    ets_fc_1$model$aic,
    ets_fc_2$model$aic,
    ets_fc_3$model$aic
  ),
  AICc = c(
    ets_fc_1$model$aicc,
    ets_fc_2$model$aicc,
    ets_fc_3$model$aicc
  )
)

print(accuracy_table_ets_models)

######################

# Split into training and testing sets for base models
train_ts_ets_box <- window(stock_box, end = c(2022,12))
test_ts_ets_box <- window(stock_box, start = c(2023,1))


# ETS Model Box-Cox(Auto)
ets_model_box_1 <- ets(train_ts_ets_box)
ets_fc_box_1 <- forecast(ets_model_box_1, h = length(test_ts_ets_box))

# ETS Model Box-Cox(Manual)
ets_model_box_2 <- ets(train_ts_ets_box, model = 'AAA')
ets_fc_box_2 <- forecast(ets_model_box_2, h = length(test_ts_ets_box))

# ETS Model Box-Cox(Manual)
ets_model_box_3 <- ets(train_ts_ets_box, model = 'ANA')
ets_fc_box_3 <- forecast(ets_model_box_3, h = length(test_ts_ets_box))


# Compare forecast accuracy
accuracy_table_ets_models_BC <- data.frame(
  Model = c("Auto_ETS_BC","ETS_BC2","ETS_BC3"),
  RMSE = c(
    Metrics::rmse(test_ts_ets_box, ets_fc_box_1$mean),
    Metrics::rmse(test_ts_ets_box, ets_fc_box_2$mean),
    Metrics::rmse(test_ts_ets_box, ets_fc_box_3$mean)
  ),
  MAE = c(
    Metrics::mae(test_ts_ets_box, ets_fc_box_1$mean),
    Metrics::mae(test_ts_ets_box, ets_fc_box_2$mean),
    Metrics::mae(test_ts_ets_box, ets_fc_box_3$mean)
  ),
  MAPE = c(
    Metrics::mape(test_ts_ets_box, ets_fc_box_1$mean),
    Metrics::mape(test_ts_ets_box, ets_fc_box_2$mean),
    Metrics::mape(test_ts_ets_box, ets_fc_box_3$mean)
  ),
  AIC = c(
    ets_fc_box_1$model$aic,
    ets_fc_box_2$model$aic,
    ets_fc_box_3$model$aic
  ),
  AICc = c(
    ets_fc_box_1$model$aicc,
    ets_fc_box_2$model$aicc,
    ets_fc_box_3$model$aicc
  )
)

print(accuracy_table_ets_models_BC)

ets_fc_box_1$mean <- InvBoxCox(ets_fc_box_1$mean, lambda)

# Shapiro-Wilk Test for normality
shapiro.test(residuals(ets_fc_1))
shapiro.test(residuals(ets_fc_box_1))

# Ljung-Box Test (for autocorrelation)
Box.test(ets_fc_1$residuals, type = "Ljung")
Box.test(ets_fc_box_1$residuals, type = "Ljung")


# Make sure to fully convert all ts objects to data frames
# Training set
df_train <- data.frame(
  date = as.numeric(time(train_ts_ets)),
  value = as.numeric(train_ts_ets),
  type = "Training"
)

# Test set
df_test <- data.frame(
  date = as.numeric(time(test_ts_ets)),
  value = as.numeric(test_ts_ets),
  type = "Test"
)

# Auto ETS forecast
df_AETS <- data.frame(
  date = as.numeric(time(ets_fc_1$mean)),
  value = as.numeric(ets_fc_1$mean),
  type = "Auto ETS"
)

# Auto ETS BC forecast
df_AETSBC <- data.frame(
  date = as.numeric(time(ets_fc_box_1$mean)),
  value = as.numeric(ets_fc_box_1$mean),
  type = "Auto ETS BC"
)

# Combine safely
df_all <- rbind(df_train, df_test, df_AETS, df_AETSBC)

# Plot
library(ggplot2)

ggplot(df_all, aes(x = date, y = value, color = type)) +
  geom_line(size = .5) +
  labs(
    title = "Model Validation on Test Set",
    x = "Year",
    y = "Value",
    color = "Series"
  ) +
  theme_minimal()

############################# SES,HOLT & HOLT-WINTERS ###############################

# SES Models
ses_model <- ses(train_ts, h = length(test_ts))
ses_fc <- forecast(ses_model, h = length(test_ts))

# Holt's Model
holt_model <- holt(train_ts, h = length(test_ts))
holt_fc <- forecast(holt_model, h = length(test_ts))

# Holt's Damped Model
holtD_model <- holt(train_ts, h = length(test_ts), damped = T)
holtD_fc <- forecast(holtD_model, h = length(test_ts))

# Holt-Winters Model
holtW_model <- hw(train_ts, h = length(test_ts), seasonal = "add", biasadj = F)
holtW_fc <- forecast(holtW_model, h = length(test_ts))

# Compare forecast accuracy
accuracy_table_ets_models_ses <- data.frame(
  Model = c("SES", "Holt", "Holt D", "Holt-Winter"),
  RMSE = c(
    Metrics::rmse(test_ts, ses_fc$mean),
    Metrics::rmse(test_ts, holt_fc$mean),
    Metrics::rmse(test_ts, holtD_fc$mean),
    Metrics::rmse(test_ts, holtW_fc$mean)
  ),
  MAE = c(
    Metrics::mae(test_ts, ses_fc$mean),
    Metrics::mae(test_ts, holt_fc$mean),
    Metrics::mae(test_ts, holtD_fc$mean),
    Metrics::mae(test_ts, holtW_fc$mean)
  ),
  MAPE = c(
    Metrics::mape(test_ts, ses_fc$mean),
    Metrics::mape(test_ts, holt_fc$mean),
    Metrics::mape(test_ts, holtD_fc$mean),
    Metrics::mape(test_ts, holtW_fc$mean)
  ),
  AIC = c(
    ses_fc$model$aic,
    holt_fc$model$aic,
    holtD_fc$model$aic,
    holtW_fc$model$aic
  ),
  AICc = c(
    ses_fc$model$aicc,
    holt_fc$model$aicc,
    holtD_fc$model$aicc,
    holtW_fc$model$aicc
  )
)

print(accuracy_table_ets_models_ses)

# converting values back to their original state
ses_fc$mean <- InvBoxCox(ses_fc$mean, lambda)
holt_fc$mean <- InvBoxCox(holt_fc$mean, lambda)
holtD_fc$mean <- InvBoxCox(holtD_fc$mean, lambda)
holtW_fc$mean <- InvBoxCox(holtW_fc$mean, lambda)

# Ljung-Box Test (for autocorrelation)
Box.test(holt_fc$residuals, type = "Ljung")

checkresiduals(holt_fc)

# Make sure to fully convert all ts objects to data frames
# Training set
df_train <- data.frame(
  date = as.numeric(time(train_ts_ets)),
  value = as.numeric(train_ts_ets),
  type = "Training"
)

# Test set
df_test <- data.frame(
  date = as.numeric(time(test_ts_ets)),
  value = as.numeric(test_ts_ets),
  type = "Test"
)

# SES forecast
df_ses <- data.frame(
  date = as.numeric(time(ses_fc$mean)),
  value = as.numeric(ses_fc$mean),
  type = "SES"
)

# Holt forecast
df_holt <- data.frame(
  date = as.numeric(time(holt_fc$mean)),
  value = as.numeric(holt_fc$mean),
  type = "Holt"
)

# Holt Damped forecast
df_holtD <- data.frame(
  date = as.numeric(time(holtD_fc$mean)),
  value = as.numeric(holtD_fc$mean),
  type = "Holt Damped"
)

# Holt-Winters forecast
df_holtW <- data.frame(
  date = as.numeric(time(holtW_fc$mean)),
  value = as.numeric(holtW_fc$mean),
  type = "Holt-Winters"
)
# Combine safely
df_all <- rbind(df_train, df_test, df_ses, df_holt, df_holtD, df_holtW)

# Plot
library(ggplot2)

ggplot(df_all, aes(x = date, y = value, color = type)) +
  geom_line(size = .5) +
  labs(
    title = "Model Validation on Test Set",
    x = "Year",
    y = "Value",
    color = "Series"
  ) +
  theme_minimal()

################################################################################

# Split into training and testing sets
train_ts_ar <- window(stock_ts, end = c(2022,12))
test_ts_ar <- window(stock_ts, start = c(2023,1)) 

# Check for stationarity using Augmented Dickey-Fuller test
adf.test(train_ts, alternative = "stationary")
# Check for stationarity using Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
kpss.test(train_ts, null = "Level")

# Perform differencing to make the series stationary
diff <- diff(stock_ts, differences = 2)
# Plot the differenced series
autoplot(diff) +
  labs(title = "Differenced Total Tires", x = "Year", y = "Differenced Total Tires") +
  theme_minimal()

# Check for stationarity using Augmented Dickey-Fuller test for total tires
adf_test_diff <- adf.test(diff, alternative = "stationary")
print(adf_test_diff)
# Check for stationarity using Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
kpss_test_diff <- kpss.test(diff, null = "Level")
print(kpss_test_diff)


Acf(diff)
Pacf(diff)



### Step 2: Examine ACF and PACF Plots
par(mfrow = c(2,1))
acf(diff, main = "ACF of Differenced Series")
pacf(diff, main = "PACF of Differenced Series")

# ARIMA Models 
arima_1 <- auto.arima(train_ts_ar, 
                      seasonal = FALSE,
                      trace = TRUE,
                      allowdrift = TRUE
)  
arima_fc_1 <- forecast(arima_1, h = length(test_ts_ar))

arima_2 <- Arima(train_ts_ar, order = c(1,2,1))
arima_fc_2 <- forecast(arima_2, h = length(test_ts_ar))

arima_3 <- Arima(train_ts_ar, order = c(1,2,2))
arima_fc_3 <- forecast(arima_3, h = length(test_ts_ar))

arima_4 <- Arima(train_ts_ar, order = c(2,2,1))
arima_fc_4 <- forecast(arima_4, h = length(test_ts_ar))

arima_5 <- Arima(train_ts_ar, order = c(2,2,2))
arima_fc_5 <- forecast(arima_5, h = length(test_ts_ar))

# Compare forecast accuracy
accuracy_table_arima_models <- data.frame(
  Model = c("Arima1", "Arima2", "Arima3", "Arima4", "Arima5"),
  RMSE = c(
    Metrics::rmse(test_ts_ar, arima_fc_1$mean),
    Metrics::rmse(test_ts_ar, arima_fc_2$mean),
    Metrics::rmse(test_ts_ar, arima_fc_3$mean),
    Metrics::rmse(test_ts_ar, arima_fc_4$mean),
    Metrics::rmse(test_ts_ar, arima_fc_5$mean)
  ),
  MAE = c(
    Metrics::mae(test_ts_ar, arima_fc_1$mean),
    Metrics::mae(test_ts_ar, arima_fc_2$mean),
    Metrics::mae(test_ts_ar, arima_fc_3$mean),
    Metrics::mae(test_ts_ar, arima_fc_4$mean),
    Metrics::mae(test_ts_ar, arima_fc_5$mean)
  ),
  MAPE = c(
    Metrics::mape(test_ts_ar, arima_fc_1$mean),
    Metrics::mape(test_ts_ar, arima_fc_2$mean),
    Metrics::mape(test_ts_ar, arima_fc_3$mean),
    Metrics::mape(test_ts_ar, arima_fc_4$mean),
    Metrics::mape(test_ts_ar, arima_fc_5$mean)
  ),
  AIC = c(
    arima_fc_1$model$aic,
    arima_fc_2$model$aic,
    arima_fc_3$model$aic,
    arima_fc_4$model$aic,
    arima_fc_5$model$aic
  ),
  AICc = c(
    arima_fc_1$model$aicc,
    arima_fc_2$model$aicc,
    arima_fc_3$model$aicc,
    arima_fc_4$model$aicc,
    arima_fc_5$model$aicc
  )
)

print(accuracy_table_arima_models)



# SARIMA Model (Auto SARIMA)
sarima_1 <- auto.arima(train_ts_ar, 
            max.d = 3,            
            max.D = 3,            
            max.p = 3, 
            max.q = 3,
            max.P = 3,
            max.Q = 3,
            seasonal = TRUE,
            stepwise = FALSE,
            approximation = FALSE,
            trace = TRUE,
            allowdrift = TRUE,
            allowmean = TRUE,
)
sarima_fc_1 <- forecast(sarima_1, h = length(test_ts_ar))

sarima_2 <- Arima(train_ts_ar, order = c(3,1,1), seasonal = list(order = c(1,1,2), period = 12))
sarima_fc_2 <- forecast(sarima_2, h = length(test_ts_ar))

sarima_3 <- Arima(train_ts_ar, order = c(3,2,1), seasonal = list(order = c(1,1,2), period = 12))
sarima_fc_3 <- forecast(sarima_3, h = length(test_ts_ar))

sarima_4 <- Arima(train_ts_ar, order = c(3,1,1), seasonal = list(order = c(1,0,2), period = 12))
sarima_fc_4 <- forecast(sarima_4, h = length(test_ts_ar))

sarima_5 <- Arima(train_ts_ar, order = c(3,1,1), seasonal = list(order = c(2,0,1), period = 12))
sarima_fc_5 <- forecast(sarima_5, h = length(test_ts_ar))

sarima_6 <- Arima(train_ts_ar, order = c(3,1,1), seasonal = list(order = c(2,1,2), period = 12))
sarima_fc_6 <- forecast(sarima_6, h = length(test_ts_ar))


# Compare forecast accuracy
accuracy_table_sarima_models <- data.frame(
  Model = c("Sarima1", "Sarima2", "Sarima3", "Sarima4", "Sarima5", "Sarima6"),
  RMSE = c(
    Metrics::rmse(test_ts_ar, sarima_fc_1$mean),
    Metrics::rmse(test_ts_ar, sarima_fc_2$mean),
    Metrics::rmse(test_ts_ar, sarima_fc_3$mean),
    Metrics::rmse(test_ts_ar, sarima_fc_4$mean),
    Metrics::rmse(test_ts_ar, sarima_fc_5$mean),
    Metrics::rmse(test_ts_ar, sarima_fc_6$mean)
  ),
  MAE = c(
    Metrics::mae(test_ts_ar, sarima_fc_1$mean),
    Metrics::mae(test_ts_ar, sarima_fc_2$mean),
    Metrics::mae(test_ts_ar, sarima_fc_3$mean),
    Metrics::mae(test_ts_ar, sarima_fc_4$mean),
    Metrics::mae(test_ts_ar, sarima_fc_5$mean),
    Metrics::mae(test_ts_ar, sarima_fc_6$mean)
  ),
  MAPE = c(
    Metrics::mape(test_ts_ar, sarima_fc_1$mean),
    Metrics::mape(test_ts_ar, sarima_fc_2$mean),
    Metrics::mape(test_ts_ar, sarima_fc_3$mean),
    Metrics::mape(test_ts_ar, sarima_fc_4$mean),
    Metrics::mape(test_ts_ar, sarima_fc_5$mean),
    Metrics::mape(test_ts_ar, sarima_fc_6$mean)
  ),
  AIC = c(
    sarima_fc_1$model$aic,
    sarima_fc_2$model$aic,
    sarima_fc_3$model$aic,
    sarima_fc_4$model$aic,
    sarima_fc_5$model$aic,
    sarima_fc_6$model$aic
  ),
  AICc = c(
    sarima_fc_1$model$aicc,
    sarima_fc_2$model$aicc,
    sarima_fc_3$model$aicc,
    sarima_fc_4$model$aicc,
    sarima_fc_5$model$aicc,
    sarima_fc_6$model$aicc
  )
)

print(accuracy_table_sarima_models)

# converting values back to their original state
#arima_fc_3$mean <- InvBoxCox(arima_fc_3$mean, lambda)
#sarima_fc_2$mean <- InvBoxCox(sarima_fc_2$mean, lambda)


# Training set
df_train <- data.frame(
  date = as.numeric(time(train_ts_ets)),
  value = as.numeric(train_ts_ets),
  type = "Training"
)

# Test set
df_test <- data.frame(
  date = as.numeric(time(test_ts_ets)),
  value = as.numeric(test_ts_ets),
  type = "Test"
)

# Auto ARIMA forecast
df_Aarima <- data.frame(
  date = as.numeric(time(arima_fc_1$mean)),
  value = as.numeric(arima_fc_1$mean),
  type = "Auto-Arima"
)

# ARIMA forecast
df_arima <- data.frame(
  date = as.numeric(time(arima_fc_5$mean)),
  value = as.numeric(arima_fc_5$mean),
  type = "Arima"
)

# Auto SARIMA forecast
df_Asarima <- data.frame(
  date = as.numeric(time(sarima_fc_1$mean)),
  value = as.numeric(sarima_fc_1$mean),
  type = "Auto-Sarima"
)

# SARIMA forecast
df_sarima <- data.frame(
  date = as.numeric(time(sarima_fc_3$mean)),
  value = as.numeric(sarima_fc_3$mean),
  type = "Sarima"
)

# Combine safely
df_all <- rbind(df_train, df_test, df_arima, df_Aarima, df_sarima, df_Asarima)

# Plot
library(ggplot2)

ggplot(df_all, aes(x = date, y = value, color = type)) +
  geom_line(size = .5) +
  labs(
    title = "Model Validation on Test Set",
    x = "Year",
    y = "Value",
    color = "Series"
  ) +
  theme_minimal()

##################################################################################


# Split into training and testing sets for final models
train_ts <- window(stock_ts, end = c(2022,12))
test_ts <- window(stock_ts, start = c(2023,1))

# Compare forecast accuracy of all top models
accuracy_table_final <- data.frame(
  Model = c("ETS", "ETS_BC", "Holt", "ARIMA", "SARIMA"),
  RMSE = c(
    Metrics::rmse(test_ts, ets_fc_1$mean),
    Metrics::rmse(test_ts, ets_fc_box_1$mean),
    Metrics::rmse(test_ts, holt_fc$mean),
    Metrics::rmse(test_ts, arima_fc_5$mean),
    Metrics::rmse(test_ts, sarima_fc_3$mean)
  ),
  MAPE = c(
    Metrics::mape(test_ts, ets_fc_1$mean),
    Metrics::mape(test_ts, ets_fc_box_1$mean),
    Metrics::mape(test_ts, holt_fc$mean),
    Metrics::mape(test_ts, arima_fc_5$mean),
    Metrics::mape(test_ts, sarima_fc_3$mean)
  )
)

print(accuracy_table_final)

# Forecast with Best Model
best_model_name <- accuracy_table_final$Model[which.min(accuracy_table_final$RMSE)]
cat("Best model based on RMSE is:", best_model_name, "\nModel = AAN")



# Training set
df_train <- data.frame(
  date = as.numeric(time(train_ts)),
  value = as.numeric(train_ts),
  type = "Training"
)

# Test set
df_test <- data.frame(
  date = as.numeric(time(test_ts)),
  value = as.numeric(test_ts),
  type = "Test"
)

# Auto ETS(Multiplictive) forecast
df_etsfinal <- data.frame(
  date = as.numeric(time(ets_fc_1$mean)),
  value = as.numeric(ets_fc_1$mean),
  type = "Auto-ETS"
)

# Auto ETS(Additive) forecast
df_ets_bcfinal <- data.frame(
  date = as.numeric(time(ets_fc_box_1$mean)),
  value = as.numeric(ets_fc_box_1$mean),
  type = "Auto-ETS-BC"
)

# Auto Holt forecast
df_holtfinal <- data.frame(
  date = as.numeric(time(holt_fc$mean)),
  value = as.numeric(holt_fc$mean),
  type = "Holt"
)

# ARIMA forecast
df_arimafinal <- data.frame(
  date = as.numeric(time(arima_fc_5$mean)),
  value = as.numeric(arima_fc_3$mean),
  type = "Arima"
)

# SARIMA forecast
df_sarimafinal <- data.frame(
  date = as.numeric(time(sarima_fc_3$mean)),
  value = as.numeric(sarima_fc_3$mean),
  type = "Sarima"
)

# Combine safely
df_all <- rbind(df_train, df_test, df_etsfinal, df_ets_bcfinal, df_holtfinal, df_arimafinal, df_sarimafinal)

# Plot
library(ggplot2)

ggplot(df_all, aes(x = date, y = value, color = type)) +
  geom_line(size = .5) +
  labs(
    title = "Model Validation on Test Set",
    x = "Year",
    y = "Value",
    color = "Series"
  ) +
  theme_minimal()




# Forecast next 6 months
ETS_Final <- ets(stock_ts, model = "AAN" )
final_forecast <- forecast(ETS_Final, h = 12)


# Convert forecast object to a data frame for ggplot
forecast_df <- data.frame(
  Date = as.Date(time(final_forecast$mean)),
  Point_Forecast = as.numeric(final_forecast$mean),
  Lo_80 = as.numeric(final_forecast$lower[,1]),
  Hi_80 = as.numeric(final_forecast$upper[,1]),
  Lo_95 = as.numeric(final_forecast$lower[,2]),
  Hi_95 = as.numeric(final_forecast$upper[,2])
)

# Historical data (training + test, if available)
historical_df <- data.frame(
  Date = as.Date(time(final_forecast$x)),
  Value = as.numeric(final_forecast$x)
)

# Create the plot
ggplot() +
  # Historical data
  geom_line(data = historical_df, aes(x = Date, y = Value), color = "black", linewidth = 0.8) +
  
  # Forecasted values
  geom_line(data = forecast_df, aes(x = Date, y = Point_Forecast), color = "blue", linewidth = 0.8) +
  
  # Confidence intervals (95%)
  geom_ribbon(data = forecast_df, aes(x = Date, ymin = Lo_95, ymax = Hi_95), 
              fill = "blue", alpha = 0.2) +
  
  # Confidence intervals (80%)
  geom_ribbon(data = forecast_df, aes(x = Date, ymin = Lo_80, ymax = Hi_80), 
              fill = "blue", alpha = 0.3) +
  
  # Labels and theme
  labs(
    title = "6-Month Forecast",
    x = "Year",
    y = "Index Value"
  ) +
  theme_minimal()
  #scale_x_date(date_labels = "%Y", date_breaks = "1 year")  # Format x-axis as years


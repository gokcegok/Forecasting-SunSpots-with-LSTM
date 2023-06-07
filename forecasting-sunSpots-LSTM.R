# ========================================================================================== #
# Libraries
# ========================================================================================== #

library(keras)
library(tensorflow)
library(dplyr)
library(tidyr)


# ========================================================================================== #
# Functions
# ========================================================================================== #

r2_score <- function(y_true, y_pred) {
  
  # Calculate R-squared
  # return: R2
  
  SS_residual <- sum((y_true - y_pred)^2)
  SS_total <- sum((y_true - mean(y_true))^2)
  r2 <- 1 - SS_residual / SS_total
  return(r2)
}


rmse <- function(y_true, y_pred) {
  
  # Calculate root mean squared error
  # return: RMSE
  
  return(sqrt(mean((y_true - y_pred)^2)))
}


mape <- function(y_true, y_pred) {
  
  # Calculate mean absolute percentage error
  # return: MAPE
  return(mean(abs((y_true - y_pred) / y_true)))
}


MinMaxScaler <- function(x) { 
  
  # Min-Max Scaling
  # return: scaled data
  
  return((x- min(x)) /(max(x)-min(x)))
}


inverseMinMax <- function(scaled, max, min) {
  
  # Inverse min-max scaling
  # scaled: min-max scaled data
  # max: max of original data
  # min: min of original data
  # return: original data
  
  return(scaled*(max - min) + min)
}


forecast <- function(model, data) {
  
  # Forecasting with future data
  # model: trained model
  # data: input matrix 
  # return: forecast result
  
  data <- array_reshape(data, c(1, dim(data)[1], 1))
  pred <- model %>% predict(data)
  return(pred)
}


# ========================================================================================== #
# Preparing Dataset
# ========================================================================================== #

dataset_path <- "Sunspots-Dataset.csv"
data <- read.csv(dataset_path)

data$Date <- as.Date(data$Date)
str(data)

time_series <- ts(data$Monthly.Mean.Total.Sunspot.Number, 
                  start=1749, frequency=12)

plot.ts(time_series)

components <- decompose(time_series)
plot(components)

scaled_sunspot <- MinMaxScaler(data$Monthly.Mean.Total.Sunspot.Number)
scaled <- matrix(scaled_sunspot, nrow = 3265, ncol = 1)

step <- 24

X <- matrix(0, nrow = length(scaled) - step, ncol = step)
y <- matrix(0, nrow = length(scaled) - step, ncol = 1)

for (i in 1:(length(scaled) - step)) {
  X[i, ] <- scaled[i:(i + step - 1)]
  y[i, ] <- scaled[i + step]
}

X <- array_reshape(X, c(dim(X)[1], dim(X)[2], 1))

# ========================================================================================== #
# Splitting into train and test sets
# ========================================================================================== #

k <- 2290
X_train <- X[1:k, , ]
y_train <- y[1:k, ]
X_test <- X[(k + 1):nrow(X), , ]
y_test <- y[(k + 1):length(y)]
X_train <- array_reshape(X_train, c(dim(X_train)[1], dim(X_train)[2], 1))
X_test <- array_reshape(X_test, c(dim(X_test)[1], dim(X_test)[2], 1))

# ========================================================================================== #
# LSTM model
# ========================================================================================== #

model <- keras_model_sequential() %>%
  layer_lstm(units = 64, input_shape = c(step, 1), return_sequences = TRUE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = 32, return_sequences = TRUE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = 64, return_sequences = TRUE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1)

model %>% compile(
  loss = "mse",
  optimizer = optimizer_adam(),
  metrics = c("mse", "mae", "mape")
)

summary(model)

model %>% fit(
  x = X_train,
  y = y_train,
  batch_size = 64,
  epochs = 100, verbose=1
)

# ========================================================================================== #
# Prediction
# ========================================================================================== #

y_pred <- model %>% predict(X_test)

min <- min(data$Monthly.Mean.Total.Sunspot.Number)
max <- max(data$Monthly.Mean.Total.Sunspot.Number)

y_pred <- inverseMinMax(c(y_pred), max, min)
y_test <- inverseMinMax(c(y_test), max, min)

cat("R2 Score:", r2_score(y_test, y_pred), "\n")
cat("RMSE:", rmse(y_test, y_pred), "\n")

y_test <- replace(y_test, y_test==0, 0.0000000001)

cat("MAPE:", mape(y_test, y_pred), "\n")

# ========================================================================================== #
# Forecasting for 3 years
# ========================================================================================== #

pure_tail <- tail(scaled, 24)

for (i in c(0:35)) {
  
  pred <- forecast(model, pure_tail)
  pure_tail <- rbind(as.matrix(pure_tail[c(2:length(pure_tail))]), pred[24])
  print(inverseMinMax(c(pred[24]), max, min))
}

# Run tests on different parameters

# Load the data

# Training data
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])

# Update training to set last part as validation
id_val = 1801:2000
Yval = Y[id_val]
Xval = X[id_val, ]
Ytrain = Y[-id_val]
Xtrain = X[-id_val, ]

# Testing data
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")
Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])

# Source the NN function
source("FunctionsNN.R")

# Recall the results of linear classifier from HW3
# Add intercept column
Xinter <- cbind(rep(1, nrow(Xtrain)), Xtrain)
Xtinter <- cbind(rep(1, nrow(Xt)), Xt)

# Default Parameters
out = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
                rate = 0.1, mbatch = 50, nEpoch = 150,
                hidden_p = 100, scale = 1e-3, seed = 12345)
plot(1:length(out$error), out$error, ylim = c(0, 100))
lines(1:length(out$error_val), out$error_val, col = "red")

# Evaluate error on testing data
test_error = evaluate_error(Xt, Yt, out$params$W1, out$params$b1, out$params$W2, out$params$b2)
test_error # 15.7

# Reduce Scale, all else the same
out = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
               rate = 0.1, mbatch = 50, nEpoch = 150,
               hidden_p = 100, scale = 1e-4, seed = 12345)
plot(1:length(out$error), out$error, ylim = c(0, 100))
lines(1:length(out$error_val), out$error_val, col = "red")

# Evaluate error on testing data
test_error = evaluate_error(Xt, Yt, out$params$W1, out$params$b1, out$params$W2, out$params$b2)
test_error # 17.2

# Increase Scale, all else the same
out = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
               rate = 0.1, mbatch = 50, nEpoch = 150,
               hidden_p = 100, scale = 1e-2, seed = 12345)
plot(1:length(out$error), out$error, ylim = c(0, 100))
lines(1:length(out$error_val), out$error_val, col = "red")

# Evaluate error on testing data
test_error = evaluate_error(Xt, Yt, out$params$W1, out$params$b1, out$params$W2, out$params$b2)
test_error # 15.8

# Increase hidden_p, leave all else the same
out = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
               rate = 0.1, mbatch = 50, nEpoch = 150,
               hidden_p = 200, scale = 1e-3, seed = 12345)
plot(1:length(out$error), out$error, ylim = c(0, 100))
lines(1:length(out$error_val), out$error_val, col = "red")

# Evaluate error on testing data
test_error = evaluate_error(Xt, Yt, out$params$W1, out$params$b1, out$params$W2, out$params$b2)
test_error # 15.1

# Zero Lambda, all else same
out = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.000,
               rate = 0.1, mbatch = 50, nEpoch = 150,
               hidden_p = 100, scale = 1e-3, seed = 12345)
plot(1:length(out$error), out$error, ylim = c(0, 100))
lines(1:length(out$error_val), out$error_val, col = "red")

# Evaluate error on testing data
test_error = evaluate_error(Xt, Yt, out$params$W1, out$params$b1, out$params$W2, out$params$b2)
test_error # 14.6

# Max Hidden P, no lambda, all else same
out = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.000,
               rate = 0.1, mbatch = 50, nEpoch = 150,
               hidden_p = 1000, scale = 1e-3, seed = 12345)
plot(1:length(out$error), out$error, ylim = c(0, 100))
lines(1:length(out$error_val), out$error_val, col = "red")

# Evaluate error on testing data
test_error = evaluate_error(Xt, Yt, out$params$W1, out$params$b1, out$params$W2, out$params$b2)
test_error # 12.9

# Zero lambda, double hidden p
out = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.000,
               rate = 0.1, mbatch = 50, nEpoch = 150,
               hidden_p = 200, scale = 1e-3, seed = 12345)
plot(1:length(out$error), out$error, ylim = c(0, 100))
lines(1:length(out$error_val), out$error_val, col = "red")

# Evaluate error on testing data
test_error = evaluate_error(Xt, Yt, out$params$W1, out$params$b1, out$params$W2, out$params$b2)
test_error # 13.7
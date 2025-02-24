# Install and load the saemix package
if(!require(saemix)){
  install.packages("saemix")
  library(saemix)
}


path <- "nonlin-1_var10_s12345.csv"
data <- read.csv(paste0("../Data/",path)) # Read the CSV file into R
head(data) # Check the structure of the data
train_data <- data[(data$Time<=17),c(1,2,3,4,5,7,9)]
test_data <- data[(data$Time>17),c(1,2,3,4,5,7,9)]

saemix_data <- saemixData(
  name.data = train_data,
  name.group = c("Group"),                # Random effect grouping variable
  name.response = "Response",              # Binary outcome variable
  name.predictors = c("Feature_1", "Feature_2", "Feature_3", "Feature_4", "Feature_5","Response"), # Predictor variables
  name.X=c("Feature_1")
)


# Model definition
binary.model <- function(psi, id, xidep) {
  Feature_1 <- xidep[, 1]  # Feature_1
  Feature_2 <- xidep[, 2]  # Feature_2
  Feature_3 <- xidep[, 3]  # Feature_3
  Feature_4 <- xidep[, 4]  # Feature_4
  Feature_5 <- xidep[, 5]  # Feature_5
  Response <- xidep[, 6]    # Response variable
  beta0 <- psi[id, 1]
  beta_1 <- psi[id, 2]  # Parameter
  beta_2 <- psi[id, 3]  # Parameter
  beta_3 <- psi[id, 4]  # Parameter
  
  # Logistic regression equation
  logit <- beta0 + beta_1 * log(1 + Feature_1) + beta_2 * (Feature_2^2)/exp(Feature_3+1) + beta_3 * (Feature_4^3)/ Feature_5
  
  # Probability of event
  pevent <- exp(logit) / (1 + exp(logit))

  # Observed likelihood
  pobs <- (Response == 0) * (1 - pevent) + (Response == 1) * pevent
  
  # Log probability
  logpdf <- log(pobs)

  
  return(logpdf)
}

simulBinary <- function(psi, id, xidep) {
  Feature_1 <- xidep[, 1]  # Feature_1
  Feature_2 <- xidep[, 2]  # Feature_2
  Feature_3 <- xidep[, 3]  # Feature_3
  Feature_4 <- xidep[, 4]  # Feature_4
  Feature_5 <- xidep[, 5]  # Feature_5
  Response <- xidep[, 6]    # Response variable
  beta0 <- psi[id, 1]
  beta_1 <- psi[id, 2]  # Parameter
  beta_2 <- psi[id, 3]  # Parameter
  beta_3 <- psi[id, 4]  # Parameter
  
  # Logistic regression equation
  logit <- beta0 + beta_1 * log(1 + Feature_1) + beta_2 * (Feature_2^2)/exp(Feature_3+1) + beta_3 * (Feature_4^3)/ Feature_5
  
  pevent<-exp(logit) / (1 + exp(logit)) 
  ysim<-rbinom(length(Feature_1),size=1, prob=pevent)
  return(ysim)
}

# Step 4: Define the Model Structure with covariate.model
saemix_model <- saemixModel(
  model = binary.model,
  simulate.function=simulBinary,
  modeltype = 'likelihood',
  name.response = 'Response',
  description = "Nonlinear binary outcome model",
  psi0 = matrix(c(0, 0, 0, 0), ncol = 4, byrow = TRUE, 
                dimnames = list(NULL, c("beta0", "beta1", "beta2", "beta3"))), 
  transform.par = c(0, 0, 0, 0),
  covariance.model = diag(c(1,0,0,0)), omega.init=diag(c(0.5,0,0,0))
)


# Step 5: Fit the Model
fit <- saemix(saemix_model, saemix_data, list(
  nbiter.saemix = c(500, 100),            # Number of iterations
  displayProgress = TRUE,                 # Show progress
  seed = 12345                            # For reproducibility
))

# Step 6: View Results and Predictions
summary(fit)                              # Summary of the fit
predicted <- predict(fit)                 # Predictions
plot(fit)                                 # Plot observed vs predicted
psi(fit)

pred <- saemixPredictNewdata(fit,test_data)



estimates_re <- psi(fit)  # Extract fitted parameter estimates

# Compute logit and probabilities manually for new data
logit <- estimates_re$beta0 + estimates_re$beta1 * log(1+test_data$Feature_1) +
  estimates_re$beta2 * (test_data$Feature_2^2)/exp(test_data$Feature_3+1) +
  estimates_re$beta3 * (test_data$Feature_4^3)/ test_data$Feature_5

# Convert logit to probabilities
predicted_prob <- exp(logit) / (1 + exp(logit))

# Check probabilities
# print(predicted_prob)
plot(data[(data$Time>17),]$True_p,predicted_prob, xlab = "True p", ylab = "Predicted p")

data_out <- data.frame(data[(data$Time>17),]$True_p, predicted_prob)

# Write to CSV file
write.csv(data_out, paste0("../Data/TrueAndPredictedTest/", "nonlin_SAEM", path), row.names = FALSE)  # Set row.names = TRUE if row names should be included


n_repeats <- 2000
n_probs <- length(predicted_prob)

# Initialize the matrix to store results
samples <- matrix(NA, nrow = n_repeats, ncol = n_probs)
MCR <- c()

# Loop to repeat the sampling 2000 times for all probabilities
for (i in 1:n_repeats) {
  set.seed(1106 + i)  # Ensure reproducibility with unique seeds
  samples[i, ] <- rbinom(n_probs, size = 1, prob = predicted_prob)
}

# Convert to a data frame for easier handling
samples_df <- as.data.frame(samples)


true_response <- data[(data$Time>17),]$Response

for (i in 1:n_repeats) {
  MCR[i] <- sum(abs(samples_df[i,] - true_response))/length(true_response)
  
  if(i%%50==0){
    print(i)
  }
}
mean(MCR)
var(MCR)

          
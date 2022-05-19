library("mgcv")

helper_function <- function(is_azure, d){
  if (is_azure == 'yes') {
    data <- read.csv("./gam_diagnostics.csv")  
  } else {
    data <- read.csv("./gam_diagnostics.csv")
  }
  
  if (d == 1) {
    gam_diagnostics <- gam(formula = w ~ s(theta, k = -1, fx = FALSE, bs = "ts"), 
                         method="REML",
                         family = binomial(link="logit"), 
                         data = data)
    predictions <- predict(gam_diagnostics, newdata = data.frame("theta" = data$theta), se.fit = TRUE, 
                           type = "response")
  } else { 
    gam_diagnostics <- gam(formula = w ~ s(theta0, theta1, k = -1, fx = FALSE, bs = "ts"),
                           method="REML",
                           family = binomial(link="logit"), 
                           data = data)
    predictions <- predict(gam_diagnostics, newdata = data.frame("theta0" = data$theta0, "theta1" = data$theta1), se.fit = TRUE, 
                           type = "response")
  }
  
  
  
  return(list("predictions"=predictions$fit, "se"=predictions$se.fit))
}

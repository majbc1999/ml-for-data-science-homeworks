
toy_data <- function(n, seed = NULL) {
       set.seed(seed)
       x <- matrix(rnorm(8 * n), ncol = 8)
       z <- 0.4 * x[,1] - 0.5 * x[,2] + 1.75 * x[,3] - 0.2 * x[,4] + x[,5]
       y <- runif(n) > 1 / (1 + exp(-z))
       return (data.frame(x = x, y = y))
}

log_loss <- function(y, p) {
       -(y * log(p) + (1 - y) * log(1 - p))
}

df_dgp <- toy_data(100000, 0)

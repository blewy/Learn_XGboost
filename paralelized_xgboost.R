# xgboost task parameters
nrounds <- 1000
folds <- 10
obj <- 'binary:logistic'
eval <- 'logloss'

# Parameter grid to search
params <- list(
  eval_metric = eval,
  objective = obj,
  eta = c(0.1,0.01),
  max_depth = c(4,6,8,10),
  max_delta_step = c(0,1),
  subsample = 1,
  scale_pos_weight = 1
)

# Table to track performance from each worker node
res <- data.frame()

# Simple cross validated xgboost training function (returning minimum error for grid search)
xgbCV <- function (params) {
  fit <- xgb.cv(
    data = data.matrix(train), 
    label = trainLabel, 
    param =params, 
    missing = NA, 
    nfold = folds, 
    prediction = FALSE,
    early.stop.round = 50,
    maximize = FALSE,
    nrounds = nrounds
  )
  rounds <- nrow(fit)
  metric = paste('test.',eval,'.mean',sep='')
  idx <- which.min(fit[,fit[[metric]]]) 
  val <- fit[idx,][[metric]]
  res <<- rbind(res,c(idx,val,rounds))
  colnames(res) <<- c('idx','val','rounds')
  return(val)
}

# Find minimal testing error in parallel
cl <- makeCluster(round(detectCores()/2)) 
clusterExport(cl, c("xgb.cv",'train','trainLabel','nrounds','res','eval','folds'))
sol <- gridSearch(
  fun = xgbCV,
  levels = params,
  method = 'snow',
  cl = cl,
  keepNames = TRUE,
  asList = TRUE
)

# Combine all model results
comb=clusterEvalQ(cl,res)
results <- ldply(comb,data.frame)
stopCluster(cl)

# Train model given solution above
params <- c(sol$minlevels,objective = obj, eval_metric = eval)
xgbModel <- xgboost(
  data = xgb.DMatrix(data.matrix(train),missing=NaN, label = trainLabel),
  param = params,
  nrounds = results[which.min(results[,2]),1]
)

print(params)
print(results)


getwd()



min_child_weight = seq(1,10,4)

searchGrid  <- expand.grid(eval_metric = "auc",
                           objective = "binary:logistic", 
                           boster = "gbtree",
                           max.depth = c(5,10,15), #Maximum depth of a tree
                           eta = c(0.01,0.5,1),#2/ntrees,  Step size shrinkage used in update to prevents overfitting 
                           gamma=0,  # Minimum loss reduction required to make a split
                           lambda=0, # L2 regularization term on weights
                           alpha=0 , # L1 regularization term on weights
                           subsample = c(0.5,0.8), #Subsample ratio of the training instance
                           colsample_bytree = 0.6, #Subsample ratio of columns when constructing each tree
                           print.every_n = 25,
                           early_stopping_rounds=10,
                           showsd = TRUE, 
                           stratified=TRUE,
                           maximize=TRUE
)

best_param = list()
best_seednumber = 1234
best_auc = 0
best_auc_iteration = 0
cv.nfold<-5
cv.nround<-100
best_index<-0

for(i in 1:nrow(searchGrid)){
  param <- list(eval_metric= searchGrid[i,"eval_metric"],
                objective = searchGrid[i,"objective"],
                boster = searchGrid[i,"boster"],
                max.depth =searchGrid[i,"max.depth"],  
                eta =searchGrid[i,"eta"], 
                gamma=searchGrid[i,"gamma"],  
                lambda=searchGrid[i,"lambda"],  
                alpha=searchGrid[i,"alpha"],  
                subsample = searchGrid[i,"subsample"],  
                colsample_bytree = searchGrid[i,"colsample_bytree"], 
                print.every_n = searchGrid[i,"print.every_n"],
                early_stopping_rounds=searchGrid[i,"early_stopping_rounds"],
                showsd = searchGrid[i,"showsd"],
                stratified= searchGrid[i,"stratified"],
                maximize= searchGrid[i,"maximize"]
                )
  
  cat("\n"," ---------- Iteration :", i,"---- \n")
  
seed.number = sample.int(10000, 1)[[1]]
set.seed(seed.number)
  
mdcv <- xgb.cv(data = as.matrix(df_train %>% select(-SeriousDlqin2yrs)),
               label = (as.numeric(df_train$SeriousDlqin2yrs)-1),
               params = param, 
               nthread=6, 
               nfold=cv.nfold, 
               nrounds=cv.nround,
               verbose = T, 
               maximize=TRUE) 

current_auc= max(mdcv$evaluation_log[,test_auc_mean])
max_auc_index = which.max(mdcv$evaluation_log[,test_auc_mean])

  if (current_auc > best_auc) 
      {
        best_auc = current_auc
        best_seednumber = seed.number
        best_param = param
        best_auc_iteration=max_auc_index
        best_index<-i
    }
}


best_auc
best_seednumber
best_param
best_auc_iteration
best_index

searchGrid[best_index,]


train.data <- xgb.DMatrix(data=as.matrix(df_train %>% select(-SeriousDlqin2yrs)), 
                          label = (as.numeric(df_train$SeriousDlqin2yrs)-1) )

set.seed(best_seednumber)
bst.model <- xgb.train(params = best_param,
                       data = train.data,
                       nrounds = best_auc_iteration, 
                       prediction = T)


ypred = predict(bst.model, as.matrix(df_train %>% select(-SeriousDlqin2yrs)))

xgb.importance(model = bst.model)

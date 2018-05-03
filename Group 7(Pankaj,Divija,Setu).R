#cleaning environment
rm(list = ls(all = T))
dev.off()

#Setting current directory
setwd("E:\\Data Science\\INSOFE\\CUTE\\CUTE 3\\Data")


#library used
library(DMwR)
library(ROCR)
library(MASS)
library(car)
library(caret)
library(ggplot2)
library(rpart)
library(e1071)
library(vegan)
library(class)
library(FNN)
library(Metrics)
library(rpart)
library(C50)
library(randomForest)


#Data pre-processing and data understanding
data = read.csv("bankdata.csv",header = T)
str(data)
summary(data)
names(data)
str(data$target)  

#changing target to 0 and 1
data$target = ifelse(data$target=="Yes",1,0)
data$target = as.factor(as.character(data$target))

#dealing with outliers
#defining a function to identify and deal with outliers
outliersMAD <- function(data, MADCutOff = 2.5, replace = NA, values = FALSE, bConstant = 1.4826, digits = 2) {
  absMADAway <- abs((data - median(data, na.rm = T))/mad(data, constant = bConstant, na.rm = T))
  
  data[absMADAway > MADCutOff] <- replace
  
  if (values == TRUE) { 
    return(round(absMADAway, digits)) 
  } else {
    return(round(data, digits)) 
  }
}
outdata = data[,-65]
data1 = data.frame(apply(outdata,2,outliersMAD))
data1 = cbind(data1,data$target)
names(data1)[65] <- "target"
str(data1)

#missing values imputation using central imputation
data1 = centralImputation(data1)
sum(is.na(data1))

#Standardizing the data with scale method
data1<-scale(data1[,-65],center = TRUE,scale = TRUE)
data1<-data.frame(data1)
data1<-cbind(data1,data$target)
names(data1)[65] <- "target"


#split into train and test
rows=seq(1,nrow(data1),1)
set.seed(123)
trainingrows=sample(rows,(70*nrow(data1))/100)
train = data1[trainingrows,]
test=data1[-trainingrows,]

#To balance the class, using smote function(but will not use to build model, because after class balancing from smote, not getting expected results)
trainWC <- SMOTE(target~., data=train,perc.over = 100, perc.under=200)
#trainWC$target <- as.numeric(trainWC$target)
prop.table(table(trainWC$target))

testWC <- SMOTE(target~., data=test,perc.over = 100, perc.under=200)
#testWC$target <- as.numeric(testWC$target)
prop.table(table(testWC$target))

#R-part Decision Tree
dtCart=rpart(target~.,data=train,method="class")
printcp(dtCart)
plotcp(dtCart)
summary(dtCart)
plot(dtCart,main="Classification Tree to check bankruptcy",margin=0.25,uniform=TRUE)
text(dtCart, use.n=TRUE)
write.csv(train,"sample.csv",row.names=F)

#pruning
dtCart<- prune(dtCart,cp=dtCart$cptable[which.min(dtCart$cptable[,"xerror"]),"CP"])
printcp(dtCart)
plotcp(dtCart)
#plot the prune tree
plot(dtCart, uniform=TRUE, 
     main="Pruned Classification Tree for bankruptcy",margin=0.10)
text(dtCart, use.n=TRUE)

# a=table(train$target,predict(dtCart,newdata = train,type="class"))
# (a[1,1])/(a[1,1]+a[1,2])*100#sensitivity
# accutrain3<-sum(diag(resulttrain3))/(sum(resulttrain3))*100
# sum(diag(a))/sum(a)*100#accuracy
# 
# b=table(test$target,predict(dtCart,newdata = test,type="class"))
# (b[1,1])/(b[1,1]+b[1,2])*100#sensitivity
# sum(diag(b))/sum(b)*100#accuracy

#confusion matrix
predtrainrpart<-predict(dtCart,newdata = train,type="class")
predtestrpart<-predict(dtCart,newdata=test,type="class")
confusionMatrix(predtestrpart,test$target,positive = "1")

#C50 Decision Tree
dtc50=C5.0(target~.,data=train,rules=TRUE)
dtc50$
summary(dtc50)

# atrainc50=table(train$target, predict(dtc50, newdata=train, type="class"))
# Trainc50sensit=(atrainc50[1,1])/(atrainc50[1,1]+atrainc50[1,2])*100#sensitivity
# Trainc50accu=sum(diag(atrainc50))/sum(atrainc50)*100#accuracy
# 
# btestc50=table(test$target, predict(dtc50, newdata=test, type="class"))
# Testc50sensit=(btestc50[1,1])/(btestc50[1,1]+btestc50[1,2])*100#sensitivity
# Trainc50accu=sum(diag(btestc50))/sum(btestc50)*100#accuracy


#Confusion Matrix
predtrainc50<-predict(dtc50,newdata = train,type="class")
predtestc50<-predict(dtc50,newdata=test,type="class")
confusionMatrix(predtestc50,test$target,positive = "1")

#-------------------------------------------------------------------------

#Random Forest Without Smote
RF1<-randomForest(target~.,data=train,keep.forest = TRUE,ntree=300)
RF1$predicted
RF1$importance
round(importance(RF1), 2)

# Extract and store important variables obtained from the random forest model
ImpRF1<-data.frame(RF1$importance)
ImpRF1<-data.frame(row.names(ImpRF1),ImpRF1[,1])
colnames(ImpRF1)=c('Attributes','Importance')
ImpRF1<-ImpRF1[order(ImpRF1$Importance,decreasing = TRUE),]
ImpRF1<-ImpRF1[1:63,]

# plot (directly prints the important attributes) 
varImpPlot(RF1)

# Predict on Train data 
predModtrain<-predict(RF1,train[,-c(65)],type="response",norm.votes = TRUE)
resulttrain<-table("actualvalues"=train$target,predModtrain)

#prediction test data
predModtest<-predict(RF1,test[,-c(65)],type="response",norm.votes = TRUE)
resulttest<-table("actualvalues"=test$target,predModtest)

#Accuracy
accutrain<-sum(diag(resulttrain))/sum(resulttrain)*100
accutest<-sum(diag(resulttest))/sum(resulttest)*100

#tuning the model, seems model is overfitted in the last attempt
RF2<-randomForest(target~.,data=train,keep.forest=TRUE,ntree=600,mtry=15,nodesize=20)

# Extract and store important variables obtained from the random forest model
ImpRF2<-data.frame(RF2$importance)
ImpRF2<-data.frame(row.names(ImpRF2),ImpRF2[,1])
colnames(ImpRF2)=c('Attributes','Importance')
ImpRF2<-ImpRF2[order(ImpRF2$Importance,decreasing = TRUE),]


# plot (directly prints the important attributes) 
varImpPlot(RF2)

#prediction on train data
predModtrain2<-predict(RF2,train[,-c(65)],type="response",norm.votes = TRUE)
resulttrain2<-table("actualvalues2"=train$target,predModtrain2)
resulttrain2
#prediction on test data
predModtest2<-predict(RF2,test[,-c(65)],type="response",norm.votes = TRUE)
resulttest2<-table("actualvalues2"=test$target,predModtest2)
resulttest2

#Accuracy
accutrain2<-sum(diag(resulttrain2))/(sum(resulttrain2))*100
accutest2<-sum(diag(resulttest2))/(sum(resulttest2))*100
accutrain2
accutest2

#Third attempt to build the random forest with diffrent arguements
RF3<-randomForest(target~.,data=train,keep.forest=TRUE,
                  ntree=900,mtry=30,nodesize=60)

# Extract and store important variables obtained from the random forest model
ImpRF3<-data.frame(RF3$importance)
ImpRF3<-data.frame(row.names(ImpRF3),ImpRF3[,1])
colnames(ImpRF3)=c('Attributes','Importance')
ImpRF3<-ImpRF3[order(ImpRF3$Importance,decreasing = TRUE),]
ImpRF3<-ImpRF3[1:24,]
# plot (directly prints the important attributes) 
varImpPlot(RF3)

#prediction on train data
predModtrain3<-predict(RF3,train[,-c(65)],type="response",norm.votes = TRUE)
resulttrain3<-table("actualvalues3"=train$target,predModtrain3)
resulttrain3
#prediction on test data
predModtest3<-predict(RF3,test[,-c(65)],type="response",norm.votes = TRUE)
resulttest3<-table("actualvalues3"=test$target,predModtest3)
resulttest3

#sentivity
senttrain3<-(resulttrain3[1,1])/(resulttrain3[1,1]+resulttrain3[1,2])*100
senttest3<-(resulttest3[1,1])/(resulttest3[1,1]+resulttest3[1,2])*100
senttrain3
senttest3

#Accuracy
accutrain3<-sum(diag(resulttrain3))/(sum(resulttrain3))*100
accutest3<-sum(diag(resulttest3))/(sum(resulttest3))*100
accutrain3
accutest3

#Confusion matrix for random forest
predtraincrf<-predict(RF3,newdata = train,type="response")
predtestcrf<-predict(RF3,newdata=test,type="class")
confusionMatrix(predtestcrf,test$target,positive = "1")


##########Logistic Regression##########
#logistic with PCA

#split into train and test
rows=seq(1,nrow(data1),1)
set.seed(123)
trainingrows=sample(rows,(70*nrow(data1))/100)
train = data1[trainingrows,]
test=data1[-trainingrows,]
x_train = train[,-ncol(train)]
x_test = test[,-ncol(test)]
y_train = train[,ncol(train)]
y_test = test[,ncol(test)]

pca = princomp(x_train) 
pca$loadings
summary(pca)
screeplot(pca,type = "lines")

pca_train = predict(pca, x_train)
pca_test = predict(pca, x_test)
pcacomptrain = pca_train[,1:3]
pcacomptest = pca_test[,1:3]
train1 = data.frame(pcacomptrain,train$target)
test1 = data.frame(pcacomptest,test$target)
model = glm(train.target~.,data = train1,family="binomial")
summary(model)

prob_train1 <- predict(model,data=train1,type="response") 
prob_test1 <- predict(model,test1,type="response")
pred_class1 <- ifelse(prob_train1>0.35,1,0)
table(train1$train.target,pred_class1)
pred1 <- prediction(prob_train1,train1$train.target)
perf1 <- performance(pred1,measure="tpr",x.measure = "fpr")

#Plot ROC curve
plot(perf1,col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.025))
perf_auc1 <- performance(pred1,measure = "auc")
auc1 <- perf_auc1@y.values[[1]]
print(auc1) #AUC is 65%
prob_test1 <- predict(model,test1,type="response")
preds_test1 <- ifelse(prob_test1>0.08,1,0)

confusionMatrix(test1$test.target,preds_test1,positive = "1")


#logistic regression without pca
model1 = glm(target~.,data=train,family="binomial")
summary(model1)
prob_train <- predict(model1,data=train,type="response") 
prob_test <- predict(model1,test,type="response")
pred_class <- ifelse(prob_train>0.35,1,0)
table(train$target,pred_class)
pred <- prediction(prob_train,train$target)
perf <- performance(pred,measure="tpr",x.measure = "fpr")

#Plot ROC curve
plot(perf,col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05)) ##Threshold of 0.35 seems good
perf_auc <- performance(pred,measure = "auc")
auc <- perf_auc@y.values[[1]]
print(auc) #AUC is 77%
prob_test <- predict(model1,test,type="response")
preds_test <- ifelse(prob_test>0.1,1,0)

confusionMatrix(preds_test, test$target,positive = "1")
#logistic regression with class balancing
trainWC = SMOTE(target~.,data=train,perc.over = 100,perc.under = 200)
prop.table(table(trainWC$target))
testWC = SMOTE(target~.,data=test,perc.over = 100,perc.under = 200)
prop.table(table(testWC$target))
modelWC = glm(target~.,data=trainWC,family="binomial")
summary(modelWC)
prob_trainWC <- predict(modelWC,data=trainWC,type="response") 
prob_testWC <- predict(modelWC,testWC,type="response")
pred_classWC <- ifelse(prob_trainWC>0.35,1,0)
table(trainWC$target,pred_classWC)
predWC <- prediction(prob_trainWC,trainWC$target)
perfWC <- performance(predWC,measure="tpr",x.measure = "fpr")

#Plot ROC curve
plot(perfWC,col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05)) ##Threshold of 0.35 seems good
perf_aucWC <- performance(predWC,measure = "auc")
aucWC <- perf_aucWC@y.values[[1]]
print(aucWC) #AUC is 78%
prob_testWC <- predict(modelWC,testWC,type="response")
preds_testWC <- ifelse(prob_testWC>0.5,1,0)

confusionMatrix(preds_testWC, testWC$target,positive = "1")

#######KNN#######
#standardize data
data2 <- decostand(data1[,-64],"range")
data2 <- cbind(data2,data1$target)
names(data2)[64] <- "target"
#Split into train and test
rows=seq(1,nrow(data2),1)
set.seed(456)
trainingrows1=sample(rows,(70*nrow(data2))/100)
train2 = data2[trainingrows1,]
test2=data2[-trainingrows,]
x_train2 = train2[,-ncol(train2)]
x_test2 = test2[,-ncol(test2)]
y_train2 = train2[,ncol(train2)]
y_test2 = test2[,ncol(test2)]

#Building KNN Classification model
#predicting with k = 1
pred=knn(x_train2,x_test2,train2$target,k=1) 
a=table(pred,test2$target)
a
accu= sum(diag(a))/nrow(x_test2)
accu #97% accuracy with k = 1
#check which value of k has best accuracy
accuracy = as.numeric()
for (i in 1:10){pred=knn(x_train2,x_test2,train2$target,k=i) 
a=table(pred,test2$target)
a
accu= sum(diag(a))/nrow(x_test2)
accuracy = rbind(accuracy,accu)
}
plot(accuracy,type = "b") #seems like accuracy is best around k = 5
#running knn with k=5
pred=knn(x_train2,x_test2,train2$target,k=5) 
a=table(pred,test2$target)
confusionMatrix(pred,test2$target,positive = "1")

##########SVM############
modelsvm = svm(target~., data = train2, kernel = "radial",cost = 10)
summary(modelsvm)
predsvm = predict(modelsvm,test2,type="response")
confusionMatrix(predsvm,test2$target)

#_______________________________________________________________________

# XGBOOST
library(mlbench)
library(caret)

######## split into train and #########

train.indices <- createDataPartition(data$target, p = .85, list = F)
trainingData <- data[train.indices,]
testData <- data[-train.indices,]

transformation <- preProcess(trainingData,method = c("range"))
trainingData <- predict(transformation, trainingData)
testData <- predict(transformation, testData)

xgb.ctrl <- trainControl(method = "repeatedcv", repeats = 3, number = 3,
                         search='random',
                         allowParallel=T)
######## check missing values#########

sum(is.na(trainingData))

##### missing value imputation using Central imputation ######
library(DMwR)
trainingData = centralImputation(trainingData)
str(data)
sum(is.na(trainingData))

######### cross validation of data ########

set.seed(20)
library(xgboost)
xgb.tune <-train(target~.,
                 data = trainingData,
                 method="xgbTree",
                 trControl=xgb.ctrl,
                 tuneLength=20,
                 verbose=T,
                 metric="Accuracy",
                 nthread=3)
xgb.tune

View(xgb.tune$results)
############ plotting data ######
a <- xgb.tune$results[order(xgb.tune$results$Accuracy, decreasing = TRUE),]
View(a)
par(mfrow=c(2,1))
hist(a$nrounds[1:10])
hist(a$nrounds[40:50])

#########To balance the dimension of the predicted matrix##########
estData <- testData[complete.cases(testData),]
# plot(xgb.tune)
preds <- predict(xgb.tune, testData)
confusionMatrix(testData$target, preds)
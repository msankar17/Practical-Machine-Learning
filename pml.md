# Practical Machine Learning
Sankar  
June 2, 2017  



## Introduction

Usually people quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants and predict the manner in which they did the exercise. 

There is a "classe" variable in the training set. We will create a report describing how we built your model, cross validation used, what the expected out of sample error is, and how we arrived at the final model. We will also use our final prediction model to predict 20 different test cases.


#Load data
We will load the data in variables for analysis purpose  
  
The training data for this project are available here:  
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  
The test data are available here:  
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.


```r
library(caret)

#Read data from train and test files
traindata=read.csv("./pml-training.csv",stringsAsFactors = FALSE)
testdata=read.csv("./pml-testing.csv",stringsAsFactors = FALSE)
```

## Pre process data
Replace all incorrect values to zero as applicable

```r
#Replace NA with zero in the traindataset
train1=traindata
train1[train1==""]=0
train1[train1=="#DIV/0!"]=0
train1[is.na(train1)]=0

test1=testdata
test1[test1==""]=0
test1[test1=="#DIV/0!"]=0
test1[is.na(test1)]=0
```

The data is converted into numeric values so that we can use them in train and prediction model.

```r
for(i in 7:159){
  train1[,i]=as.numeric(train1[,i])
  test1[,i]=as.numeric(test1[,i])
}
```

We will remove the first 5 columns as they do not add any specific details to the prediction exercise

```r
#Remove first 5 columns as they are not adding any specific detail
train1=train1[,-(1:5)]
orgtest=test1
test1=test1[,-(1:5)]
```

  
We will remove the variables that are having zero or near zero variance from analysis purpose from the below code. Also we use records with new_window="yes", since other records mostly doesnt have proper values for derived metrics. using them as well in variance analysis would yield incorrect results

```r
#Identify and remove near zero and zero variance variables
nzv= nearZeroVar(train1[train1$new_window=="yes",], saveMetrics = FALSE)
train2= train1[,-nzv]
test1=test1[,-nzv]
```

Remove the first two columns(columns 5,6 from the original data set) as we have compelted variance analysis and is not needed in the training model

```r
train2=train2[,-(1:2)]
test1=test1[,-(1:2)]
```


Check if there is any multicollinearity problem in the data set and drop out columns which are highly corellated among themselves which could hinder our ability to create a proper regression analysis. We didnt also check for corellation with Classe variable with others as some variables could have indirect corelation on Classe variable and shouldnt be taken

```r
#corelation matrix calculation
cor1=cor(train2[,1:142])
highcor=findCorrelation(cor1,cutoff = 0.75)
train3=train2[,-highcor]
test1=test1[,-highcor]
dim(train3)
```

```
## [1] 19622    84
```

```r
dim(test1)
```

```
## [1] 20 84
```

#RF model with CV with columns and 1% data
Lets validate once with RF validation with with PCA(Principle component analysis) and repeat 10 fold cross validation 3 times. lets print the model output and look at the accuracy

```r
#Random forest with 1% test data
trindex1= createDataPartition(train2$classe,p=0.01,list = FALSE)
train31=train2[trindex1,] 

control1=trainControl(method = "repeatedcv",number=10,repeats=3)
model=train(classe~.,data=train31,method="rf", preProcess="pca", trContrl=control1 )
model$finalModel[[5]]
```

```
##    A  B C  D  E class.error
## A 41  2 6  5  2   0.2678571
## B  3 24 3  4  4   0.3684211
## C 17  2 8  6  2   0.7714286
## D  4  3 5 17  4   0.4848485
## E  7  7 4  2 17   0.5405405
```

seems like out model accuracy is 47.0743769% which is very low and we would perform some more anlysis on that  

#Important variables through Decision tree modeling
Random forest is a time consuming computation and accuracy increases with more training data. But with more number of variables, the computation would take more time to finish.  
  
We can use Variable importance anlaysis to find which columns provide more meaning to the prediction and can use them in the next model training. We can use decistion trees for training the model that is helpful for predicting 

```r
#increasing test data % increases prediction accuracy
#identify important variables
model51=train(classe~.,data=train1,method="rpart")
impcols=row.names(varImp(model51)$importance)[varImp(model51)$importance>0]
impcols=c(impcols,"classe")
redtrain=train1[,names(train1) %in% impcols]

trindex= createDataPartition(redtrain$classe,p=0.5,list = FALSE)
redtrain1=redtrain[trindex,]
```
The model from decision trees has ~55% accuracy but has helped us identify the important columns alone which would improve the RF computation.

#Model Random Forest with important columns


```r
#Random Forest on the reduced columns of the dataset
control1=trainControl(method = "repeatedcv",number=10,repeats=3)
modelred=train(classe~.,data=redtrain1,method="rf", preProcess="pca", trContrl=control1 )

modelred$finalModel[[5]]
```

```
##      A    B    C    D    E class.error
## A 2741   40    7    2    0  0.01756272
## B  103 1675   76   37    8  0.11795682
## C    3   60 1603   40    5  0.06312098
## D    4   25  101 1452   26  0.09701493
## E    1   15   13   49 1726  0.04323725
```

The model accuracy has significantly improved to 92.0623051 and the outputs are shown below

#Model Stochastic Gradient Boosting model with important columns
Lets try GBM model with our dataset and see if it increases our prediction accuracy


```r
modelgbm=train(classe~.,data=redtrain1,method="gbm")
```

The model is not cross validated, but we would look at the accuracy and kappa metrics as below

```r
modelgbm$results[,c(2,4,5,6)]
```

```
##   interaction.depth n.trees  Accuracy     Kappa
## 1                 1      50 0.7222222 0.6471648
## 4                 2      50 0.8655150 0.8298109
## 7                 3      50 0.9307535 0.9124005
## 2                 1     100 0.7969150 0.7429631
## 5                 2     100 0.9406653 0.9249392
## 8                 3     100 0.9774830 0.9715170
## 3                 1     150 0.8419330 0.8000682
## 6                 2     150 0.9705934 0.9627986
## 9                 3     150 0.9900919 0.9874683
```

Looking at the above we have 99% Accuracy in with Boosting models and 93% accuracy with RF with 1/2 of training data which can run in a reasonable time

Lets predict the output of test data using Boosting model 

```r
#Predict the values from the test set
impcols1=c(impcols,"problem_id")
redtest=orgtest[,names(orgtest) %in% impcols1]
predict(modelgbm,redtest)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

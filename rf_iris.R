require(randomForest)

## Classification:
data(iris)
set.seed(71)
iris.rf <- randomForest(Species ~ ., data=iris, importance=TRUE,
                        proximity=TRUE)
# Many features here - ie could use  maxnodes=4 to set a max on the tree size


## Look at the results
print(iris.rf)

## Look at error rate for individual trees
print(iris.rf$err.rate[1:10,]) # first 10 trees

## Plot the error rates as function of number of trees
plot(iris.rf)

## Look at an individual tree
getTree(iris.rf, 3, labelVar=TRUE)
# plotting the tree takes work - see: http://stats.stackexchange.com/questions/41443/how-to-actually-plot-a-sample-tree-from-randomforestgettree

## Use the Random Forest for Prediction
test_observation = iris[2,1:4] # simulate a test observation using the 2nd iris data observation
predict(iris.rf, test_observation)

## Look at predicted classes for individual observations
print(iris.rf$err.rate[1:10]) # first 10 observations

## Look at confusion matrix
print(iris.rf$confusion) 

## Look at fraction of votes in each class, for each observation
print(iris.rf$votes[1:10,]) # first 10 observations
print(iris.rf$votes)# all observations

#Scatterplot the data, coloring by class.
# Note that we see overlap between the blue and green classes
plot(iris$Petal.Length,iris$Petal.Width,pch=16,cex=2,
     col=rgb(iris.rf$votes[,1],iris.rf$votes[,2],iris.rf$votes[,3]))

# add a background to the scatterplot
temp = iris[1:2,]
for (x in seq(1, 7, 0.1)){
  for (y in seq(0, 2.5, 0.1)){
    temp[1,1] = median(iris[,1])
    temp[1,2] = median(iris[,2])
    temp[1,3] = x
    temp[1,4] = y
    color = predict(iris.rf, temp, predict.all=TRUE, norm.votes=TRUE)
    r = rowSums(color$individual=='setosa')
    g = rowSums(color$individual=='versicolor')
    b = rowSums(color$individual=='virginica')
    points(x,y, col = rgb(r/500,g/500,b/500))
  }
}

## Look for outliers
## The outliers are observations that go through trees different than other observations in the same claee
## Here we use the outlier metric for scale the transparency (outliers are less transparent)
#compute the outlier metric
out = outlier(iris.rf, iris[,5])  
#compute the transparency from the outlier metric
out_shade = (out-min(out))/(max(out)-min(out))
out_shade = (out_shade+0.01)/1.01
#Compute the color from each class
r = c(1,0,0)[unclass(iris$Species)]
g = c(0,1,0)[unclass(iris$Species)]
b = c(0,0,1)[unclass(iris$Species)]
#Make the plot
plot(iris$Petal.Length,iris$Petal.Width,pch=16,cex=2,
     col=rgb(r,g,b,out_shade))


## Plot the variable importance
varImpPlot(iris.rf)

## There is an option in the software to do a search for a good value of mtry:
tuneRF(iris[,-5], iris[,5], 500, ntreeTry=100, stepFactor=1.1, improve=0.0001,
       trace=TRUE, plot=TRUE, doBest=FALSE)


######################## Boosting Trees

install.packages("adabag")
require(adabag)
require(rpart)

## rpart library should be loaded
data(iris)

# create a training sample
train <- c(sample(1:50, 25), sample(51:100, 25), sample(101:150, 25))
# create the boosted trees model from the training set
iris.adaboost <- boosting(Species ~ ., data = iris[train, ], mfinal = 10,
                          control = rpart.control(maxdepth = 1))
iris.adaboost

# plot the variable importance
barplot(iris.adaboost$imp[order(iris.adaboost$imp, decreasing = TRUE)],
        ylim = c(0, 100), main = "Variables Relative Importance",
        col = "lightblue")

# Table of prediction results
table(iris.adaboost$class, iris$Species[train],
      dnn = c("Predicted Class", "Observed Class"))

# compute the error
1 - sum(iris.adaboost$class == iris$Species[train]) /
  length(iris$Species[train])

# training boosted trees for prediction
iris.predboosting <- predict.boosting(iris.adaboost,
                                      newdata = iris[-train, ])
iris.predboosting

# use the built in r-fold cross validation
iris.boostcv <- boosting.cv(Species ~ ., v = 10, data = iris, mfinal = 10,
                            control = rpart.control(maxdepth = 1))

# check the results
iris.boostcv

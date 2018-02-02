install.packages("RWeka", dependencies = TRUE)
install.packages("e1071", dependencies = TRUE)
install.packages("caret", dependencies = TRUE)

mySVM <- function(d) {
  # SVM classifier
  dataset <- divideData(d)
  col<-c("Male.Life", "Female.Life", "Continent")
  svmfit <- svm(Continent ~., data = dataset$training, kernel = "linear", cost = .1, scale = FALSE)
  plot(svmfit,dataset$training[,col])
  p <- predict(svmfit, dataset$training[,col], type="class")
  plot(p)
  trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
  set.seed(3233)
  library(caret)
  fit <- train(Continent ~., data = dataset$training, method = "svmLinear",
               trControl=trctrl,
               preProcess = c("center", "scale"),
               tuneLength = 10)
}

myKNN <- function(d) {
  # kNN classifier
  fit <- IBk(paste(tail(colnames(d),1),".",sep="~"), data=d, control = Weka_control(K=ceiling(sqrt(nrow(d))), X=TRUE))
}
myRipper <- function(d) {
  # Ripper classifier
  c <- ncol(d)
  fit <- train(d[,1:c-1], d[,c], method="JRip",preProcess = c("center", "scale"),tuneLength = 10,trControl = trainControl(method = "cv"))
}

myC45 <- function(d) {
  # C4.5 classifier
  fit <- J48(paste(tail(colnames(d),1),".",sep="~"), data=d)
}
myPredict <- function(fit, d) {
  #Predictor
  num_col <- ncol(d)
  predictions <- predict(fit, d[,1:num_col-1])
  t <- table(predictions,d[,num_col])
}

divideData <- function(d) {
  # Divide the data set into test and training sets
  set.seed(1)
  num_rows <- nrow(d)
  n <- as.integer(row.names(d))
  s <- sample(num_rows,0.8*num_rows)
  dataset <- list(training=d[s,],test=d[n[!n %in% s],])
}

generateStats <- function(t) {
  # Generates the confusion matrix values for each class of the prediction table
  rnames <- rownames(t)
  cnames <- colnames(t)
  tp <- c()
  fp <- c()
  fn <- c()
  tn <- c()
  for(a in cnames) {
    tp <- c(tp,sum(t[a,a])) #tp
    fp <- c(fp,sum(t[a,cnames[!cnames %in% a]])) #fp
    fn <- c(fn,sum(t[rnames[!rnames %in% a],a])) #fn
    tn <- c(tn,sum(t[rnames[!rnames %in% a],cnames[!cnames %in% a]])) #tn
  }
  x <- data.frame(tp,fp,fn,tn)
  rownames(x) = cnames
  x
}

printConfusionMatrix <- function(t) {
  # Prints the confusion matrix for each class
  cat("\nConfusion Matrices:\n\n")
  rnames <- rownames(t)
  for(a in rnames) {
    x <- matrix(t[a,],2,2)
    rownames(x) <- paste("p:",c(a,paste("not", a)), sep="")
    colnames(x) <- paste("a:",c(a,paste("not", a)), sep="")
    print(x)
    cat("\n")
  }
}

generateMeasures <- function(t) {
  # Generates various measures for each of the class
  rnames <- rownames(t)
  accuracy <- c()
  precision <- c()
  recall <- c()
  fmeasure <- c()
  for(a in rnames) {
    acc <- sum(t[a,"tp"],t[a,"tn"]) / sum(t[a,"tp"],t[a,"tn"],t[a,"fp"],t[a,"fn"])
    accuracy <- c(accuracy,acc)
    prec <- sum(t[a,"tp"]) / sum(t[a,"tp"],t[a,"fp"])
    precision <- c(precision,prec)
    rec <- sum(t[a,"tp"]) / sum(t[a,"tp"],t[a,"fn"])
    recall <- c(recall,rec)
    fm <- (2 * prec * rec)/(prec + rec)
    fmeasure <- c(fmeasure,fm)
  }
  x <- data.frame(accuracy, precision, recall, fmeasure)
  rownames(x) = rnames
  colnames(x) = c("accuracy", "precision", "recall", "fmeasure")
  x
}


totalAccuracy <- function(t) {
  # Total accuracy of the dataset for a given algorithm
  sum(diag(t))/sum(t)
}

classificationResults <- function(algo,d) {
  cat("\n")
  cat(paste("Classification for",algo))
  cat(":\n\n")
  
  # The main function
  dataset <- divideData(d)
  fit <- switch (algo,
                 C45 = myC45(dataset$training),
                 KNN = myKNN(dataset$training),
                 Ripper = myRipper(dataset$training),
                 SVM = mySVM(dataset$training)
  )
  
  predictions <- myPredict(fit, dataset$test)
  
  cat("\nPrediction Table:\n")
  print(predictions)
  cat("\n")
  
  if(algo == "C45") {
    plot(fit)
  }
  
  st = generateStats(predictions)
  
  printConfusionMatrix(st)
  
  results = generateMeasures(st)
  
  cat("\nMeasures table:\n\n")
  print(results)
  cat("\n\n")
  cat(paste("Total accuracy of",algo,"for the test set:",totalAccuracy(predictions)))
  cat("\n\n")
}

library(RWeka)
library(e1071)
library(caret)

setwd('C:/Users/rushi/Downloads')            #set working directory to where you saved the .csv file

d <- read.csv("LifeExpectancyDataset.csv",header = TRUE)

d <- d[,c(4,5,6)]

d <- d[sample(nrow(d)),]                     #execute this command only if you want to shuffle dataset

classificationResults("Ripper",d)
classificationResults("C45",d)
classificationResults("KNN",d)
classificationResults("SVM",d)

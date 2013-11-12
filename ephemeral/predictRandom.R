test <- read.csv('test.tsv',sep = "\t")
train <- read.csv('train.tsv',sep ='\t')
trainData <- train[,c(4:26)]
testData <-test[,c(4:26)]
###############
trainData1 <- train[,3]
testData1 <-test[,3]
for (i in 1:length(trainData1) ){
  filename = sprintf('files/train%d.txt',i)
  write(trainData1[i],filename)
}

for (i in 1:length(testData1) ){
  filename = sprintf('files/test%d.txt',i)
  write(testData1[i],filename)
}
#bin/mallet import-dir --input ../kaggle/ephemeral/files --output ../kaggle/ephemeral/files/tutorial.mallet --keep-sequence --remove-stopwords --stoplist-file stoplists/en.txt
#bin/mallet train-topics --input ../kaggle/ephemeral/files/tutorial.mallet --num-topics 2 --output-state ../kaggle/ephemeral/files/topic-state.gz --output-topic-keys ../kaggle/ephemeral/files/keys.txt --output-doc-topics ../kaggle/ephemeral/files/tutorial_composition.txt
a <- read.csv('tutorial_composition.csv',sep = "\t",header=FALSE,skip=1)
a<-a[,c(4,6)]
b <- a[,3] * a[,4] + a[,5] * a[,6]

a1<- read.csv('extratrain.csv',sep='\t',header=FALSE)
a2<- read.csv('extratest.csv',header=FALSE)
c <- a1[,1] * a1[,2] + a1[,3] * a1[,4]
d <- a2[,1] * a2[,2] + a2[,3] * a2[,4]
trainData$conf <- c
testData$conf <-d

# 
###############
trainLabels<- train[,27]
trainData[,2]<- as.numeric(paste(trainData[,2]))
testData[,2]<- as.numeric(paste(testData[,2]))
rf1 <- 
colnames(train)
predict <- test[,2];
predict <- as.data.frame(predict)
predict$label = 1
colnames(predictions) = c('urlid','label')
write.csv(predictions,'submission2.csv',row.names = FALSE)
uLevels <- function(names){ #col 1
  for(c in names){
    print(c)
    trainData[,c] <<- factor(as.character(trainData[,c]))
    testData[,c] <<- factor(as.character(testData[,c]))
    map <- mapLevels(x=list(trainData[,c], testData[,c]), codes=FALSE, combine=TRUE)
    mapLevels(trainData[,c]) <<- map
    mapLevels(testData[,c]) <<- map
  }
}
trainLabels <-as.factor(trainLabels)
trainLabels <- as.data.frame(trainLabels)
trainLables <- trainLabels[,1]
trainData[which(is.na(trainData[,2])),2] = 0 
testData[which(is.na(testData[,2])),2] = 0
rf2 <- randomForest(x = trainData,y = trainLabels,na.action = na.roughfix)
label <- predict(rf2,testData)
predictions <- test[,2]
predictions <- as.data.frame(predictions)
#predictions <- predictions[,1]
predictions$label <- label
colnames(predictions) = c('urlid','label')
write.csv(predictions,'submission4.csv',row.names = FALSE)

install.packages('gdata')
install.packages('RTextTools')
library('gdata')
library('Boruta')
a <- Boruta(trainData,trainLabels,doTrace = 2,confidence = 0.7)
trainData1 <-trainData[,c(1:9,11:23)]
testData1 <-testData[,c(1:9,11:23)]
rf2 <- randomForest(x = trainData1,y = trainLabels,na.action = na.roughfix)
label <- predict(rf,testData)
predictions$label <- label
colnames(predictions) = c('urlid','label')
write.csv(predictions,'submission3.csv',row.names = FASLE)
###RtextTools
library('RTextTools')
doc_matrix <- create_matrix(train$boilerplate, language="english", removeNumbers=TRUE, removePunctuation=TRUE, stemWords=TRUE, removeSparseTerms=.998)
doc_matrix <- create_matrix(train$boilerplate[671:671], language="english", removeNumbers=TRUE, removePunctuation=TRUE, stemWords=TRUE, removeSparseTerms=.998)
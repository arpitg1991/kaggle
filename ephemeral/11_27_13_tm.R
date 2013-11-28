library(gdata)
library(randomForest)
library(topicmodels)

# -------------------------------------
# Load Data
# -------------------------------------
train <- read.csv('train_raw.tsv',sep = "\t")
test <- read.csv("test.tsv",sep = "\t")

boilerPlate_train <- as.character(train[,3])
boilerPlate_test <- as.character(test[,3])

library(tm)
library(lsa)
library("tau")
library("SnowballC")

corpus <- Corpus(VectorSource(boilerPlate_train))
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, stemDocument, language = "english")
corpus <- tm_map(corpus, function(x) removeWords(x, stopwords("english")))
dtm_train <- DocumentTermMatrix(corpus)

corpus_test <- Corpus(VectorSource(boilerPlate_test))
corpus_test <- tm_map(corpus_test, tolower)
corpus_test <- tm_map(corpus_test, removePunctuation)
corpus_test <- tm_map(corpus_test, stemDocument, language = "english")
corpus_test <- tm_map(corpus_test, function(x) removeWords(x, stopwords("english")))
dtm_test <- DocumentTermMatrix(corpus_test)

lda_train <- LDA(dtm_train,12)

require(reshape2)
require(ggplot2)
require(RColorBrewer)

lda_topics <- topics(lda_train,12)

train_topics2 <- posterior(lda_train, dtm_train)
train_topics <- train_topics2[2]
train_topics <- as.data.frame(train_topics)
train_topics1 <- cbind(train[,2],train_topics[,c(1:12)])
colnames(train_topics1) <- c('urlid',c(1:12))
write.table(file="topicModel_train.csv",train_topics1,row.names=FALSE)

test_topics2 <- posterior(lda_train, dtm_test)
test_topics <- test_topics2[2]
test_topics <- as.data.frame(test_topics)
test_topics1 <- cbind(test[,2],test_topics[,c(1:12)])
colnames(test_topics1) <- c('urlid',c(1:12))
write.table(file="topicModel_test.csv",test_topics1,row.names=FALSE)
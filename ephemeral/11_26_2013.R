gram_train_3 <- data.frame(read.csv('trainfilerainTfidf3gram.csv'))
gram_train_2 <- data.frame(read.csv('trainfilerainTfidf2gram.csv'))
trainReal <- data.frame(read.csv('train_raw.tsv',sep='\t'))
trainReal <- trainReal[,c(2,27)]
gram_test_3 <- data.frame(read.csv('testTfidf3gram.csv'))
gram_test_2 <- data.frame(read.csv('testTfidf2gram.csv'))

tm_train = data.frame(read.csv('trainConf.csv', header = FALSE, sep = ' '))
tm_test = data.frame(read.csv('testConf.csv', header = FALSE, sep = ' '))
colnames(tm_train) <- c("urlid",c(1:2))
colnames(tm_test) <- c("urlid", c(1:2))

train_merged <- merge(x = tm_train, y = gram_train_2, by = "urlid")
test_merged <- merge(x = tm_test, y = gram_test_2, by = "urlid")
#train_merged <- merge(x = train_merged, y = gram_train_3, by = "urlid")
#test_merged <- merge(x = test_merged, y = gram_test_3, by = "urlid")
trainWithRealLabel <- merge(x=train_merged,y=trainReal,by="urlid")
write.table(train_merged, "train_merged.tsv", sep="\t", row.names=FALSE, col.names=TRUE)
write.table(test_merged, "test_merged.tsv", sep="\t", row.names=FALSE, col.names=TRUE)
write.table(trainWithRealLabel, "trainWithRealLabel.tsv", sep="\t", row.names=FALSE, col.names=TRUE)

gram_2 <- data.frame(read.csv('LR.csv'))
gram_3 <- data.frame(read.csv('testTfidf3gram.csv'))
final_model <- merge(x=gram_2,y=gram_3,by="urlid")
final_urlid <- final_model$urlid
final_labels <- transform((final_model$label.x*0.3 + final_model$label.y*0.7))
final_pred <- cbind(final_urlid, final_labels)
colnames(final_pred) <- c('urlid','label')
write.table(final_pred, "finalfinal.csv", sep=",", row.names=FALSE, col.names=TRUE)

test <- read.csv('test.tsv',sep = "\t")
predict <- test[,2];
predict <- as.data.frame(predict)
predict$label = 1
colnames(predict) = c('urlid','label')
write.csv(predict,'submission1.csv',row.names = FALSE)


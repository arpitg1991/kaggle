gram_train_3 <- data.frame(read.csv('trainfilerainTfidf3gram.csv'))
gram_train_2 <- data.frame(read.csv('trainfilerainTfidf2gram.csv'))

gram_test_3 <- data.frame(read.csv('testTfidf3gram.csv'))
gram_test_2 <- data.frame(read.csv('testTfidf2gram.csv'))

tm_train = data.frame(read.csv('trainConf.csv', header = FALSE, sep = ' '))
tm_test = data.frame(read.csv('testConf.csv', header = FALSE, sep = ' '))
colnames(tm_train) <- c("urlid",c(1:12))
colnames(tm_test) <- c("urlid", c(1:12))

train_merged <- merge(x = tm_train, y = gram_train_2, by = "urlid")
test_merged <- merge(x = tm_test, y = gram_test_2, by = "urlid")

write.table(train_merged, "train_merged.tsv", sep="\t", row.names=FALSE, col.names=TRUE)
write.table(test_merged, "test_merged.tsv", sep="\t", row.names=FALSE, col.names=TRUE)

gram_2 <- data.frame(read.csv('LR.csv'))
gram_3 <- data.frame(read.csv('testTfidf3gram.csv'))
final_urlid <- gram_2$urlid
final_labels <- transform((gram_2$label*0.3 + gram_3$label*0.7))
final_pred <- cbind(final_urlid, final_labels)
colnames(final_pred) <- c('urlid','label')
write.table(final_pred, "finalfinal.csv", sep=",", row.names=FALSE, col.names=TRUE)

library(e1071)
library(pROC)
#load("~/Downloads/out.RData")
out.df <- read.table('~/shared_scratch/group 5/task2/train.txt')
#data.meta <- read.table('~/Downloads/dataset1_meta.txt')[,c("nUMI")]
pcs <- read.table('~/shared_scratch/group 5/task2/pca_cells.txt')[,c(1,2)]

# subset out ambiguous
out.df.subset <- subset(out.df,V12!="ambs")

# define x and y
y <- ifelse(out.df.subset$V12=="doublet",1,0)
##data.mat <- out.df.subset[,c(1,11)];
data.mat <- out.df.subset[,-c(2,10,12)];
data.mat <- apply(data.mat, 2, function(x) {(x-mean(x))/sd(x)})
data.mat <- data.frame(y=factor(y),data.mat)
data.mat1 <- cbind(data.mat,pcs) # entropy and pcs and numis
data.mat2 <- cbind(y,out.df.subset[,c(1)]) # numis
data.mat3 <- cbind(data.mat2,pcs) #numis and pcs

# class imbalance weights
##wts <- 100/table(data.mat$y)

# with entropy and PCs
fit <- glm(y~., data=as.data.frame(data.mat), family=binomial(link='logit'));
##fit <- svm(y~.,data=data.mat1,type="C-classification",kernel="linear",cost=.001,class.weights=wts)
##print(sum(fit$fitted==data.mat1$y)/length(fit$fitted))

fit1 <- glm(y~., data=as.data.frame(data.mat1), family=binomial(link='logit'));

# just nUMIs
fit2 <- glm(y~., data=as.data.frame(data.mat2), family=binomial(link='logit'));
##fit2 <- svm(y~.,data=data.mat2,type="C-classification",kernel="linear",cost=.001,class.weights=wts)
##print(sum(fit2$fitted==data.mat2[,1])/length(fit2$fitted))

# just nUMIs and PCs
fit3 <- glm(y~., data=data.mat3, family=binomial(link='logit'));
##fit3 <- svm(y~.,data=data.mat3,type="C-classification",kernel="linear",cost=.001,class.weights=wts)
##print(sum(fit3$fitted==data.mat3[,1])/length(fit3$fitted))


# now predict on test set using the first model
test <- read.table('~/shared_scratch/group 5/task2/test.txt')
test.pcs <- read.table('~/shared_scratch/group 5/task2/test_pca_cells.txt')[,c(1,2)]
##data.mat <- test[,c(1)]
##data.mat <- test[,-c(2,10)];
data.mat <- apply(data.mat, 2, function(x) {(x-mean(x))/sd(x)})
data.mat1 <- cbind(data.mat,test.pcs) # entropy and pcs and numis
yhat <-predict.glm(fit, as.data.frame(data.mat1),type="response")
yhat.label <- rep("singlet",length(yhat))
yhat.label[which(yhat>0.3)] <- "doublet"

cell.names <- read.table('~/jamboree/doublet-datasets/dataset1/Ye2_L001_001_test_labels_predict_me.txt')
write.table(cbind(cell.names,yhat,yhat.label),file='~/shared_scratch/group 5/task2/test_preds2.txt',row.names=FALSE)

roc(out.df$V12, fit$prob[,1])
##Area under the curve: 0.8892
roc(out.df$V12, fit1$fitted)
##Area under the curve: 0.8874
roc(out.df$V12, fit2$fitted)
##Area under the curve: 0.8535
roc(out.df$V12, fit3$fitted)
##Area under the curve: 0.8574
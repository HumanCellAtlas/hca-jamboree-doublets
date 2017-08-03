library(caret)
library(e1071)
#load("~/Downloads/out.RData")
out.df <- read.table('~/Downloads/train.txt')
#data.meta <- read.table('~/Downloads/dataset1_meta.txt')[,c("nUMI")]
pcs <- read.table('~/Downloads/pca_cells.txt')[,c(1,2)]

# subset out ambiguous
out.df.subset <- subset(out.df,V12!="ambs")

# define x and y
y <- ifelse(out.df.subset$V12=="doublet",1,0)
data.mat <- cbind(y,out.df.subset[,c(1,11)])
data.mat1 <- cbind(data.mat,pcs) # entropy and pcs and numis
data.mat2 <- cbind(y,out.df.subset[,c(1)]) # numis
data.mat3 <- cbind(data.mat2,pcs) #numis and pcs

# class imbalance weights
wts <- 100/table(data.mat$y)

# with entropy and PCs
fit <- svm(y~.,data=data.mat1,type="C-classification",kernel="linear",cost=.001,class.weights=wts)
print(sum(fit$fitted==data.mat1$y)/length(fit$fitted))

# just nUMIs
fit2 <- svm(y~.,data=data.mat2,type="C-classification",kernel="linear",cost=.001,class.weights=wts)
print(sum(fit2$fitted==data.mat2[,1])/length(fit2$fitted))

# just nUMIs and PCs
fit3 <- svm(y~.,data=data.mat3,type="C-classification",kernel="linear",cost=.001,class.weights=wts)
print(sum(fit3$fitted==data.mat3[,1])/length(fit3$fitted))


# now predict on test set using the first model
test <- read.table('~/Downloads/test.txt')
test.pcs <- read.table('~/Downloads/test_pca_cells.txt')[,c(1,2)]
data.mat <- test[,c(1,11)]
data.mat1 <- cbind(data.mat,test.pcs) # entropy and pcs and numis
yhat <-predict(fit,data.mat1)
cell.names <- read.table('~/Downloads/Ye2_L001_001_test_labels_predict_me.txt')
write.table(cbind(cell.names,yhat),file='~/Downloads/test_preds.txt',row.names=FALSE)

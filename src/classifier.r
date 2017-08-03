### download dataset 1
library(ggplot2)
library("Matrix")
expr = readMM("~/jamboree//doublet-datasets/dataset1//Ye2_sparse_molecule_counts.mtx")
genes = read.csv("~/jamboree//doublet-datasets/dataset1//Ye2_gene_id.csv", header = F, stringsAsFactor = F)
barcodes = read.csv("~/jamboree//doublet-datasets/dataset1//Ye2_barcode_id.csv", header = F, stringsAsFactor = F)
lbl = read.csv("~/jamboree/doublet-datasets/dataset1/Ye2_L001_001_train_labels.txt",sep="\t",header=F)
lbl.predict = read.csv("~/jamboree/doublet-datasets/dataset1/Ye2_L001_001_test_labels_predict_me.txt",sep="\t",header=F)
umis = t(expr)
rownames(umis) = genes$V2; colnames(umis) = barcodes$V2
#umis = as.matrix(umis[ rowSums(umis) > 10, colSums(umis) > 1000 | (colnames(umis) %in% lbl.predict[,1]) ])
umis = as.matrix(umis[ rowSums(umis) > 10,
                       colnames(umis) %in% c(as.character(lbl.predict[,1]),as.character(rownames(lbl)))])
barcodes.shared = intersect(lbl[,1],colnames(umis))
#umis = umis[,barcodes.shared];
rownames(lbl)=lbl[,1]; lbl=lbl[barcodes.shared,]
dim(umis)

#normalise umis
umis1 = t( t(umis)/apply(umis,2,sum) )

# choose informative genes
tm = apply(umis1,1,mean); tsd = apply(umis1,1,sd)
fit = lm(log(tsd) ~ log(tm))
qplot(log(tm),log(tsd))+geom_line(aes( log(tm),(fit$fitted.values),colour="red"))

qplot( log(tm),log(tsd)-fit$fitted.values  )
sum( log(tsd)-fit$fitted.values > 0.2 )
genes.inf = rownames(umis)[ log(tsd)-fit$fitted.values > 0.2 ]

### make PCA
library(pcaMethods)
p = pca( t( cbind(umis1)[genes.inf,]),nPcs=40)
qplot(p@scores[,1],p@scores[,2],alpha=I(0.1),colour = as.factor( lbl$V2%in%"singlet") )
### run random forest classifier
library(randomForest)
library(pROC)
set.seed(415)
feat = data.frame( log(apply(umis,2,sum)),
                   log(apply(umis,2,function(x) sum(x>0))),
                   log(apply(umis1,2,sd)),
                   apply(umis,2,function(x) sum(x==1)/sum(x>=1)) ) 
pred = as.data.frame(cbind( feat[,1],feat[,2],feat[,3],feat[,4],p@scores[,1:20]))
train = sample(nrow(lbl),5000)
train = !is.na(match(1:nrow(lbl),train))
test = !train
fit <- randomForest( as.numeric(lbl[,2] %in% "singlet") ~., data=pred[barcodes.shared,][,],
                     importance=TRUE, 
                     ntree=500)
x = predict( fit,pred[barcodes.shared,][train,])
roc(as.numeric(lbl[train,2] %in% "singlet") ~ x)
x = predict( fit,pred[barcodes.shared,][test,])
roc(as.numeric(lbl[test,2] %in% "singlet") ~ x)
qplot(x,colour=as.factor(lbl[test,2] %in% "singlet"),geom="density" )

x = predict( fit,pred)
roc(as.numeric(lbl[test,2] %in% "singlet") ~ x)
qplot(x,colour=as.factor(lbl[test,2] %in% "singlet"),geom="density" )

res = as.data.frame( x );
res$status = "doublet"; res$status[res$x > 0.75 ] = "singlet"
res$name=rownames(res)

write.table(res[as.character(lbl.predict[,1]),c("name","status","x")],file="~/shared_scratch/group7/task1_classification.txt",
            sep="\t",quote=F,row.names=FALSE,col.names=FALSE)

  



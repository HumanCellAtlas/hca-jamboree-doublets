# Libraries and Code
library(ggvis)
source("VarGenes.R")
library(Rtsne)
library(viridis)
library(Matrix)

# Load training Labels
trainLabels = read.table("../../../../../jovyan/jamboree/doublet-datasets/dataset1/Ye2_L001_001_train_labels.txt", row.names = 1)
colnames(trainLabels) = c("class")
cellnames = rownames(trainLabels)
trainLabels = trainLabels$class
names(trainLabels) = cellnames

# Load Training data
trainData = read.table("../../../../../jovyan/shared_scratch/group9/task2/doublets-dataset/dataset1/Ye2_L001_001_train.dge.txt.gz", header=TRUE, row.names = 1)


# Check if trainLabels cells are in the matrix and subset for cells with moderate to high complexity
cells.use = intersect(colnames(trainData), rownames(trainLabels))
cells.use = cells.use[colSums(trainData[,cells.use] > 0) > 300] # Use cells with > 300 genes

trainLabels = trainLabels[cells.use]
trainData  = trainData[,cells.use]

# Metrics
numGenes = colSums(trainData > 0)
numTranscripts = colSums(trainData)
normData = scale(trainData, center=FALSE, scale=numTranscripts)
entropyData = apply(normData,2,function(x) sum(-x[x>0]*log(x[x>0]) ))

# Combine everything
df = data.frame(numGenes=numGenes, numTranscripts=numTranscripts, entropy=entropyData, cell=fullLabels)

#Some plots
df %>% ggvis(~log(numGenes), ~log(numTranscripts), fill = ~factor(cell)) %>% layer_points()
df %>% group_by(cell) %>% ggvis(~entropy, fill = ~cell) %>% layer_densities()
df %>% group_by(cell) %>% ggvis(~numGenes, fill = ~cell) %>% layer_densities()
df %>% group_by(cell) %>% ggvis(~numTranscripts, fill = ~cell) %>% layer_densities()
df %>% ggvis(~numGenes) %>% layer_densities()

# Compute variable genes variable genes using NB fit
pdf("VarGenes.pdf",w=15,h=7)
var_genes = new.meanCVfit(trainData, mean.min = 0.005, diffCV.cutoff = 0.2)
dev.off()
write(var_genes,"VariableGenes_dataset1.txt")

# PCA with log(NormCounts+1)
med_trans = median(colSums(trainData))
logNormCounts = log(med_trans*normData + 1)
pca.logNormcounts.out = princomp(t(logNormCounts[var_genes,]))
# Pick top 20 PCs
logNormCounts.scores = as.data.frame(pca.logNormcounts.out$scores[,c(1:20)])
logNormCounts.loadings = as.data.frame(pca.logNormcounts.out$loadings[,c(1:20)])

logNormCounts.scores$cell = df[rownames(logNormCounts.scores),"cell"]
logNormCounts.scores$numGenes = df[rownames(logNormCounts.scores),"numGenes"]
logNormCounts.scores$numTranscripts = df[rownames(logNormCounts.scores),"numTranscripts"]
logNormCounts.scores$entropy = df[rownames(logNormCounts.scores),"entropy"]

# PCA Plot
logNormCounts.scores %>% group_by(cell) %>% ggvis(~Comp.1, ~Comp.2, shape =~cell, stroke =~cell, size:=3) %>% layer_points()


#TSNE based on first 10 PCs
Noise = matrix(1e-5*rnorm(10*nrow(logNormCounts.scores)), nrow = nrow(logNormCounts.scores)) 
tsne.logNorm.out = Rtsne(X=as.matrix(logNormCounts.scores[,c(1:10)]) + Noise , theta = 0.3, pca = FALSE, verbose = TRUE)


# random Forest classification (based on variable gene normalized counts, entropy, numTranscripts and numGenes)
library(randomForest)
predictor_Data = as.matrix(logNormCounts[var_genes,])
predictor_Data = rbind(predictor_Data, dfNorm[colnames(predictor_Data), "entropy"])
predictor_Data = rbind(predictor_Data, dfNorm[colnames(predictor_Data), "numTranscripts"])
predictor_Data = rbind(predictor_Data, dfNorm[colnames(predictor_Data), "numGenes"])
rownames(predictor_Data)[c(416:418)] = c("entropy","numTranscripts","numGenes")
training.set = c(); test.set=c()
training.label = c(); test.label=c();
for (i in levels(dfNorm$cell)){
  cells.in.clust = rownames(subset(dfNorm, cell == i));
  n = round(length(cells.in.clust)*0.7);
  train.temp = cells.in.clust[sample(length(cells.in.clust))][1:n]
  test.temp = setdiff(cells.in.clust, train.temp)
  training.set = c(training.set,train.temp); test.set=c(test.set,test.temp)
  training.label = c(training.label, rep(i,length(train.temp))); test.label = c(test.label, rep(i, length(test.temp)));
}

# Train data
rf=randomForest(x=t(predictor_Data[,training.set]), y=factor(training.label), importance = TRUE, ntree = 301, keep.inbag=TRUE, replace=FALSE) 
Conf_OOB0 = rf$confusion
test.predict = predict(rf,t(predictor_Data[,test.set]))
Conf_test = table(test.label,test.predict) # This gives an estimate of the validation accuracy on held out data
test.predict.prob = predict(rf,t(predictor_Data[,test.set]), type="prob")
#auc(test.label, test.predict.prob[,"singlet"])

# Testing 

# Read data
trainData = read.table("../../../../../jovyan/shared_scratch/group9/task2/doublets-dataset/dataset1/", header=TRUE, row.names = 1)
testData_ids= read.table("../../../../../jovyan/jamboree/doublet-datasets/dataset1/Ye2_L001_001_test_labels_predict_me.txt")
testData = readMM("../../../../../jovyan/jamboree/doublet-datasets/dataset1/Ye2_sparse_molecule_counts.mtx")
cellids = read.csv("../../../../../jovyan/jamboree/doublet-datasets/dataset1/Ye2_barcode_id.csv", header = FALSE)$V2
rownames(testData) = cellids
geneids = as.character(read.csv("../../../../../jovyan/jamboree/doublet-datasets/dataset1/Ye2_gene_id.csv", header = FALSE)$V2)
colnames(testData) = geneids
testData = t(testData)
genes = intersect(rownames(testData), rownames(trainData))
testData = testData[genes,testData_ids$V1]

# Metrics
numGenesTest = colSums(testData > 0)
numTranscriptsTest = colSums(testData)
normDataTest = scale(testData, center=FALSE, scale=numTranscriptsTest)
entropyDataTest = apply(normDataTest,2,function(x) sum(-x[x>0]*log(x[x>0]) ))

med_trans1 = median(colSums(testData))
logNormCountsTest = log(med_trans1*normDataTest + 1)
logNormCountsTest = rbind(logNormCountsTest, matrix(0, nrow=4, ncol = ncol(logNormCountsTest)))
rownames(logNormCountsTest)[25645:25648] = setdiff(var_genes, rownames(testData))

# Combine everything
dfTest = data.frame(numGenes=numGenesTest, numTranscripts=numTranscriptsTest, entropy=entropyDataTest)

predictor_DataTest = as.matrix(logNormCountsTest[var_genes,])
predictor_DataTest = rbind(predictor_DataTest, dfTest[colnames(predictor_DataTest), "entropy"])
predictor_DataTest = rbind(predictor_DataTest, dfTest[colnames(predictor_DataTest), "numTranscripts"])
predictor_DataTest = rbind(predictor_DataTest, dfTest[colnames(predictor_DataTest), "numGenes"])
rownames(predictor_DataTest)[c(416:418)] = c("entropy","numTranscripts","numGenes")


# Predict using the earlier RF
final.test.predict.prob = predict(rf,t(predictor_DataTest), type="prob") # Class probabilities
final.test.predict = predict(rf,t(predictor_DataTest))

FinalOutput = data.frame(prediction = final.test.predict, doublet_prob = final.test.predict.prob[,"doublet"],
                         singlet_prob = final.test.predict.prob[,"singlet"])

rownames(FinalOutput) = names(final.test.predict)
write.table(FinalOutput, file = "Final_predictions_Group9_Karthik.txt", sep="\t")






doubIDs <- read.table('jamboree/doublet-datasets/dataset2/Ye032917_S4_L003_001_train_labels.txt')
doubIDs$TF <- rep(FALSE, nrow(doubIDs))
doubIDs$TF[as.character(doubIDs$V2) == 'doublet'] <- TRUE
colnames(doubIDs)[1:2] <- c('UMI', 'TYPE')

require(Matrix)
require(ggplot2)

expr <- readMM('jamboree/doublet-datasets/dataset2/Ye3_sparse_molecule_counts.mtx')
idx <- as.character(read.csv('jamboree/doublet-datasets/dataset2/Ye3_barcode_id.csv', header = FALSE)$V2)
genes <- as.character(read.csv('jamboree/doublet-datasets/dataset2/Ye3_gene_id.csv', header = FALSE)$V2)
cellIdx <- is.element(idx, as.character(doubIDs$UMI))

colnames(expr) <- genes
rownames(expr) <- idx

mat <- as.matrix(expr[cellIdx, ])
mat <- mat[, apply(mat, 2, function(x){ !all(x == 0)} )]
mat <- t(mat)
log2Mat <- mat
type <- as.character(doubIDs$TYPE)
names(type) <- doubIDs$UMI
mtype <- type[colnames(log2Mat)]
log2Mat <- log2Mat[rowMeans(log2Mat) > 1, ]

riboIdx <- c(grep(rownames(log2Mat), pattern = 'MT-'),
             grep(rownames(log2Mat), pattern = 'RP(S|L)'))
riboExpr <- log2Mat[riboIdx, ]

aidx <- !is.element(seq_len(nrow(riboExpr)), riboIdx)
ratio <- apply(riboMat, 2, var )/apply(log2Mat[aidx, ], 2, var ) 
ratioDf <- data.frame( Ratio = ratio, Class = mtype)
ggplot(ratioDf, aes(x = log2(Ratio) )) + 
  geom_density(aes(fill = Class), alpha = 0.8) + 
  ggtitle('Ratio of Variance of Ribo Genes to none-Ribo Genes')

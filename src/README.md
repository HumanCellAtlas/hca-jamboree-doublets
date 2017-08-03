
Classifier
==========

In order to characterize doublets, we tried to assess different features that separate doublets from singlets – either in gene expression and PCA space.
The features we extracted are:

1.	Total UMIs / cell
2.	Number of detected genes
3.	Standard deviation of the normalized transcript.
4.	Fraction of genes with single UMI.
5.	First 20 PCA eigen vectors.

We integrated the features into a random forest model on the training set.

For datasets without labels, we tried to generate a synthetic dataset, by choosing pairs of cells at random, and downsampling from their shared transcriptome.

Our prediction algorithm performed well on the labelled training set and test set, and on the simulated data. However, cross validation worked poorly - as training on the simulated data didn't succeed on the labelled set and vice versa (we should add numbers here).


Files
=====

- `classifier.r` Main classifying script
- `predictor-genes.r` Alternative approach identifying predictor genes that could be added as additional features


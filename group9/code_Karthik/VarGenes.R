# NEW CODE FOR ESTIMATING VARIABLE GENES
# ======================================
# Parameters:
#   count.data (data.frame): [rows=genes, columns=cells] of # UMIs/cell
#   diffCV.cutoff (numeric): parameter for difference in Coefficient of Variation between null dist & selected variable genes
#                   (Difference is in log space, so this amounts to a fold-change)
#   mean.min/mean.max (numeric): crop selectable genes according to their mean
#                   (Use to eliminate super low noisy genes and high end if the null fits really poorly at high end.)
#   main.use (character): title to display for the overall three-panel plot
# ======================================
# Returns:
#   (character vector)
#   List of variable genes
# ======================================

require(MASS)

new.meanCVfit <- function(count.data, diffCV.cutoff=3, mean.min=.005, mean.max=1000, main.use="", do.plot=T) {
  
  # Format for the plot
  if (do.plot) par(mfrow=c(1,3), oma=c(0,0,2,0))
  
  # Calculate empirical mean, var and CV
  mean_emp <- apply(count.data, 1, mean)
  var_emp <- apply(count.data, 1, var)
  cv_emp <- sqrt(var_emp) / mean_emp
  
  # Fit gamma distribution to 'size factors' to build negative binomial
  a <- colSums(count.data)
  size_factor <- a/mean(a)
  fit <- fitdistr(size_factor, "Gamma")
  if (do.plot) {
    print(fit)
    hist(size_factor, 50, probability=TRUE, xlab="UMIs per cell / mean UMIs per cell", main=paste0("Size Factors & Gamma Fit (a=", round(fit$estimate[1], 2), ")"))
    curve(dgamma(x, shape=fit$estimate[1], rate=fit$estimate[2]),from=0, to=quantile(size_factor, 0.999), add=TRUE, col="red")
  }
  
  # Create null negative binomial distribution
  #   Gamma distributions of individual genes are just scaled versions. If X ~ Gamma(a,b)
  #   then cX ~ Gamma(a, b/c)
  a_i <- rep(fit$estimate[1], length(mean_emp))
  names(a_i) <- names(mean_emp)
  b_i <- fit$estimate[2] / mean_emp
  names(b_i) <- names(mean_emp)
  mean_NB <- a_i / b_i
  var_NB <- a_i*(1+b_i) / (b_i^2)
  cv_NB <- sqrt(var_NB)/mean_NB
  
  # Calculate difference in genes' CV and null CV
  diffCV = log(cv_emp) - log(cv_NB)
  if (do.plot) {
    hist(diffCV, 150, xlab="log(CVgene / CVnull)", main="Diff CV")
    abline(v=diffCV.cutoff, lty=2, col='red')
  }
  pass.cutoff <- names(diffCV)[which(diffCV > diffCV.cutoff & (mean_emp > mean.min & mean_emp < mean.max))]
  
  # Plot variable gene selection
  if (do.plot) {
    plot(mean_emp,cv_emp,pch=16,cex=0.5,col="black",xlab="Mean Counts",ylab="CV (counts)", log="xy", main = "Selection of Variable Genes")
    curve(sqrt(1/x), add=TRUE, col="red", log="xy", lty=2, lwd=2)
    or = order(mean_NB)
    lines(mean_NB[or], cv_NB[or], col="magenta", lwd=2)
    points(mean_emp[pass.cutoff], cv_emp[pass.cutoff], pch=16, cex=0.5, col='blue')
    title(main.use, outer=T)
  }
  
  # Return list of variable genes
  return(pass.cutoff)
}
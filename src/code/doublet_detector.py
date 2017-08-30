from helper_functions import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import numpy.random as nprnd


def detect_doublets(E, doub_frac=3.0, exclude_abundant_genes=1.0, min_counts=3, min_cells=5, vscore_percentile = 75, pca_coords = [], num_pc=50, k=30, genes_use=[], include_doubs_in_pca=False, return_parents=False, pca_method='sparse'):
    '''
    Identifies likely doublets by finding cells that are highly similar to
    simulated doublets (combinations of random pairs of observed cells). The
    output is a "doublet score" for each observed cell. Note that these scores
    should NOT be interpreted as probabilities.

    After assigning scores, you should examine the histogram of doublet scores
    and overlay the scores 2-D/3-D visualization of the data
    (e.g., t-SNE or SPRING). When a reasonable number of doublets are present,
    I usually see a long right tail on the histogram and co-localization
    (clusters) of high-scoring cells in the t-SNE/SPRING plot.

    There are a few complementary strategies for using the doublet scores to
    remove likely doublets:

    1) If you have clustered your data, you could remove entire clusters that
       are primarily comprised of cells with high doublet scores.

    2) Set a threshold based on the histogram.

    3) Use the expected doublet rate (e.g., 5%) to inform your threshold. Since
       doublets resulting from the combination of two highly similar cells will
       not be detected, this is likely an upper bound on the number of
       detectable doublets.

    Using default settings, the following steps are run to assign doublet scores:
    Using a single-cell counts matrix as input,
    1. Cell-level normalization by total counts
    2. Find highly variable genes expressed above a minimum level
    3. Run PCA using highly variable genes
    4. Simulate doublets by averaging the PCA coordinates of random pairs of
       cells (termed "parents"). The average is weighted by the total counts of
       the parent cells.
    5. Build a k-nearest-neighbor graph using the PCA coordinates of observed
       and simulated cells (Euclidean distance is used by default).
    6. For each observed cell, calculate the doublet score as the fraction of
       neighbors that are simulated doublets. The score is adjusted to account
       for the number of simulated doublets.

    The important parameters are:
    - k (number of neighbors in the knn graph): as long as this isn't too low
      or way too high, the scores generally aren't very sensitive. A reasonable
      place to start is ~sqrt(number of cells)
    - num_pc (number of PCs to use when constructing the knn graph): The best
      value will depend on the complexity of the dataset, but again the scores
      shouldn't be super sensitive to this parameter. If you've used PCA for
      other analyses (e.g., clustering or t-SNE), a similar value should work
      well here.
    - doub_frac (number of doublets to simulate, as a fraction of the number
      of observed cells): Simulating more doublets will only improve the
      accuracy of the detector, especially for very complex datasets, since
      you'll more closely approximate the true distribution of doublets. Of
      course, the cost is a longer compute time. I usually start with a
      doub_frac of 3-5.


    INPUTS:
    - E: Counts matrix (2-D numpy array; rows=cells, columns=genes)

    - doub_frac: Number of doublets to simulate, as a fraction of the number
      of cells in E. In general, the higher the better, though it will slow
      things down.

    - k: Number of neighbors for KNN graph. This will automatically be scaled by
      doub_frac. To start, try setting k=int(sqrt(number of cells))

    - min_counts and min_cells: For filtering genes based on expression levels.
      To be included, genes must be expressed in at least min_counts copies in
      at least min_cells cells.

    - vscore_percentile: For filtering genes based on variability. V-score is a
      measure of above-Poisson noise. To be included, genes must have a V-score
      in the top vscore_percentile percentile.

    - num_pc: number of principal components for constructing knn graph

    - genes_use: pre-filtered list of gene names to use for PCA; if supplied,
      no additional gene filtering is performed

    - include_doubs_in_pca: use simulated doublets for PCA; if True, runs much
      slower and unclear if performance is improved


    OUTPUTS:
    - doub_score_obs: doublet scores of observed cells

    - doub_score: doublet score of observed and simulated cells

    - doub_labels: labels for the indices of doub_score. 0 for observed cells
      and 1 for simulated doublets

    '''

    if pca_coords == []:
        if include_doubs_in_pca:
            print('Simulating doublets')
            E, doub_labels, parent_ix = simulate_doublets_from_counts(E, doublet_frac = doub_frac)
    
    
        print('Total count normalizing')
        counts = np.sum(E, axis=1)
        E = tot_counts_norm(E, exclude_dominant_frac = exclude_abundant_genes)
    
        if len(genes_use) == 0:
            print('Finding highly variable genes')
    
            Vscores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b = get_vscores(E)
            gene_filter = ((np.sum(E[:,gene_ix]>=min_counts, axis=0) >= min_cells) & (Vscores>np.percentile(Vscores,vscore_percentile)))
            gene_filter = gene_ix[gene_filter]
        else:
            gene_filter = genes_use
    
    
        print('Using', len(gene_filter), 'genes for PCA')
        PCdat = get_PCA(E[:,gene_filter], numpc=num_pc, method=pca_method)
    
        if not include_doubs_in_pca:
            print('Simulating doublets')
            PCdat, doub_labels, parent_ix = simulate_doublets_from_pca(PCdat, counts, doublet_frac = doub_frac)
    else:
        PCdat = pca_coords
        print('Simulating doublets')
        PCdat, doub_labels, parent_ix = simulate_doublets_from_pca(PCdat, counts, doublet_frac = doub_frac)



    n_obs = np.sum(doub_labels == 0)
    n_sim = np.sum(doub_labels == 1)


    ########################

    k_detect = int(round(k * (1+n_sim/float(n_obs))))

    print('Running KNN classifier with k =', k_detect)
    model = KNeighborsClassifier(n_neighbors=k_detect, metric='euclidean')
    model.fit(PCdat, doub_labels)

    raw_score = model.predict_proba(PCdat)
    raw_score = raw_score[:, 1]
    n_doub_neigh = raw_score * k_detect
    n_sing_neigh = k_detect - n_doub_neigh
    doub_score = n_doub_neigh / (n_doub_neigh + n_sing_neigh * n_sim / float(n_obs))

    doub_score_obs = doub_score[doub_labels == 0]


    if return_parents:
        print('Aggregating parents of doublets')
        doub_neigh_parents = []
        neigh = model.kneighbors(PCdat, return_distance=False)
        for ii in range(n_obs):
            iineigh = neigh[ii,:]
            doubix = iineigh[doub_labels[iineigh] == 1]
            if len(doubix) > 0:
                doub_neigh_parents.append(parent_ix[doubix-n_obs,:])
            else:
                doub_neigh_parents.append([])
        print('Done')
        return doub_score_obs, doub_score, doub_labels, doub_neigh_parents


    print('Done')
    return doub_score_obs, doub_score, doub_labels



def simulate_doublets_from_counts(E, doublet_frac = 1):
    '''
    Simulate doublets by summing the counts of random cell pairs.

    Inputs:
    E (npy matrix of size (num_cells, num_genes)): counts matrix, ideally without total-counts normalization.
    doublet_frac (float): number of doublets to simulate, as a fraction of the number of cells in E.
                          A total of int(doublet_frac * E[0]) doublets will be simulated.

    Returns:
    Edoub (npy matrix of size (num_cells+num_sim_doubs, num_genes)): counts matrix with the simulated doublet data appended to the original data matrix E.
    doub_labels ( )
    '''
    n_obs = E.shape[0]
    n_doub = int(n_obs * doublet_frac)


    pair_ix = nprnd.randint(0, n_obs, size=(n_doub, 2))

    Edoub = E[pair_ix[:, 0],:] + E[pair_ix[:, 1],:]


    Edoub = np.append(E, Edoub, axis=0)
    doub_labels = np.concatenate((np.zeros(n_obs), np.ones(n_doub)))

    return Edoub, doub_labels, pair_ix



def simulate_doublets_from_pca(PCdat, total_counts, doublet_frac = 1):
    '''
    Simulate doublets by averaging PCA coordinates of random cell pairs.
    Average is weighted by total counts of each parent cell.
    '''

    n_obs = PCdat.shape[0]
    n_doub = int(n_obs * doublet_frac)


    pair_ix = nprnd.randint(0, n_obs, size=(n_doub, 2))

    pair_tots = np.hstack((total_counts[pair_ix[:, 0]][:,None], total_counts[pair_ix[:, 1]][:,None]))
    pair_tots = np.array(pair_tots, dtype=float)
    pair_fracs = pair_tots / np.sum(pair_tots, axis=1)[:,None]

    PCdoub = PCdat[pair_ix[:, 0],:] * pair_fracs[:, 0][:,None] + PCdat[pair_ix[:, 1],:] * pair_fracs[:, 1][:,None]


    PCdoub = np.append(PCdat, PCdoub, axis=0)
    doub_labels = np.concatenate((np.zeros(n_obs), np.ones(n_doub)))

    return PCdoub, doub_labels, pair_ix

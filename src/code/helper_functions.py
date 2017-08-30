import os
import numpy as np
import scipy
import scipy.stats
import sklearn.cluster
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csc_matrix
import scipy.io
import json
import matplotlib.pyplot as plt


######### LOADING DATA
def text_to_expr(fname,delim='\t',start_row=0,start_column=0,update=0,data_type='int'):
    '''
     Load a text counts matrix from a text file.
     Can be gzipped (automatically detected).
     data_type should be "int" or "float"
    '''
    if fname.endswith('.gz'):
        tmpsuffix = str(np.random.randint(1e9))
        os.system('gunzip -c "' + fname + '" > tmp' + tmpsuffix)
        f = open('tmp' + tmpsuffix)
    else:
        f = open(fname)

    expr = []
    ct = 0
    for l in f:
        if ct>start_row-1:
            l = l.strip('\n').split(delim)
            if data_type == 'int':
                expr += [[int(x) for x in l[start_column:]]]
            elif data_type == 'float':
                expr += [[int(x) for x in l[start_column:]]]
            else:
                print('Unrecognized data type. Must be "int" or "float".')
                return

        ct += 1
        if update > 0:
            if ct % update == 0:
                print(ct)

    f.close()

    if fname.endswith('.gz'):
        os.system('rm tmp' + tmpsuffix)

    return expr

def read_npy_gzip(fname):
    '''
    Load .npy.gz counts matrix.
    '''
    tmpsuffix = str(np.random.randint(1e9))
    os.system('gunzip -c "' + fname + '" > tmp' + tmpsuffix)
    np_mat = np.load('tmp' + tmpsuffix)
    os.system('rm tmp' + tmpsuffix)
    return np_mat

def load_genes(fname):
    f = open(fname)
    return [l.strip('\n') for l in f]


def load_counts(prefix, save_as_npy = True):
    if os.path.isfile(prefix + '.counts.npy.gz'):
        print('loading from npy file')
        dat = read_npy_gzip(prefix + '.counts.npy.gz')
    else:
        print('loading from tsv file')
        dat = np.array(text_to_expr(prefix + '.counts.tsv.gz','\t',1,1))
        if save_as_npy:
            np.save(prefix + '.counts.npy', dat)
            os.system('gzip "' + prefix + '.counts.npy"')
    return dat


def load_mtx(fname):
    dat = scipy.io.mmread(fname)
    return dat.toarray()

########## GENE FILTERING

def runningquantile(x, y, p, nBins):
    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]


    dx = (x[-1] - x[0]) / nBins
    xOut = np.linspace(x[0]+dx/2, x[-1]-dx/2, nBins)

    yOut = np.zeros(xOut.shape)

    for i in range(len(xOut)):
        ind = np.nonzero((x >= xOut[i]-dx/2) & (x < xOut[i]+dx/2))[0]
        if len(ind) > 0:
            yOut[i] = np.percentile(y[ind], p)
        else:
            if i > 0:
                yOut[i] = yOut[i-1]
            else:
                yOut[i] = np.nan

    return xOut, yOut


def get_vscores(E, min_mean=0, nBins=50, fit_percentile=0.1, error_wt=1):
    mu_gene = np.mean(E, axis=0)
    gene_ix = np.nonzero(mu_gene > min_mean)[0]
    mu_gene = mu_gene[gene_ix]
    FF_gene = np.var(E[:,gene_ix], axis=0) / mu_gene

    data_x = np.log(mu_gene)
    data_y = np.log(FF_gene / mu_gene)

    x, y = runningquantile(data_x, data_y, fit_percentile, nBins)
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    gLog = lambda input: np.log(input[1] * np.exp(-input[0]) + input[2])
    h,b = np.histogram(np.log(FF_gene[mu_gene>0]), bins=200)
    b = b[:-1] + np.diff(b)/2
    max_ix = np.argmax(h)
    c = np.max((np.exp(b[max_ix]), 1))
    errFun = lambda b2: np.sum(abs(gLog([x,c,b2])-y) ** error_wt)
    b0 = 0.1
    b = scipy.optimize.fmin(func = errFun, x0=[b0], disp=False)
    a = c / (1+b) - 1


    v_scores = FF_gene / ((1+a)*(1+b) + b * mu_gene);
    CV_eff = np.sqrt((1+a)*(1+b) - 1);
    CV_input = np.sqrt(b);

    return v_scores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b


###### Clustering stuff

def get_hierch_order(hm, dist_metric='euclidean', linkage_method='ward'):
    from scipy.spatial.distance import pdist
    from fastcluster import linkage


    np.random.seed(0)
    D = pdist(hm, dist_metric)
    Z = linkage(D, linkage_method)
    n = len(Z) + 1
    cache = dict()
    for k in range(len(Z)):
        c1, c2 = int(Z[k][0]), int(Z[k][1])
        c1 = [c1] if c1 < n else cache.pop(c1)
        c2 = [c2] if c2 < n else cache.pop(c2)
        cache[n+k] = c1 + c2
    o = np.array(cache[2*len(Z)])

    return o

def spec_clust(A, k):
    spec = sklearn.cluster.SpectralClustering(n_clusters=k, affinity = 'precomputed',assign_labels='discretize')
    return spec.fit_predict(A)

####################    Faster SPRING

def tot_counts_norm(E, exclude_dominant_frac = 1):
    if exclude_dominant_frac == 1:
        tots_use = np.sum(E, axis = 1)
    else:
        tots = np.sum(E, axis = 1)
        included = np.all((E / tots[:,None]) < exclude_dominant_frac, axis = 0)
        tots_use = np.sum(E[:,included], axis = 1)
        print('Excluded', np.sum(~included), 'genes from normalization')

    w = np.mean(tots_use)/tots_use
    Enorm = E * w[:,None]

    return Enorm

def gene_stats(E):
    m = np.mean(E,axis=0)
    ix = m > 0
    m = m[ix]

    ff = np.var(E[:,ix],axis=0) / m
    gene_ix = np.nonzero(ix)[0]

    return m, ff, gene_ix

def get_PCA(E, base_ix=[], numpc=50, method='sparse', normalize=True):
    if len(base_ix) == 0:
        base_ix = np.arange(E.shape[0])

    if method == 'sparse':
        if normalize:
            zstd = np.std(E[base_ix,:],axis=0)
            Z = csc_matrix(E / zstd)
        else:
            Z = E

        pca = TruncatedSVD(n_components=numpc)
        pca.fit(Z[base_ix,:])
        return pca.transform(Z)

    else:
        if normalize:
            zmean = np.mean(E[base_ix,:],axis=0)
            zstd = np.std(E[base_ix,:],axis=0)
            Z = (E - zmean) / zstd
        else:
            Z = E
        pca = PCA(n_components=numpc)
        pca.fit(Z[base_ix,:])
        return pca.transform(Z)



def get_knn_graph(X, k=5, dist_metric='euclidean'):
    nbrs = NearestNeighbors(n_neighbors=k, metric=dist_metric).fit(X)
    knn = nbrs.kneighbors(return_distance=False)
    links = set([])

    A = np.zeros((X.shape[0], X.shape[0]))
    for i in range(knn.shape[0]):
        for j in knn[i,:]:
            links.add(tuple(sorted((i,j))))
            A[i,j] = 1
            A[j,i] = 1


    return links, A, knn





#========================================================================================#
def get_distance_matrix(M):
	'''
	##############################################
	Input
		M = Data matrix. Rows are datapoints (e.g. cell) and columns are features (e.g. genes)

	Output (D)
		D = All Pairwise euclidian distances between points in M
	##############################################
	'''
	D = np.zeros((M.shape[0],M.shape[0]))
	for i in range(M.shape[0]):
		Mtiled = np.tile(M[i,:][None,:],(M.shape[0],1))
		D[i,:] = np.sqrt(np.sum((Mtiled - M)**2, axis=1))
	return D

#========================================================================================#
def filter_cells(E, min_reads):
	'''
	##############################################
	Filter out cells with total UMI count < min_reads

	Input
		E         = Expression matrix. Rows correspond to cells and columns to genes
		min_reads = Minimum number of reads required for a cell to survive filtering

	Output  (Efiltered, cell_filter)
		Efiltered   = Filtered expression matrix
		cell_filter = Boolean mask that reports filtering. True means that the cell is
		              kept; False means the cells is removed
	##############################################
	'''
	total_counts = np.sum(E,axis=1)
	cell_filter = total_counts >= min_reads
	if np.sum(cell_filter) == 0:
		return None, cell_filter
	else: return E[cell_filter,:],cell_filter


def get_knn_edges(dmat, k, map_to_base_only, base_ix):
	'''
	##############################################
	Calculate knn-graph edges from a distance matrix.

	Input
		dmat = Square distance matrix. (dmat)_ij = the distance between i and k
		k    = Number of edges to assign each node (i.e. k in the knn-graph)

	Output (edge_list)
		edge_list = A list of unique undirected edges in the knn graph. Each edge comes in
		            the form of a tuple (i,j) representing an edge between i and j.
	##############################################
	'''
	edge_dict = {}
	for i in range(dmat.shape[0]):
		if map_to_base_only:
			if i in base_ix:
				sorted_nodes = base_ix[np.argsort(dmat[i,base_ix])[1:k+1]]
			else:
				sorted_nodes = base_ix[np.argsort(dmat[i,base_ix])[:k]]
		else:
			sorted_nodes = np.argsort(dmat[i,:])[1:k+1]
		for j in sorted_nodes:
			ii,jj = tuple(sorted([i,j]))
			edge_dict[(ii,jj)] = dmat[i,j]

	return edge_dict.keys()


def save_spring_dir(E,D,k,gene_list,project_directory, custom_colors={},cell_groupings={}, use_genes=[], map_to_base_only=False, base_ix=[]):
	'''
	##############################################
	Builds a SPRING project directory and transforms data into SPRING-readable formats

	Input (Required)
		E                  = (numpy array) matrix of gene expression. Rows correspond to
		                     celles and columns correspond to genes.
		D                  = (numpy array) distance matrix for construction of knn graph.
		                     Any distance matrix can be used as long as higher values
		                     correspond to greater distances.
		k                  = Number of edges assigned to each node in knn graph
		gene_list          = An ordered list of gene names with length length E.shape[1]
		project_directory  = Path to a directory where SPRING readable files will be
							 written. The directory does not have to exist before running
							 this function.

	Input (Optional)
		cell_groupings     = Dictionary with one key-value pair for each cell grouping.
							 The key is the name of the grouping (e.g. "SampleID") and
							 the value is a list of labels (e.g. ["sample1","sample2"...])
							 If there are N cells total (i.e. E.shape[0] == N), then the
							 list of labels should have N entries.
		custom_colors      = Dictionary with one key-value pair for each custom color.
							 The key is the name of the color track and the value is a
							 list of scalar values (i.e. color intensities). If there are
							 N cells total (i.e. E.shape[0] == N), then the list of labels
							 should have N entries.
	##############################################
	'''
	os.system('mkdir '+project_directory)
	if not project_directory[-1] == '/': project_directory += '/'

	# Build graph
	# print 'Building graph'
	edges = get_knn_edges(D,k,map_to_base_only,base_ix)

	# save genesets
	#print 'Saving gene sets'
	custom_colors['Uniform'] = np.zeros(E.shape[0])
	write_color_tracks(custom_colors, project_directory+'color_data_gene_sets.csv')
	all = []

	# save gene colortracks
	#print 'Savng coloring tracks'
	os.system('mkdir '+project_directory+'gene_colors')
	II = len(gene_list) / 50 + 1
	for j in range(50):
		fname = project_directory+'/gene_colors/color_data_all_genes-'+repr(j)+'.csv'
		if len(use_genes) > 0: all_gene_colors = {g : E[:,i+II*j] for i,g in enumerate(gene_list[II*j:II*(j+1)]) if g in use_genes}
		else: all_gene_colors = {g : E[:,i+II*j] for i,g in enumerate(gene_list[II*j:II*(j+1)]) if np.mean(E[:,i+II*j])>0.05}
		write_color_tracks(all_gene_colors, fname)
		all += all_gene_colors.keys()

	# Create and save a dictionary of color profiles to be used by the visualizer
	#print 'Color stats'
	color_stats = {}
	for i in range(E.shape[1]):
		mean = np.mean(E[:,i])
		std = np.std(E[:,i])
		max = np.max(E[:,i])
		centile = np.percentile(E[:,i],99.6)
		color_stats[gene_list[i]] = (mean,std,0,max,centile)
	for k,v in custom_colors.items():
		color_stats[k] = (0,1,np.min(v),np.max(v)+.01,np.percentile(v,99))
	json.dump(color_stats,open(project_directory+'/color_stats.json','w'),indent=4, sort_keys=True)


	# save cell labels
	#print 'Saving categorical color data'
	categorical_coloring_data = {}
	for k,labels in cell_groupings.items():
		label_colors = {l:frac_to_hex(float(i)/len(set(labels))) for i,l in enumerate(list(set(labels)))}
		categorical_coloring_data[k] = {'label_colors':label_colors, 'label_list':labels}
	json.dump(categorical_coloring_data,open(project_directory+'/categorical_coloring_data.json','w'),indent=4)


	#print 'Writing graph'
	nodes = [{'name':i,'number':i} for i in range(E.shape[0])]
	edges = [{'source':i, 'target':j, 'distance':0} for i,j in edges]
	out = {'nodes':nodes,'links':edges}
	open(project_directory+'graph_data.json','w').write(json.dumps(out,indent=4, separators=(',', ': ')))

#========================================================================================#
def row_sum_normalize(A):
	print(A.shape)
	d = np.sum(A,axis=1)
	A = A / np.tile(d[:,None],(1,A.shape[1]))
	return A

#========================================================================================#
def write_graph(n_nodes, edges,path):
	nodes = [{'name':i,'number':i} for i in range(n_nodes)]
	edges = [{'source':i, 'target':j, 'distance':0} for i,j in edges]
	out = {'nodes':nodes,'links':edges}
	open(path+'/graph_data.json','w').write(json.dumps(out,indent=4, separators=(',', ': ')))

#========================================================================================#
def write_color_tracks(ctracks, fname):
	out = []
	for name,score in ctracks.items():
		line = ','.join([name]+[repr(round(x,1)) for x in score])
		out += [line]
	out = sorted(out,key=lambda x: x.split(',')[0])
	open(fname,'w').write('\n'.join(out))

#========================================================================================#
def frac_to_hex(frac):
	rgb = tuple(np.array(np.array(plt.cm.jet(frac)[:3])*255,dtype=int))
	return '#%02x%02x%02x' % rgb


#========================================================================================#
def run_all_spring(E, gene_list, sample_name, save_dir = './', base_ix = [], normalize = True,
                   exclude_dominant_frac = 1.0, min_counts = 3, min_cells = 5, min_vscore_pctl = 75,
                   show_vscore_plot = False, exclude_gene_names = [],
                   num_pc = 50, pca_method = 'sparse', pca_norm = True,
                   k_neigh = 4, cell_groupings = {}, run_force = False, output_spring = True):
    if len(base_ix) == 0:
        base_ix = np.arange(E.shape[0])


    # total counts normalize

    tot_counts_final = np.sum(E, axis=1)
    if normalize:
        print('Normalizing')
        E = tot_counts_norm(E, exclude_dominant_frac = exclude_dominant_frac)

    # Get gene stats (above Poisson noise, i.e. V-scores)
    print('Filtering genes')
    Vscores, CV_eff, CV_input, gene_ix, mu_gene, FF_gene, a, b = get_vscores(E[base_ix, :])

    # # Remove user-excluded genes from consideration
    # if len(exclude_gene_names) > 0:
    #     keep_ix = np.array([ii for ii,gix in enumerate(gene_ix) if gene_list[gix] not in exclude_gene_names])
    #     print 'Excluded', len(gene_ix)-len(keep_ix), 'genes'
    #     gene_ix = gene_ix[keep_ix]
    #     Vscores = Vscores[keep_ix]
    #     mu_gene = mu_gene[keep_ix]
    #     FF_gene = FF_gene[keep_ix]

    # Filter genes: minimum V-score percentile and at least min_counts in at least min_cells
    min_log_vscore = np.percentile(np.log(Vscores), min_vscore_pctl)

    ix = ((np.sum(E[:,gene_ix] >= min_counts,axis=0) >= min_cells) & (np.log(Vscores) >= min_log_vscore))
    gene_filter = gene_ix[ix]
    print('Using', len(gene_filter), 'genes')

    # Remove user-excluded genes from consideration
    if len(exclude_gene_names) > 0:
        keep_ix = np.array([ii for ii,gix in enumerate(gene_filter) if gene_list[gix] not in exclude_gene_names])
        print('Excluded', len(gene_filter)-len(keep_ix), 'genes')
        gene_filter = gene_filter[keep_ix]

    if show_vscore_plot:
        x_min = 0.5*np.min(mu_gene)
        x_max = 2*np.max(mu_gene)
        xTh = x_min * np.exp(np.log(x_max/x_min)*np.linspace(0,1,100))
        yTh = (1 + a)*(1+b) + b * xTh
        plt.figure(figsize=(8, 6))
        plt.scatter(np.log(mu_gene), np.log(FF_gene), c = [.8,.8,.8]);
        plt.scatter(np.log(mu_gene)[ix], np.log(FF_gene)[ix], c = [0,0,0]);
        plt.plot(np.log(xTh),np.log(yTh));
        plt.title(sample_name)
        plt.xlabel('log(mean)');
        plt.ylabel('log(FF)');


    # RUN PCA
    # if method == 'sparse': normalizes by stdev
    # if method == anything else: z-score normalizes
    print('Running PCA')
    Epca = get_PCA(E[:,gene_filter], base_ix, numpc=num_pc, method=pca_method, normalize = pca_norm)

    if output_spring:
        # Calculate Euclidean distances in the PC space (will be used to build knn graph)
        print('Getting distance matrix')
        D = get_distance_matrix(Epca)

        # Build KNN graph and output SPRING format files
        save_path = save_dir + sample_name

        print('Saving SPRING files to ' + save_path)
        custom_colors = {'Total Counts': tot_counts_final}

        if len(cell_groupings) > 0:
            save_spring_dir(E, D, k_neigh, gene_list, save_path,
                            custom_colors = custom_colors,
                            cell_groupings = cell_groupings)
        else:
            save_spring_dir(E, D, k_neigh, gene_list, save_path,
                            custom_colors = custom_colors)

    links, A, _ = get_knn_graph(Epca, k=k_neigh)
    if run_force:
        print('Running FORCE')
        # Create random starting positions.
        starting_positions = np.random.random((Epca.shape[0], 2)) * 500
        force_graph = force.Force(starting_positions, links,
                                 bounds=10**5,  gravity = 0.01)
        tick = 0
        max_tick = 100
        while tick < max_tick:
            force_graph.fast_tick()
            if tick % 10 == 0:
                print('%i / %i' %(tick, max_tick))

            tick += 1
        coords = force_graph.current_positions

        print('Done!')
        return  E, Epca, A, gene_filter, coords

    print('Done!')
    return  E, Epca, A, gene_filter


#========================================================================================#

def gene_plot(x, y, E, gene_list, gene_name, col_range=(0,100), order_points=False, x_buffer=0, y_buffer=0,
        fig_size=(5,5), point_size=15, colormap='Reds', bg_color=[1,1,1], ax=''):
    '''
    Plot gene expression values on a scatter plot.

    Input
        x (1-D numpy float array, length=n_cells): x coordinates for scatter plot
        y (1-D numpy float array, length=n_cells): y coordinates for scatter plot
        E (2-D numpy float matrix, shape=(n_cells, n_genes)): gene expression counts matrix
        gene_list (list of strings, length=n_cells): full list of gene names
        gene_name (string): name of gene to visualize
        col_range (float tuple, length=2): (color_floor, color_ceiling) percentiles
        order_points (boolean): if True, plot points with higher color values on top of points with lower values
        x_buffer (float): white space to add to x limits
        y_buffer (float): white space to add to y limits
        fig_size (float tuple, length=2): size of figure
        point_size (float): size of scatter plot points
        colormap: color scheme for coloring the scatter plot
        bg_color (RGB/HEX/color name): background color

    Output
        fig: figure handle
        ax: axis handle
        pl: scatter plot handle
    '''
    # get gene index and color data
    gene_ix = gene_list.index(gene_name)
    colordat = E[:,gene_ix]

    # get min and max color values
    cmin = np.percentile(colordat, col_range[0])
    cmax = np.percentile(colordat, col_range[1])
    if cmax == 0:
        cmax = max(colordat)

    # order points by intensity, if desired
    if order_points:
        plot_ord = np.argsort(colordat)
    else:
        plot_ord = np.arange(len(colordat))

    # make the plot
    return_all = False
    if ax == '':
        return_all = True
        fig, ax = plt.subplots(1, 1, figsize = fig_size)

    pl = ax.scatter(x[plot_ord], y[plot_ord], c=colordat[plot_ord], s=point_size, edgecolor='none',
                    cmap=colormap, vmin=cmin, vmax=cmax)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim((min(x) - x_buffer, max(x) + x_buffer))
    ax.set_ylim((min(y) - y_buffer, max(y) + y_buffer))
    ax.patch.set_color(bg_color)

    if return_all:
        return fig, ax, pl
    else:
        return pl


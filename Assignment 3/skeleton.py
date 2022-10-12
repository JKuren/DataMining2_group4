from cmath import log, nan
import numpy as np
from sklearn import datasets
from sklearn import cluster
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle
import math

# ----------------------------------functions for unsupervised evaluation----------------------------------
def distance(points, point): # compute the distance from a point to other/another points
    if len(points.shape) == 1:
        return np.sqrt(sum((points - point) ** 2))
    else:
        return np.sqrt(np.sum((points - point) ** 2, axis = 1))  #Ouput: float or array of float

# TODO 1: compute SSE
#  Input: data - the 2d locations of data points
#         r_clusters - the clusters obtained by some clustering algorithm
#         r_centers - the centers of clusters
#  HINT: (1) for frequently used function, like computing the distance between a point and other point(s), it is better to write it as an independent sub-function for multiple calling
#        (2) utilize the nd-array's built-in functions/utilities to make your code concise and efficient
#        (3) OVERALL COMMENT: r_cluster and r_labels contain the same information but in different form, you can use r_labels to replace r_cluster as the input, if you prefer it
def compute_SSE(data, r_clusters, r_centers):
    SSE = 0
    for index,cluster in enumerate(r_clusters):
        for i in cluster:
            SSE += (distance(data[i], r_centers[index])**2)
    return SSE

def compute_SSB(data, r_clusters, r_centers): # computer SSB
    center_1 = np.mean(data, axis=0)
    SSB = 0
    for k in range(len(r_clusters)):
        SSB += len(r_clusters[k]) * np.sum(distance(r_centers[k, :] ,center_1) ** 2)
    return SSB

# TODO 2: compute the average silhouette coefficient (for all points)
def compute_avg_silhouette_coefficient(data, r_clusters):
    sc_list = []
    a_list = []
    b_temp = []
    b_list = []
    a=0
    b=0
    for index,cluster in enumerate(r_clusters):

        for i in cluster:
            for j in cluster:
                a += distance(data[i], data[j])
            a = a/(len(cluster)-1)
            a_list.append(a)
            a=0

        for i in cluster:
            for ind, b_cluster in enumerate(r_clusters):
                if ind == index:
                    continue
                else:
                    for j in b_cluster:
                        b += distance(data[i], data[j])
                    b = b/len(cluster)
                    b_temp.append(b)
                    b = 0
            b_list.append(min(b_temp))
            b_temp = []

    for a,b in zip(a_list, b_list):
        sc = (b - a) / max(a,b)
        sc_list.append(sc)
        
    avg_sc = sum(sc_list)/len(sc_list)
    return avg_sc

# TODO 3: compute the proximity matrix
#   HINT: use the similarity matrix, instead of dissimilarity
#         normalization is needed, i.e., the maximum value of similarity is 1
def compute_proximity_matrix(data):
    proximity = []
    for point in data:
            row = []
            for other in data:
                row.append(abs(1-(distance(point,other)/len(data))))
            proximity.append(row)
    proximity = np.array(proximity)
    return proximity

def compute_clustering_matrix(r_labels): # compute the clustering matrix
    N = data.shape[0]
    clustering = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if r_labels[i] == r_labels[j]:  # check whether two points belong to the same cluster
                clustering[i,j] = 1
    return clustering

# TODO 4: compute the correlation between two matrices A and B
def correlation(A, B):
    cov = np.mean(np.multiply((A-np.mean(A)),(B-np.mean(B))))
    correlation_val = cov / (np.std(A)*np.std(B))
    return correlation_val


# ----------------------------------functions for supervised evaluation----------------------------------
# ---------- classification-oriented evaluation ----------------
# TODO 5: compute the precision and recall
#  Input: labels - the known labels (ground truth)
#         r_clusters - the clusters obtained by some clustering algorithm

def compute_precision_and_recall(labels, r_clusters):
    no_classes = len(set(labels))
    no_clusters = len(r_clusters)
    no_obj_classes = [0]*no_classes
    for i in labels:
        no_obj_classes[i] += 1
    precision = np.zeros((no_classes,no_clusters))
    recall = np.zeros((no_classes,no_clusters))
    for index,cluster in enumerate(r_clusters):
        mj = 0
        row = [0]*no_classes
        for i in cluster:
            row[labels[i]] += 1
            mj += 1
        precision_row = []
        precision_row[:] = [mij / mj for mij in row]
        precision[index] = precision_row
        recall_row = np.divide(row,no_obj_classes)
        recall[index] = recall_row
    return precision, recall

def compute_purity(precision): # compute the purity
    return np.sum(np.max(precision, axis=1))

# TODO 6: compute the entropy
#  HINT: the 0 element(s) in precision will not be considered for computing entropy
def compute_entropy(precision, r_clusters):
    m_i = [0]*len(r_clusters)
    m = len(data)
    #calc m_i
    for index,cluster in enumerate(r_clusters):
        m_i[index] = len(cluster)

    e_i = [0]*len(r_clusters)

    #calc e_i
    for index, cluster in enumerate(r_clusters):
        for i in precision:
            for j in i:
                if j == 0:
                    continue
                else:
                    e_i[index] += j*math.log2(j)
            e_i[index] = -e_i[index]
    
    entropy = 0
    for index, cluster in enumerate(r_clusters):
        entropy += (m_i[index] / m)*e_i[index]

    return entropy

#---------- similarity-oriented evaluation ----------------
# TODO 7: compute rand_statistic and Jaccard_coeff
#  Input: labels - the known labels (ground truth)
#         r_lables - the clustering result by some algorithm, to be evaluated
def compute_binary_similarity(labels, r_labels):
    f_dict = {}
    for lab_i, rlab_i in list(zip(compute_clustering_matrix(labels), compute_clustering_matrix(r_labels))):
        for lab, r_lab in list(zip(lab_i, rlab_i)):
            f_add = f_dict.get(f'f{lab}{r_lab}',0) + 1 
            f_dict[f'f{lab}{r_lab}']  =  f_add

    f_00 = f_dict.get('f0.00.0',0)
    f_11 = f_dict.get('f1.01.0',0)
    f_10 = f_dict.get('f1.00.0',0)
    f_01 = f_dict.get('f0.01.0',0)

    rand_statistic = (f_00+f_11) / (f_00+f_01+f_10+f_11)
    Jaccard_coeff = f_11/ (f_01+f_10+f_11)

    return rand_statistic, Jaccard_coeff


# ------------------------------------------ other functions --------------------------------------
def generate_sparse_data(n): #random sparse data points. Input: the number of points
    data = []
    for i in range(n):
        data.append([np.random.randn(), np.random.randn()])  # the location of one point
    data = np.array(data)
    return data

def labels_to_clusters(labels): # derive the clusters from labels. Example: if the labels are [0,0,1,1], then the clusters will be [[0,1], [2,3]] where the number is the index of data points
    cluster_num = np.max(labels) + 1
    clusters = []
    for k in range(cluster_num):
        clusters.append([])
    for i in range(len(labels)):
        clusters[labels[i]].append(i)
    return clusters

def plot_data(data): # plot the data points
    fig = plt.figure(figsize=(7, 4.5))
    ax  = fig.add_subplot(1, 1, 1)
    ax.scatter(data[:,0], data[:,1], c='black')
    plt.xlabel('x')
    plt.ylabel('y')

# TODO 10 (optional): personalized your plot style
def plot_clusters(data, r_clusters): # plot the data points with different colors for different clusters
    fig = plt.figure(figsize=(7, 4.5))
    ax  = fig.add_subplot(1, 1, 1)
    for k in range(len(r_clusters)):
        ax.scatter(data[r_clusters[k],0], data[r_clusters[k],1])
    plt.xlabel('x')
    plt.ylabel('y')

def plot_line(values, x_ticks, x_label, y_label): # plot the variation of values
    fig = plt.figure(figsize=(7, 4.5))
    ax  = fig.add_subplot(1, 1, 1)
    ax.plot(values)
    plt.xticks(range(len(values)), x_ticks)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

# TODO 11 (optional): personalized your plot style
def plot_matrix(matrix, axis_name): # plot a matrix
    n = matrix.shape[0]
    fig = plt.figure(figsize=(7, 4.5))
    ax  = fig.add_subplot(1, 1, 1)
    im = ax.imshow(matrix, cmap="Blues",  vmin=0)   # you can change cmap to use different colors
    tick_locator = ticker.MaxNLocator()
    cb1 = plt.colorbar(im)
    cb1.locator = tick_locator
    cb1.update_ticks()
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.xlabel(axis_name)
    plt.ylabel(axis_name)



#-------------------------------------------- The main process ------------------------------------------
# TODO 8: generate two datasets, one is in random sparse structure while another distributed as specific overall shapes
#  HINT: (1) use the tiny dataset to check the correctness of your functions above. You may also need to build other tiny datasets for further check
#        (2) for random sparse data points: call function generate_sparse_data(n), and you can choose any parameter "n" (n >= 50) as you like
#        (3) for data points distributed as different shapes: use functions in package sklearn.datasets, like datasets.make_circles(...), datasets.make_moons(...), datasets.make_blobs(...), etc. Choose parameters as you like (at least 50 points and 3 clusters)
#------------------------------------- Step 1: generate different datasets -------------------------------------
# ----- 1.1 tiny dataset (only for debugging) --------
# data = np.array([[0,1],[0,2], [0,4], [0,5]])  # the location of 2d data points
# labels = np.array([0,0,1,1])  # the known classification
# n_list = [2] # the number of clusters
# ----- 1.2 random sparse data points (only for unsupervised evaluation) --------
# data = generate_sparse_data(100)
# n_list = [3,4,5,6,7,8,9,10,11,12,13,14,15] # for testing the best number of clusters in unsupervised evaluation
# SSEs, ASCs  = [], [] #ditto
# ----- 1.3 data points distributed as different shapes (for both supervised and unsupervised evaluations) --------
data, labels = datasets.make_blobs(n_samples=100, n_features=6, centers=6, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)
n_list = [3,4,5] # for testing the best number of clusters in unsupervised evaluation
n_list = [6] # the fixed number of clusters in supervised evaluation

plot_data(data)
for n_cluster in n_list:
    # ------------------------------------- Step 2: Clustering ------------------------------------------
    # TODO 9: replace the kmeans algorithm by the DBSCAN algorithm (optional: you can also try other algorithms)
    #  HINT: call function cluster.DBSCAN(...)

    # result = cluster.KMeans(n_clusters=n_cluster, random_state=0).fit(data) # call the kmeans algorithm
    # r_labels = result.labels_  # the cluster labels for points
    # r_clusters = labels_to_clusters(r_labels) # the clusters
    # r_centers = result.cluster_centers_  # the centers of clusters
    # print('labels:', r_labels)
    # print('clusters:', r_clusters)
    # print('cluster centers:\n', r_centers)
    # plot_clusters(data, r_clusters) # plot the data points
    # print('\n')

    result = cluster.DBSCAN(eps=n_cluster).fit(data) # call the DBSCAN algorithm
    r_labels = result.labels_  # the cluster labels for points
    r_clusters = labels_to_clusters(r_labels) # the clusters
    #r_centers = result.cluster_centers_  # the centers of clusters
    print('r_labels:', r_labels)
    print('clusters:', r_clusters)
    #print('cluster centers:\n', r_centers)
    plot_clusters(data, r_clusters) # plot the data points
    print('\n')

    #--------------------------------- Step 3: Unsupervised evaluation ---------------------------------
    #---------- 3.1 evaluation with a given number of clusters -----------------
    # print('Part 1: Unsupervised evaluation')
    # SSE = compute_SSE(data, r_clusters, r_centers)
    # # SSEs.append(SSE)
    # print('SSE:', np.round(SSE,2))
    # SSB = compute_SSB(data, r_clusters, r_centers)
    # print('SSB:', np.round(SSB,2))
    average_silhouette_coefficient = compute_avg_silhouette_coefficient(data, r_clusters)
    # ASCs.append(average_silhouette_coefficient)
    print('average silhouette coefficient:', np.round(average_silhouette_coefficient,2))
    proximity_matrix = compute_proximity_matrix(data)
    print('proximity matrix:\n', np.round(proximity_matrix,2))
    # plot_matrix(proximity_matrix, 'Points')
    clustering_matrix = compute_clustering_matrix(r_labels)
    print('clustering matrix:\n', np.round(clustering_matrix,2))
    # plot_matrix(clustering_matrix, 'Points')
    corr = correlation(proximity_matrix, proximity_matrix)
    print('corr:', np.round(corr,2))
    print('\n')

    #---------------------------------Step 4: Supervised evaluation -------------------------------
    #---------- 4.1 classification-oriented evaluation ----------------
    print('Part2: Supervised evaluation - classification-oriented')
    precision, recall = compute_precision_and_recall(labels, r_clusters)
    print('precision:\n', np.round(precision,2))
    print('recall:\n', np.round(recall,2))
    purity = compute_purity(precision)
    print('purity:', np.round(purity,2))
    entropy = compute_entropy(precision, r_clusters)
    print('entropy:', np.round(entropy,2))
    print('\n')

    # ------------ 4.2 similarity-oriented evaluation ----------------
    print('Part 3: Supervised evaluation - similarity-oriented')
    rand_statistic, Jaccard_coeff = compute_binary_similarity(labels, r_labels)
    print('rand_statistic:', np.round(rand_statistic,2))
    print('Jaccard_coeff:', np.round(Jaccard_coeff,2))
    print('\n')

#------- 3.2 select the best number of clusters for unsupervised clustering -----------
#plot_line(SSEs, [str(n) for n in n_list], 'number of clusters', 'SSE')
#plot_line(ASCs, [str(n) for n in n_list], 'number of clusters', 'Average silhouette coefficient')
plt.show()
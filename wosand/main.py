import time

from connection import *

import numpy as np
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import squareform as squareform
from scipy.stats import mode

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from core import get_library
from core import get_cosine_distance_matrix
from core import get_string_distance_matrix
from core import get_graph

from image_processing import modify_matrix
from clustering_evaluation import get_clusters_shannon_entropy
from clustering_evaluation import evaluate
from clustering_evaluation import elbow
from clustering_evaluation import concensus
from clustering_evaluation import within_clust_sim
from clustering_evaluation import between_clust_sim
from clustering_evaluation import paperElbow
from sklearn import metrics
from sklearn import linear_model
from sklearn.externals import joblib

from skimage.exposure import adjust_sigmoid
from skimage.exposure import adjust_log,adjust_gamma
import math

DEBUG = True
PRODUCTION = False
plots = False

import warnings
warnings.filterwarnings('ignore')


def main(name,subset = False):
    # FOCUS NAME
    focus_name = name
    print("\nFOCUS NAME: {0}".format(focus_name))
    
    # Cosine distance [True, False]
    all = {
        'title'         :   True, 
        'coauthors'     :   True,
        'institutions'  :   True,
        'journals'      :   False,
        'year'          :   False,  
        'subjects'      :   False,
        'keywords'      :   True, 
        'ref_authors'   :   False,
        'ref_journals'  :   True
    }
    # Graphs [True, False]
    graphs = {
        "coauthorship"  :   False,
        "references"    :   False,
        "subjects"      :   False,
        "keywords"      :   False
    }
    
    #get objects from data 
    library,catalog = get_library(focus_name, DEBUG,subset)
    
    #initialize matrices
    final_matrix = np.zeros((len(library),len(library)))
    mask_matrix = np.identity(len(library))

    graph_mat = np.identity(len(library))
    
    #get optimal matrix DISTANCES
    mat = 1 - get_string_distance_matrix(library,catalog,'author_id','exact_match')

    #get mask
    mask_matrix = get_string_distance_matrix(library,catalog,'author_name','syllab_jaccard')
    mask = np.array(mask_matrix)
    mask[mask>0] = 1

    #plots
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Mask with initials", fontdict={'fontsize': 18})
    im = ax.imshow(mask_matrix, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.colorbar(im)'''
    #return
    

    mat_title = 1 - get_cosine_distance_matrix(library,catalog,'title','cosine')
    mat_coauthors = 1- get_cosine_distance_matrix(library,catalog,'coauthors','cosine')
    mat_institutions = 1-get_cosine_distance_matrix(library,catalog,'institutions','cosine')
    mat_journals = 1-get_cosine_distance_matrix(library,catalog,'journals','cosine')
    mat_year = 1-get_cosine_distance_matrix(library,catalog,'year','cosine')
    mat_subjects = 1-get_cosine_distance_matrix(library,catalog,'subjects','cosine')
    mat_keywords = 1-get_cosine_distance_matrix(library,catalog,'keywords','cosine')
    mat_ref_authors = 1-get_cosine_distance_matrix(library,catalog,'ref_authors','cosine')
    mat_ref_journals = 1-get_cosine_distance_matrix(library,catalog,'ref_journals','cosine')
    mat_coauthorship = 1 - (get_graph(library, catalog, focus_name, 'coauthorship')*mask_matrix)
    fig = plt.figure()
    #("Cosine similarity matrices", fontdict={'fontsize': 18})
    ax = fig.add_subplot(251)
    plt.title("Title")
    ax.imshow(mat_title, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(252)
    ax.set_title("(Institutions")
    ax.imshow(mat_institutions, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(253)
    ax.set_title("Ref_journals")
    ax.imshow(mat_ref_journals, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(254)
    ax.set_title("Ref_author")
    ax.imshow(mat_coauthors, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(255)
    ax.set_title("Keyword")
    ax.imshow(mat_keywords, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(256)
    

    mat_title = 1 - get_cosine_distance_matrix(library,catalog,'title','cosine') * mask
    mat_coauthors = 1- get_cosine_distance_matrix(library,catalog,'coauthors','cosine') * mask
    mat_institutions = 1-get_cosine_distance_matrix(library,catalog,'institutions','cosine') * mask
    mat_journals = 1-get_cosine_distance_matrix(library,catalog,'journals','cosine') * mask
    mat_year = 1-get_cosine_distance_matrix(library,catalog,'year','cosine') * mask
    mat_subjects = 1-get_cosine_distance_matrix(library,catalog,'subjects','cosine') * mask
    mat_keywords = 1-get_cosine_distance_matrix(library,catalog,'keywords','cosine') * mask
    mat_ref_authors = 1-get_cosine_distance_matrix(library,catalog,'ref_authors','cosine') * mask
    mat_ref_journals = 1-get_cosine_distance_matrix(library,catalog,'ref_journals','cosine') * mask
    mat_coauthorship = 1 - (get_graph(library, catalog, focus_name, 'coauthorship')*mask_matrix) * mask
    
    ax.set_title("Title (2)") 
    ax.imshow(adjust_sigmoid(mat_title), cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(257)
    ax.set_title("Institution (2)")
    ax.imshow(adjust_sigmoid(mat_institutions), cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(258)
    ax.set_title("Ref_journals (2)")
    ax.imshow(adjust_sigmoid(mat_ref_journals), cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(259)
    ax.set_title("Coauthors (2)")
    ax.imshow(adjust_sigmoid(mat_coauthors), cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(250)
    ax.set_title("Keywords (2)")
    ax.imshow(adjust_sigmoid(mat_keywords), cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    mat_coauthorship = 1 - (get_graph(library, catalog, focus_name, 'coauthorship')*mask_matrix)
    mat_references = 1 - (get_graph(library, catalog, focus_name, 'references') * mask_matrix)
    '''
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_title("Coauthorship")
    ax.imshow(mat, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(122)
    ax.set_title("References")
    ax.imshow(1-mask_matrix, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    '''
    #return

    
    #get distance matrices aggregated by sum (counter increases each time) 
    counter = 0
    mat_temp = []
    mat_list = []
    for (k,v) in all.items():
        if v:
            mat_temp = get_cosine_distance_matrix(library,catalog,k,'cosine')* mask
            """plt.subplot(250+counter+2)
            plt.imshow(mat_temp, cmap=cm.coolwarm, interpolation='none', vmin=0, vmax=1)
            plt.title(k)
            plt.colorbar()"""
            mat_list.append(adjust_sigmoid(mat_temp))
            counter = counter + 1
            

    #get graph matrices aggregated by max (counter increases only of 1) 
    graph_mat = None
    for (k,v) in graphs.items():
        #mat_temp = get_graph(library, catalog, focus_name, k)
        #mat_list.append(mat_temp)
        #counter = counter + 1
        if v:
            if graph_mat is None:
                graph_mat = np.zeros((len(library), len(library)))
            mat_temp = get_graph(library, catalog, focus_name, k)
            graph_mat = adjust_sigmoid(np.maximum(mat_temp, graph_mat))

    if graph_mat is not None:
        counter = counter + 1
        #get final matrix, apply also mask to graph matrix
        mat_list.append(adjust_sigmoid(graph_mat * mask_matrix))
        """plt.figure()
        plt.imshow(graph_mat, cmap=cm.coolwarm, interpolation='none', vmin=0, vmax=1)
        plt.title("Graphs")
        plt.colorbar()"""
    mat_inst = mat_list[0]   
    #IMPORTANT BEFORE CLUSTERING
    #normalize final matrix and convert to DISTANCE matrix
    final_matrix = 1 - np.max(mat_list, axis=0)
    #correct negative values due to floating point precision to zero
    final_matrix[final_matrix<0] = 0
    #print(final_matrix)
    if final_matrix.size == 0 or final_matrix.size == 1:
        print("Only one paper found.")
        return 
    #else:
    #    #print("Number of papers: {0}".format(len(library)))
    #    #print("Number of authors: {0}".format(len(set([x.author_id for x in library]))))
        
    #statistics
    temp = final_matrix[final_matrix<1]
    mean = np.mean(temp[temp>0])
    median = np.median(temp[temp>0])
 
    if DEBUG:

        print("Min", np.min(final_matrix[final_matrix>0]))
        print("Max", np.max(final_matrix[final_matrix<1]))
        print("Mean", mean)
        print("Median", median)
        print("Variance", np.var(final_matrix[final_matrix<1]))

    '''
    c_mats = [
    modify_matrix(final_matrix,alpha=1,beta=mean),
    modify_matrix(final_matrix,alpha=100,beta=mean),
    modify_matrix(final_matrix,alpha=1,beta=median),
    modify_matrix(final_matrix,alpha=100,beta=median), 
    modify_matrix(final_matrix,alpha=1,beta=1),
    modify_matrix(final_matrix,alpha=100,beta=1),
    modify_matrix(final_matrix,alpha=1,beta=0.5),
    modify_matrix(final_matrix,alpha=100,beta=0.5)]
    
    fig = plt.figure()
    #("Cosine similarity matrices", fontdict={'fontsize': 18})
    ax = fig.add_subplot(251)
    plt.title("Optimal")
    ax.imshow(c_mats[0], cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(252)
    ax.set_title("Title")
    ax.imshow(c_mats[1], cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(253)
    ax.set_title("Coauthors")
    ax.imshow(c_mats[2], cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(254)
    ax.set_title("Institutions")
    ax.imshow(c_mats[3], cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(255)
    ax.set_title("Journals")
    ax.imshow(c_mats[4], cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(256)
    ax.set_title("Year") 
    ax.imshow(c_mats[5], cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(257)
    ax.set_title("Subjects")
    ax.imshow(c_mats[6], cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(258)
    ax.set_title("Keywords")
    ax.imshow(c_mats[7], cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(259)
    ax.set_title("Obtained matrix")
    ax.imshow(final_matrix, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(250)
    ax.set_title("Optimal")
    ax.imshow(mat, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    '''
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_title("Final")
    ax.imshow(final_matrix, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(122)
    ax.set_title("Contrast")
    ax.imshow(mat, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #return
    
    #save to file
    #np.savetxt("distance_matrix_{0}.csv".format(focus_name.split()[0]), final_matrix, fmt="%.2e", delimiter=",")
   
    #plot distance matrices
    
    '''
    if plots:
        
        plt.figure()
        plt.subplot(121)
        plt.title("Exp matrix")
        plt.imshow(adjust_gamma(final_matrix,gamma =.5), cmap=cm.GnBu, interpolation='none')
        plt.colorbar()
        plt.subplot(122)
        plt.title("Log matrix")
        plt.imshow(adjust_log(final_matrix), cmap=cm.GnBu, interpolation='none')
        plt.colorbar()
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(15,45,800, 500)
    '''
    #other plots for the report 
    

    '''
    fig = plt.figure()
    ax = fig.add_subplot(151)
    ax.set_title("Optimal", fontdict={'fontsize': 15})
    ax.imshow(mat, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(152)
    ax.set_title("Institutions", fontdict={'fontsize': 15})
    ax.imshow(mat_institutions, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(153)
    ax.set_title("Coauthorship", fontdict={'fontsize': 15})
    ax.imshow(mat_coauthorship, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(154)
    ax.set_title("References", fontdict={'fontsize': 15})
    ax.imshow(mat_references, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax = fig.add_subplot(155)
    ax.set_title("Aggregated", fontdict={'fontsize': 15})
    im = ax.imshow(final_matrix, cmap=cm.GnBu, interpolation='none', vmin=0, vmax=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #return
    '''
    
    #Compute optimal clustering
    ground_truth = {}
    for p1 in library:
        flag = 0
        if p1.author_id not in ground_truth:
            ground_truth[p1.author_id] = []
        ground_truth[p1.author_id].append(p1.unique_identifier)
    #print("Ground truth: ", str(list(ground_truth.values())))

    #Hierarchical clustering on final_mat
    #single linkage = minimum spanning tree
    #complete linkage, average distance, etc...
    #http://pages.cs.wisc.edu/~jerryzhu/cs769/clustering.pdf
    vector_distances = squareform(final_matrix, force='tovector', checks=False)
    #print("Mat shape: {0} Vector shape: {1}".format(final_matrix.shape, vector_distances.shape))
    
    linkage_matrix = hac.linkage(vector_distances, method='single', metric='euclidean') #single, average
    #print("Linkage matrix:\n", linkage_matrix)
    relevant_thresholds =  [0]+list(np.array(linkage_matrix[:,2]))
    
    overall_entropy_history = []
    coauthor_entropy_history = []
    country_entropy_history = []
    subject_entropy_history = []
    journal_entropy_history = []
    concensus_history = []
    within_history = []
    between_history = []
    variance_history = []
    
    truth_fmeasure = 0
    truth_ = 0
    
    previous_clusters = {}
    
    for x in relevant_thresholds:
        x+=0.0000000000001
        clusters_list = hac.fcluster(linkage_matrix, x, criterion='distance') #'maxclust'
        
        #mapping structure and evaluation structure
        clusters = {}
        for k,v in enumerate(clusters_list):     
            #print(library[k].author_name + " - " + str(library[k].author_id) + " - Cluster: " + str(clusters_list[k]))
            #print(k, v)
            if v not in clusters:
                clusters[v] = []
            #like assigning Id to clusters... library[k].unique_identifier would be better
            clusters[v].append(library[k].unique_identifier)
            
        new_cluster =  1
        previous_cluster1 = 1
        previous_cluster2 = 1
        
        # identifying which cluster was formed
        if len(previous_clusters)> 0:
            for (o_cluster,o_papers) in previous_clusters.items():
                for (n_cluster,n_papers) in clusters.items():
                    if o_papers[0] in n_papers[0] and len(o_papers)<len(n_papers):
                        new_cluster = n_cluster
                        previous_cluster1 = o_cluster 
        
        for paper in clusters[new_cluster]:
            for (o_cluster,o_papers) in previous_clusters.items():
                if o_cluster != previous_cluster1:
                    if paper in o_papers:
                        previous_cluster2 = o_cluster
                    
        '''
        print (clusters)
        print (previous_clusters)
        print ("new cluster:")
        print (new_cluster)
        print ("old clusters:")
        print (previous_cluster1)
        print (previous_cluster2)
        '''
        
        #num of clusters for current threshold
        num_clusters = len(clusters)
        
        #evaluation

        concensus_history.append(concensus(1 -final_matrix, catalog, clusters))
        within_history.append(within_clust_sim(1-final_matrix, catalog, clusters))
        between_history.append(between_clust_sim(1-final_matrix, catalog, clusters))
        #print("Within clusters: {0}".format(within_clust_sim(final_matrix, catalog, clusters)))
        precision, recall, f_measure = evaluate(clusters.values(), ground_truth.values())
        if f_measure >= truth_fmeasure:
            truth_fmeasure = f_measure
            truth_ = x
            truth_precision = precision
            truth_recall = recall
        """
        # entropies with respect to features, for each cluster:
        # - first split the feature of interest (e.g. coauthors) and put them in the "entropy_feature" list, keep repetitions 
        # - then count the ocurrences of each class (e.g. each coauthor)
        # - finally calculate the shannon entropy   
        # - sum up all the entropies 
        entropy_coauthors = {}
        entropy_subjects = {}
        entropy_countries = {}
        entropy_years = {}
        entropy_journals = {}
        for clust, author_uids in clusters.items():
        
            #entropy coauthors for the current cluster
            coauthors_in_clusters = []
            for uid in author_uids:
                for coauthor in library[catalog[uid]].coauthors.split(" "):
                    if coauthor.strip(): coauthors_in_clusters.append(coauthor)
            entropy_coauthors[clust] = get_clusters_shannon_entropy(coauthors_in_clusters)
            
            #entropy test on subjects
            subjects_in_clusters = []
            for uid in author_uids:
                for subject in library[catalog[uid]].subjects.split(" "):
                    if subject.strip(): subjects_in_clusters.append(subject)
            entropy_subjects[clust] = get_clusters_shannon_entropy(subjects_in_clusters)
            
            #entropy test on countries
            countries_in_clusters = []
            for uid in author_uids:
                for country in library[catalog[uid]].countries.split(" "):
                    if country.strip(): countries_in_clusters.append(country)
            entropy_countries[clust] = get_clusters_shannon_entropy(countries_in_clusters)
            
            #entropy on years
            years_in_clusters = []
            for uid in author_uids:
                for year in library[catalog[uid]].year.split(" "):
                    if year.strip(): years_in_clusters.append(year)
            entropy_years[clust] = get_clusters_shannon_entropy(years_in_clusters)
            
            #entropy on journal
            journals_in_clusters = []
            for uid in author_uids:
                for journal in library[catalog[uid]].journals.split(" "):
                    if journal.strip(): journals_in_clusters.append(journal)
            entropy_journals[clust] = get_clusters_shannon_entropy(journals_in_clusters)
        """
        
        #entropy coauthors for the current cluster
        coauthors_in_cluster = []
        for uid in clusters[new_cluster]:
            for coauthor in library[catalog[uid]].coauthors.split(" "):
                if coauthor.strip(): coauthors_in_cluster.append(coauthor)
        new_entropy_coauthors = get_clusters_shannon_entropy(coauthors_in_cluster)
        old_entropy_coauthors = new_entropy_coauthors
        if len(previous_clusters)> 0:
            coauthors_in_cluster = []
            for uid in previous_clusters[previous_cluster1]:
                for coauthor in library[catalog[uid]].coauthors.split(" "):
                    if coauthor.strip(): coauthors_in_cluster.append(coauthor)
            temp = get_clusters_shannon_entropy(coauthors_in_cluster)
            
            coauthors_in_cluster = []
            for uid in previous_clusters[previous_cluster2]:
                for coauthor in library[catalog[uid]].coauthors.split(" "):
                    if coauthor.strip(): coauthors_in_cluster.append(coauthor)
            old_entropy_coauthors = max(temp,get_clusters_shannon_entropy(coauthors_in_cluster))
        
            
            
        #entropy test on subjects
        subjects_in_cluster = []
        for uid in clusters[new_cluster]:
            for subject in library[catalog[uid]].subjects.split(" "):
                if subject.strip(): subjects_in_cluster.append(subject)
        new_entropy_subjects = get_clusters_shannon_entropy(subjects_in_cluster)

        old_entropy_subjects = new_entropy_subjects        
        if len(previous_clusters)> 0:
            
            subjects_in_cluster = []
            for uid in previous_clusters[previous_cluster1]:
                for subject in library[catalog[uid]].subjects.split(" "):
                    if subject.strip(): subjects_in_cluster.append(subject)
            temp = get_clusters_shannon_entropy(subjects_in_cluster)
            
            subjects_in_cluster = []
            for uid in previous_clusters[previous_cluster2]:
                for subject in library[catalog[uid]].subjects.split(" "):
                    if subject.strip(): subjects_in_cluster.append(subject)
            old_entropy_subjects = max(temp,get_clusters_shannon_entropy(subjects_in_cluster))

        #entropy test on countries
        
        countries_in_cluster = []
        for uid in clusters[new_cluster]:
            for country in library[catalog[uid]].countries.split(" "):
                if country.strip(): countries_in_cluster.append(country)
        new_entropy_countries = get_clusters_shannon_entropy(countries_in_cluster)

        old_entropy_countries = new_entropy_countries        
        if len(previous_clusters)> 0:
            
            countries_in_cluster = []
            for uid in previous_clusters[previous_cluster1]:
                for country in library[catalog[uid]].countries.split(" "):
                    if country.strip(): countries_in_cluster.append(country)
            temp = get_clusters_shannon_entropy(countries_in_cluster)
            
            countries_in_cluster = []
            for uid in previous_clusters[previous_cluster2]:
                for country in library[catalog[uid]].countries.split(" "):
                    if country.strip(): subjects_in_cluster.append(country)
            old_entropy_countries = max(temp,get_clusters_shannon_entropy(countries_in_cluster))
        

        #entropy on journal
 
        journals_in_cluster = []
        for uid in clusters[new_cluster]:
            for journal in library[catalog[uid]].journals.split(" "):
                if journal.strip(): journals_in_cluster.append(journal)
        new_entropy_journals = get_clusters_shannon_entropy(journals_in_cluster)

        old_entropy_journals = new_entropy_journals        
        if len(previous_clusters)> 0:
            
            journals_in_cluster = []
            for uid in previous_clusters[previous_cluster1]:
                for journal in library[catalog[uid]].journals.split(" "):
                    if journal.strip(): journals_in_cluster.append(journal)
            temp = get_clusters_shannon_entropy(journals_in_cluster)
            
            journals_in_cluster = []
            for uid in previous_clusters[previous_cluster2]:
                for journal in library[catalog[uid]].journals.split(" "):
                    if journal.strip(): subjects_in_cluster.append(journal)
            old_entropy_journals = max(temp,get_clusters_shannon_entropy(journals_in_cluster))
        

        #overal entropy 
        all_terms = journals_in_cluster+ countries_in_cluster +subjects_in_cluster +coauthors_in_cluster
        
        entropy_overall = get_clusters_shannon_entropy(all_terms)
        
        coauthor_entropy_history.append(new_entropy_coauthors-old_entropy_coauthors)
        country_entropy_history.append(new_entropy_countries-old_entropy_countries)
        subject_entropy_history.append(new_entropy_subjects-old_entropy_subjects)
        journal_entropy_history.append(new_entropy_journals-old_entropy_journals)
    
        overall_entropy_history.append(entropy_overall)
        
        previous_clusters = clusters
        
    #best 

    my__index = concensus_history.index(max(concensus_history))
    my_ = relevant_thresholds[my__index] 
    
    
    #my__index = paperElbow(range(len(relevant_thresholds),0,-1),relevant_thresholds)
    #my_ = relevant_thresholds[my__index] 
   
    my_clusters_list = hac.fcluster(linkage_matrix, my_, criterion='distance')
    my_clusters = {}
    for k,v in enumerate(my_clusters_list):     
        if v not in my_clusters:
            my_clusters[v] = []
        my_clusters[v].append(library[k].unique_identifier)
    
    #print fmeasure if not in PRODUCTION mode
    if not PRODUCTION:
        my_precision, my_recall, my_f_measure = evaluate(my_clusters.values(), ground_truth.values())
        print("My  value: {0} F-Measure: {1} Precision: {2} Recall: {3}".format(my_, my_f_measure, my_precision, my_recall))
        print("Ground truth  value {0} F-Measure: {1} Precision: {2} Recall: {3}:".format(truth_, truth_fmeasure, truth_precision,truth_recall))
        #print("{0:.3f} / {1:.3f} / {2:.3f}".format(my_precision,my_recall,my_f_measure))
    #else print titles of the articles and clusters
    else:
        print("\nHIERARCHICAL CLUSTERING RESULTS")
        for my_clust, my_author_uids in my_clusters.items():
            print("Author {0} assigned to the following papers:".format(my_clust))
            for uid in my_author_uids:
                print(" - {0}".format(library[catalog[uid]].printout))
            print("\n")

    if plots:
        #normalization for plotting
        coauthor_entropy_history_norm = coauthor_entropy_history
        subject_entropy_history_norm = subject_entropy_history
        country_entropy_history_norm = normalize(country_entropy_history)
        journal_entropy_history_norm = normalize(journal_entropy_history)
        overall_entropy_history_norm = normalize(overall_entropy_history)
        
        #plot the overall results
        max_y = max(max(
            subject_entropy_history_norm,
            coauthor_entropy_history_norm,
            country_entropy_history_norm,
            journal_entropy_history_norm,
            overall_entropy_history_norm
            ))
        min_y = min(min(
            subject_entropy_history_norm,
            coauthor_entropy_history_norm,
            country_entropy_history_norm,
            journal_entropy_history_norm,
            overall_entropy_history_norm
            ))
            
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(15,600,800, 400)
        '''
        plt.figure("Threshold as a function of # of clusters")
        plt.title('Thresholds', fontdict={'fontsize': 18})
        
        plt.ylabel('Thresholds', fontdict={'fontsize': 14})
        plt.xlabel('Number of clusters', fontdict={'fontsize': 14})

        plt.axis([1, len(relevant_thresholds)+1,0,1])         
        plt.plot(range(len(relevant_thresholds),0,-1),relevant_thresholds, 'b', label="Subjects")
        plt.vlines(truth_,0,len(relevant_thresholds))
        
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(15,600,800, 400)
                
        plt.figure("Optimal  representation")
        plt.title('Hierarchical agglomerative clustering thresholds and entropies', fontdict={'fontsize': 18})
        plt.ylabel('Normalized information entropy', fontdict={'fontsize': 14})
        plt.xlabel('HAC ting threshold', fontdict={'fontsize': 14})
        #
        plt.axis([0, 1 , min(min(coauthor_entropy_history_norm),min(country_entropy_history_norm),min(journal_entropy_history_norm),min(subject_entropy_history_norm)), max_y])
        #plt.axis([0, len(within_history)+1 , 0, max_y])
        clusters = [x for x in range(len(within_history)+1,1,-1)]
        clusters = [x for x in range(1,len(within_history)+1)]
        plt.plot(relevant_thresholds, subject_entropy_history_norm, 'b', label="Subjects")
#        plt.plot(relevant_thresholds, overall_entropy_history_norm, 'r', label="Overall")
        plt.plot(relevant_thresholds, coauthor_entropy_history_norm, 'g', label="Coauthors")
        plt.plot(relevant_thresholds, country_entropy_history_norm, 'm', label="Countries")
        plt.plot(relevant_thresholds, journal_entropy_history_norm, 'y', label="Journals")
        plt.legend(loc=2)

        plt.vlines(truth_, min_y, max_y)
        plt.legend(loc=2)
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(15,600,800, 400)
        plt.figure()
        plt.title('Hierarchical agglomerative clustering thresholds and similarities', fontdict={'fontsize': 18})

        plt.plot(relevant_thresholds, within_history, color='r', linewidth=1, label="Intra clusters")
        plt.plot(relevant_thresholds, between_history, color='g', linewidth=1, label="Inter clusters")
        plt.plot(relevant_thresholds, concensus_history, color='b', linewidth=1, label="Consensus")
        plt.plot(my_, concensus_history[list(relevant_thresholds).index(my_)], 'k', marker='o', markersize=10)
        min_y = min(min(concensus_history,within_history,between_history))
        max_y = max(max(concensus_history),max(within_history),max(between_history))
        plt.legend(loc=2)
        plt.ylabel('Similarity', fontdict={'fontsize': 14})
        plt.xlabel('HAC ting threshold', fontdict={'fontsize': 14})
        plt.vlines(truth_, min_y, max_y)
        plt.axis([0, 1 , min_y, max_y])
       
        #plot clustering according to the best 
        fig, ax = plt.subplots() 
        fig.canvas.set_window_title('Hierarchical clustering')
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
        ax.set_title("Hierarchical Agglomerative Clustering for {0}".format(focus_name.split()[0]), fontdict={'fontsize': 18})
        hac.dendrogram(linkage_matrix, color_threshold=my_, orientation='top', leaf_font_size=12, 
            leaf_label_func=lambda id: library[id].author_name + " - " + str(library[id].author_id))
        plt.ylabel("Threshold", fontdict={'fontsize': 14})
        thismanager = plt.get_current_fig_manager()
        thismanager.window.setGeometry(850,45,800, 500)
        '''

def normalize(values):
    norm_values = [(x-min(values))/(max(values)-min(values)) if (max(values)-min(values))!= 0 else 0 for x in values]
    return norm_values
    
if __name__ == "__main__":
    
    # focus_names = ['Zighed %', 'Muller J%', 'Meyer %', 'Morel %', 'Karakiewicz %']
    #focus_names = ['Beer %']
    #focus_names = ['Karakiewicz %', 'Nikolic %', 'Stokes %', 'Rohrmann %', 'Casteilla %', 'Rico %', 'Pita %','Beer %', 'Cartier %', 'Bruce %', 'Kraft %', 'Eklund %', 'Zighed %','Bassetti %', DE AQUI EN ADELANTE SON NUEVOS 'Jouet %', 'Arlot %', 'Pujolle %','Barba %', 'Gaillot %']
    #focus_names = ['Beer %', 'Cartier %', 'Bruce %', 'Kraft %']
    focus_names = ['Abe %'] #['Barba %', 'Gaillot %', ]
    # Nikolic, Stokes, Rohrmann, Casteilla, Rico, Pita
    
    """focus_names = []
        "Lefebvre A%", "Ades %", "Zighed %", "Liu W%", "Bassand %", "Boussaid %", "Meyer R%", "Morel M%", "Abe %", \
        "Karakiewicz %", "Nikolic %", "Allemand J%", "Arlot %", "Barba %", "Bassetti %", "Blaise %", "Casteilla %", "Eklund %", "Chiron %", "Gaillot %",\
        "Godard %", "Gosse %", "Jouet %", "Pita %", "Puget %", "Stokes %", "Rico %", "Rohrmann %", \
        "Muller J%", "Velcin %", "Pujolle %", "Rodriguez F%", "Louvet %", "De Dinechin %", "Daumas %"]"""
    
    """focus_names = []
    conn = Connection()
    #This query is wrong is too detailed, we need only focus names without initials
    sql_results = conn.exee("SELECT DISTINCT(author) FROM articles_authors_disambiguated WHERE authorId>0 AND authorId>=3000 ORDER BY author")
    if sql_results:
        for el in sql_results:
            focus_names.append(el[0] + "%")"""
    #main('', subset=True)
            
    focus_names.sort()
    for n in focus_names:
        main(n)
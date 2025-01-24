#!/usr/bin/env python3
# -*- coding: utf-8 -*-


##########################################    Libraries    ##########################################


import numpy as np
import pandas as pd
import math as math
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import gower
import torch
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from kneebow.rotor import Rotor
from matplotlib.figure import Figure
from minisom import MiniSom
from scipy import optimize
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, OneHotEncoder
from statistics import mean 
from utils import *
from utils_cascade import *


#####################################################################################################
#####################################################################################################




#### WORKFLOW MECANISMO DE AUSENCIA ####
def workflow_absence_mechanism(): 
    ## Inputs
    df_name_ = eval(input("Qual o nome do dataframe que irá criar ausência? "))
    mecha = input("Qual o mecanismo de ausência? (MAR, MNAR, MCAR) ")
    p_miss = float(input("Qual o percentual de valores ausentes? (formato: 0.00) "))

    df_name = pd.DataFrame(df_name_)

    #df_mec_ausencia = input("Qual o nome do dataframe incompleto (output)? ")

    ## Generating null data
    aux_ausencia = eraser(df_name, p_miss=p_miss, mecha=mecha, p_obs=0.5)
    df_mec_ausencia = pd.DataFrame(aux_ausencia['X_incomp'].numpy(), columns= df_name.columns)

    return df_mec_ausencia





#### WORKFLOW IMPUTAÇÃO UNIVARIADA ####
def workflow_imputacao_univariada():
    df_name_ = eval(input("Qual o nome do dataframe que irá utilizar? "))
    df_name = pd.DataFrame(df_name_)

    df_filled, df_null = split_dataset(df_name)       # split dataset
    
    # Generating a univariate imputation for comparison with cascade imputation
    aux_mean, aux_median = univariate_imputation(df_filled, df_null)

    # # Concatenate df_filled and df_cascade_imp 
    df_g1_mean   = pd.concat([df_filled, aux_mean]).sort_index()    
    df_g1_median = pd.concat([df_filled, aux_median]).sort_index()

    return df_g1_mean, df_g1_median





#### WORKFLOW IMPUTAÇÃO CASCATA ####
def workflow_imputacao_cascata():
    df_name = eval(input("Qual o nome do dataframe que irá utilizar? "))

    # Normalization
    apply_normalization = input("Quer normalizar o dataset? (true/false) ")
    
    if apply_normalization.lower() == "true":
        normalization_type = input("Qual tipo de normalização? (standard, minmax, maxabs ou hotencoder) ")
        df_scaled = normalize_data(df_name, apply_normalization=True, normalization_type="minmax") 
    else:
        df_scaled = df_name

    df_filled, df_null = split_dataset(df_scaled)       # split dataset  
    df_mask = get_binarized_df(df_null)                 # binarized dataset
    df_correl = correlation(df_filled)                  # correlation

    
    # Morphology Absence
    cluster_algorithm = input("Qual o algoritmo será usado na morfologia da ausência? (som, kmodes, dbscan, agglomerative_cluster)")
    if cluster_algorithm == "kmodes": 
        n_clusters1 = int(input("Qual será o valor de K?"))
        labels = morphology_absence(df_mask, cluster_algorithm="kmodes", n_clusters=n_clusters1)
    elif cluster_algorithm == "dbscan": 
        eps = float(input("Qual será o valor do eps? (formato: 0.00)"))
        min_samples = int(input("Qual será o valor de mínimo da amostra?"))
        metric = input("Qual a distancia que será usada? (euclidean, precomputed)")
        labels = morphology_absence(df_mask, cluster_algorithm="dbscan", eps=eps, min_samples=min_samples, metric=metric)   
    elif cluster_algorithm == "agglomerative_cluster": 
        n_clusters2 = int(input("Qual será o valor de K?"))
        affinity = input("Qual a distancia que será usada? (euclidean, precomputed)")
        if affinity =="precomputed": 
            labels = morphology_absence(df_mask, cluster_algorithm="agglomerative_cluster", n_clusters=n_clusters2, affinity="precomputed", linkage='complete')
        else:
            labels = morphology_absence(df_mask, cluster_algorithm="agglomerative_cluster", n_clusters=n_clusters2, affinity="euclidean", linkage='ward')
    elif cluster_algorithm == "som":
        dim_x = int(input("Qual será a dimensão de x?"))
        dim_y = int(input("Qual será a dimensão de y?"))
        sigma = float(input("Qual será o valor de sigma? (formato: 0.00)"))
        lr = float(input("Qual será o valor do learning rate? (formato: 0.00)"))
        max_iter = int(input("Qual será o valor máximo de iteração?"))
        labels = morphology_absence(df_mask, cluster_algorithm="som", dim_x=dim_x, dim_y=dim_y, sigma=sigma, lr=lr, max_iter=max_iter) 
    else:
        print('Error: Algoritmo não encontrado')
    
    
    df_mask = labels_col_df(df_mask, labels)         # create label column 

    # Cluster ordering criterion
    order_cluster = input("Qual será o critério de ordenação dos grupos? (tupleLessMissing, tupleMoreMissing, fieldLessMissing, fieldMoreMissing, fieldPerTupleLessMissing, fieldPerTupleMoreMissing, random, noSort)")
    ordered_clu_list = cluster_order_criterion(df_mask, order_cluster=order_cluster)

    # Attribute ordering criterion
    order_column = input("Qual será o critério de ordenação dos atributos? (lessCorrelation, moreCorrelation, lessMissing, moreMissing, noSort)")
    ordered_col_list = attribute_order_criterion(df_mask, df_correl, order_column=order_column)
        
    # Cascade imputation
    n_clusters_knn = int(input("Qual será o valor de K a  ser usado no KNN?"))
    df_cascade_imp = cascade_imputation(df_filled, df_null, df_mask, ordered_clu_list, ordered_col_list, knn_neighbors=n_clusters_knn, 
                                            knn_metric='euclidean')
    
    # Concatenate df_filled and df_cascade_imp
    df_g2_imput = pd.concat([df_filled, df_cascade_imp]).sort_index()
    
    return df_g2_imput





#### WORKFLOW CALCULO DOS ERROS ####
def workflow_calculo_erros_g1():
    df_original = eval(input("Qual o nome do dataframe original? "))
    df_media = eval(input("Qual o nome do dataframe imputado por média? "))
    df_mediana = eval(input("Qual o nome do dataframe imputado por mediana? "))    
    #df_cascata = eval(input("Qual o nome do dataframe imputado por cascata? "))

    # Univariate imputation error by mean
    avg_rmse_media, avg_sim_error_media, correl_bias_media = error_metrics(df_original, df_media)

    # Univariate imputation error by median
    avg_rmse_mediana, avg_sim_error_mediana, correl_bias_mediana = error_metrics(df_original, df_mediana)

    return avg_rmse_media, avg_sim_error_media, correl_bias_media, avg_rmse_mediana, avg_sim_error_mediana, correl_bias_mediana




def workflow_calculo_erros_g2():
    df_original = eval(input("Qual o nome do dataframe original? "))
    df_cascata = eval(input("Qual o nome do dataframe imputado por cascata? "))

    # Cascade imputation error
    avg_rmse_cascata, avg_sim_error_cascata, correl_bias_cascata = error_metrics(df_original, df_cascata)

    return avg_rmse_cascata, avg_sim_error_cascata, correl_bias_cascata
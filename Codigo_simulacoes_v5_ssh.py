# pip install utils
# pip install torch
# pip install minisom
# pip install pyclustering
# pip install wget 
# pip uninstall scikit-learn
# pip install scikit-learn==1.2.2
# pip install gower
# pip install kmodes
# pip install kneebow
# pip install ucimlrepo
# pip install --upgrade pandas

## Libraries
import logging
import numpy as np
import pandas as pd
import os
import time
from joblib import Parallel, delayed
from utils import *
from utils_cascade import *

# Configurando o nível de registro (pode ser DEBUG, INFO, WARNING, ERROR, ou CRITICAL)
logging.basicConfig(level=logging.INFO)



############################################################################################################################
################################        PEGANDO UMA BASE PARA O EXPERIMENTO

#iris_df = pd.read_parquet('docs/Dados/iris_df(teste).parquet')
wine_df = pd.read_parquet('docs/Dados/wine_df.parquet')
#pima_df = pd.read_parquet('docs/Dados/pima_df.parquet')
#boston_df = pd.read_parquet('docs/Dados/boston_df.parquet').drop(columns=['RAD'])
#abalone_df = pd.read_parquet('docs/Dados/abalone_df.parquet').drop(columns=['Sex'])
#gamma_df = pd.read_parquet('docs/Dados/gamma_df.parquet')
#glass_df = pd.read_parquet('docs/Dados/glass_df.parquet')
#forest_df = pd.read_parquet('docs/Dados/forest_df.parquet')
#yeast_df = pd.read_parquet('docs/Dados/yeast_df.parquet')


#df = iris_df
df = wine_df
#df = pima_df
#df = boston_df
#df = abalone_df
#df = gamma_df
#df = glass_df
#df = forest_df
# df = yeast_df



############################################################################################################################
################################        SELECIONANDO OS PARÂMETROS

# Ler os parâmetros do arquivo CSV 
params = pd.read_csv("docs/params_teste_full.csv", sep = ';', decimal=',') #.fillna(0)
#params

## Filtrando por dados
params_df = params.loc[(params['dados'] == 'wine')]  

#params_df



############################################################################################################################
################################        IMPUTAÇÃO EM CASCATA PARALELIZADA
############################################################################################################################

def preprocess_data(params_df, df):
    cache = {}
    for _, row in params_df.iterrows():
        params = row.to_dict()
        mec_ausencia = params['mec_ausencia']
        pct_ausencia = params['pct_ausencia']
        chave = (mec_ausencia, pct_ausencia)
        
        if chave not in cache:
            dados, _, _, df_incompleto = eraser_with_params(df.copy(), params)

            df_g1_mean, df_g1_median, time_g1 = univariate_imputation_with_params(df_incompleto)
            avg_mse_media, avg_rmse_media, avg_mae_media, avg_mape_media, avg_sim_error_media, correl_bias_media = error_metrics(df.copy(), df_g1_mean)
            avg_mse_mediana, avg_rmse_mediana, avg_mae_mediana, avg_mape_mediana, avg_sim_error_mediana, correl_bias_mediana = error_metrics(df.copy(), df_g1_median)

            cache[chave] = (df_incompleto, avg_mse_media, avg_rmse_media, avg_mae_media, avg_mape_media, avg_sim_error_media, correl_bias_media,
                            avg_mse_mediana, avg_rmse_mediana, avg_mae_mediana, avg_mape_mediana, avg_sim_error_mediana, correl_bias_mediana, time_g1)

    return cache



def process_cascade_imputation(row, df, cache):
    params = row.to_dict()
    mec_ausencia = params['mec_ausencia']
    pct_ausencia = params['pct_ausencia']
    chave = (mec_ausencia, pct_ausencia)

    df_incompleto, avg_mse_media, avg_rmse_media, avg_mae_media, avg_mape_media, avg_sim_error_media, correl_bias_media, \
    avg_mse_mediana, avg_rmse_mediana, avg_mae_mediana, avg_mape_mediana, avg_sim_error_mediana, correl_bias_mediana, time_g1 = cache[chave]
    
    ## Cascata Simplificada
    idx_simulacao, df_g2_imput_only, time_g2_only = cascade_only_imputation_with_params(df_incompleto.copy(), params)
    avg_mse_cascata_only, avg_rmse_cascata_only, avg_mae_cascata_only, avg_mape_cascata_only, avg_sim_error_cascata_only, correl_bias_cascata_only = error_metrics(df.copy(), df_g2_imput_only)
    
    ## Cascata Simplificada
    idx_simulacao, df_g2_imput, time_g2 = cascade_imputation_with_params(df_incompleto.copy(), params)
    avg_mse_cascata, avg_rmse_cascata, avg_mae_cascata, avg_mape_cascata, avg_sim_error_cascata, correl_bias_cascata = error_metrics(df.copy(), df_g2_imput)

    return pd.DataFrame({        
        "idx_simulacao": [idx_simulacao],
        "avg_mse_media": [avg_mse_media],
        "avg_rmse_media": [avg_rmse_media],
        "avg_mae_media": [avg_mae_media],
        "avg_mape_media": [avg_mape_media],
        "avg_sim_error_media": [avg_sim_error_media],
        "correl_bias_media": [correl_bias_media],
        "avg_mse_mediana": [avg_mse_mediana],
        "avg_rmse_mediana": [avg_rmse_mediana],
        "avg_mae_mediana": [avg_mae_mediana],
        "avg_mape_mediana": [avg_mape_mediana],
        "avg_sim_error_mediana": [avg_sim_error_mediana],
        "correl_bias_mediana": [correl_bias_mediana],
        "time_g1": [time_g1],
        "avg_mse_cascata": [avg_mse_cascata],
        "avg_rmse_cascata": [avg_rmse_cascata],
        "avg_mae_cascata": [avg_mae_cascata],
        "avg_mape_cascata": [avg_mape_cascata],
        "avg_sim_error_cascata": [avg_sim_error_cascata],
        "correl_bias_cascata": [correl_bias_cascata],
        "time_g2": [time_g2],
        "avg_mse_cascata_only": [avg_mse_cascata_only],
        "avg_rmse_cascata_only": [avg_rmse_cascata_only],
        "avg_mae_cascata_only": [avg_mae_cascata_only],
        "avg_mape_cascata_only": [avg_mape_cascata_only],
        "avg_sim_error_cascata_only": [avg_sim_error_cascata_only],
        "correl_bias_cascata_only": [correl_bias_cascata_only],
        "time_g2_only": [time_g2_only]
    })


cache = preprocess_data(params_df, df)

start_parallel = time.time()
results = Parallel(n_jobs=os.cpu_count())(delayed(process_cascade_imputation)(row, df, cache) for _, row in params_df.iterrows())
final_results_df = pd.concat(results, ignore_index=True)
end_parallel = time.time()

print("O código rodou em: ", end_parallel - start_parallel) 



# Salvar em um arquivo CSV
nome_resultado = f'Resultado_simulacao_wine_full'
final_results_df.to_csv(f"/home/tgomes/{nome_resultado}.csv", index=False) 


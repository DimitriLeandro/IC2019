# ESSE ARQUIVO VAI IMPLEMENTAR FUNCOES PARA INICIAR O OBJETO DO CLASIFICADOR DESEJADO 
# E TAMBEM DO OBJETO NORMALIZADOR (STANDARD SCALER). A IDEIA E QUE HAJA UMA FUNCAO QUE
# RETORNE UM CLASSIFICADOR JA TREINADO E UM NORMALIZADOR TAMBEM. ASSIM, PARA CADA JANELA
# GRAVADA COM O RESPEAKER, EU APLICO O DaS, EXTRAIO AS FEATURES, NORMALIZO O VETOR DE 
# FEATURES E DEPOIS CLASSIFICO A AMOSTRA. 
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def ajustarDatasetTreinamento(caminhoCSV, verbose=False):
    
    # ABRINDO
    if verbose == True:
        print("Carregando o CSV do dataset")
    dataframe = pd.read_csv(caminhoCSV)
    if verbose == True:
        print("CSV carregado")
    
    # SEPARANDO DATA E TARGET
    if verbose == True:
        print("Separando o que é data e target")
    data     = dataframe.iloc[:, 1:-1].to_numpy()
    target   = dataframe.iloc[:, -1].to_numpy()

    return data, target

def iniciarObjNormalizador(data, verbose=False):
    if verbose == True:
        print("Iniciando objeto normalizador")
    objNormalizador = StandardScaler()
    if verbose == True:
        print("Treinando objeto normalizador")
    objNormalizador.fit(data)
    if verbose == True:
        print("Normalizando os dados de treinamento")
    data = objNormalizador.transform(data)
    return data, objNormalizador

def iniciarObjClassificador(data, target, classificador=None, verbose=False):
    
    if classificador == None:
        classificador = "SGDClassifier(alpha=0.0001, average=False, class_weight=None, early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True, l1_ratio=0.15, learning_rate='optimal', loss='log', max_iter=1000, n_iter_no_change=5, n_jobs=1, penalty='l2', power_t=0.5, random_state=None, shuffle=True, tol=0.0001, validation_fraction=0.1, verbose=0, warm_start=False)"
    
    if verbose == True:
        print("Instanciando objeto classificador")
    objClassificador = eval(classificador)
    
    if verbose == True:
        print("Treinando o classificador")
    objClassificador.fit(data, target)
    
    return objClassificador

def main(caminhoCSV, classificador=None, verbose=False):
    
    # OBTENDO X E Y
    data, target = ajustarDatasetTreinamento(caminhoCSV, verbose)
    
    # NORMALIZANDO AS FEATURES E OBTENDO O OBJ NORMALIZADOR
    data, objNormalizador = iniciarObjNormalizador(data, verbose)

    # TREINANDO UM CLASSIFICADOR COM TODAS AS DIMENSOES    
    objClassificador = iniciarObjClassificador(data, target, classificador, verbose)
    
    if verbose == True:
        print("objClassificador e objNormalizador prontos: operação finalizada.")
        
    return objClassificador, objNormalizador   

# ESSE ALGORITMO VAI SERVIR PRA GERAR UM DATASET DE DADOS FICTICIOS QUE SERAO UTILIZADOS PRA 
# TREINAR O REGRESSOR PARA FAZER AS ESTIMATIVAS DE DIRECAO DE CHEGADA

import numpy as np
import time
from math import cos, sin, pi, sqrt
from pandas import DataFrame

def radParaGrau(angulo):
    return (angulo*180)/pi

def grauParaRad(angulo):
    return (angulo*pi)/180

def vetorUnitario(vetor):
    return vetor/np.linalg.norm(vetor)

def tempoParaAmostras(tempo, freqAmostragem):
    return tempo * freqAmostragem

def escreverCabecalho(qtdMics):
	# O CABECALHO DESSE DATASET SERA O DELAY ENTRE DOIS MICROFONES I E J. 
	# VOU FAZER TODAS AS COMBINACOES POSSIVEIS MSM QUE ISSO SEJA REDUNDANTE
	# AO FINAL, O DATASET TB TEM QUE TER A CLASSIFICACAO CORRETA

	cabecalhoCSV = []
	for micI in range(0, 8):
		for micJ in range(micI + 1, 8):
			cabecalhoCSV.append("delay" + str(micI) + str(micJ))

	# TEREMOS DUAS CLASSIFICACOES CORRETAS, O ANGULO AZIMUTAL E O DE ELEVACAO
	cabecalhoCSV.append("azimutalReal") 
	cabecalhoCSV.append("elevacaoReal")

	return cabecalhoCSV

# AGORA SIM VAMOS COMECAR O ALGORITMO--------------------------------------------

# DEFININDO O CABECALHO DO CSV
cabecalhoCSV = escreverCabecalho(8)

# COORDENADAS ORIGINAIS DO DREGON DATASET
coordenadasMics = np.array([
    [0.0420, 0.0615, -0.0410],
    [-0.0420, 0.0615, 0.0410],
    [-0.0615, 0.0420, -0.0410],
    [-0.0615, -0.0420, 0.0410],
    [-0.0420, -0.0615, -0.0410],
    [0.0420, -0.0615, 0.0410],
    [0.0615, -0.0420, -0.0410],
    [0.0615, 0.0420, 0.0410]
])

# CRIANDO A MATRIZ DE DADOS
matrizDadosCSV = []

# VOU PRECISAR DA FREQUENCIA DE AMOSTRAGEM DO DATASET
freqAmostragem = 44100

# VOU CALCULAR OS DELAYS COMO SE O SOM TIVESSE VINDO DAS SEGUINTES COMBINACOES
# DE AZIMUTAIS E ELEVACOES

azimutaisDesejados = [45,60,75,90]
elevacoesDesejadas = [-30,-15,0]
for azimutalAtual in azimutaisDesejados:
    for elevacaoAtual in elevacoesDesejadas:
#for azimutalAtual in np.arange(0, 91, 10):
#    for elevacaoAtual in np.arange(-90, 1, 10):
        
        # TENHO QUE PASSAR PRA RADIANOS PRA FAZER AS CONTAS
        azimutalAtualRad = grauParaRad(azimutalAtual)
        elevacaoAtualRad = grauParaRad(elevacaoAtual)

        # FAZENDO TODAS AS COMBINACOES DOS MICROFONES
        linhaAtualMatrizDados = []
        for micI in range(0, 8):
            for micJ in range(micI + 1, 8):

                # COORDENADAS DA DIFERENCA DOS MICS
                coordenadasDiferenca = coordenadasMics[micI] - coordenadasMics[micJ]

                # COORDENADAS DO VETOR WAVEFRONT (JA VOU DEIXAR ELE UNITARIO)
                w = vetorUnitario(np.array([
                    cos(azimutalAtualRad)*cos(elevacaoAtualRad),
                    sin(azimutalAtualRad)*cos(elevacaoAtualRad),
                    sin(elevacaoAtualRad)
                ]))

                # CALCULANDO O PRODUTO INTERNO E DIVIDINDO PELA VELOCIDADE DO SOM
                delayTemporal = coordenadasDiferenca[0] * w[0]
                delayTemporal += coordenadasDiferenca[1] * w[1]
                delayTemporal += coordenadasDiferenca[2] * w[2]
                delayTemporal /= 340

                # VOU COLOCANDO OS RESULTADOS NA LINHA ATUAL DA MATRIZ DE DADOS
                linhaAtualMatrizDados.append(tempoParaAmostras(delayTemporal, freqAmostragem))
        
        # AGORA, JA POSSO COLOCAR AS CLASSIFICACOES CORRETAS NO FINAL DA LINHA E COLOCAR A LINHA NA MATRIZ
        linhaAtualMatrizDados.append(azimutalAtual)
        linhaAtualMatrizDados.append(elevacaoAtual)
        matrizDadosCSV.append(linhaAtualMatrizDados)
        
# AGORA QUE ACABEI DE GERAR OS DADOS FICTICIOS, POSSO FAZER O DATAFRAME E ESCREVE-LO NUM CSV
dataFrame = DataFrame(matrizDadosCSV, columns=cabecalhoCSV)

# ESCREVENDO O CSV
nomeCSV = str(time.time()) + ".csv"
dataFrame.to_csv(nomeCSV, index=False)
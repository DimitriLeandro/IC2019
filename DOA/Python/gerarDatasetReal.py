# ESSE ALGORITMO VAI SER USADO PARA CRIAR UM DATASET DE DOA USANDO ARQUIVOS WAV
# A IDEIA E FAZER UM CSV EM QUE CADA COLUNA SEJA O DELAY CALCULADO ENTRE DOIS MICROFONES I E J
# COM ISSO, VAI SER POSSIVEL FAZER A ESTIMATIVA DE DIRECAO DE CHEGADA USANDO REGRESSAO 
# OBS: PRESSUPOE-SE QUE UM ARQUIVO ARQUIVO WAV JA CONTENHA TODOS OS MICROFONES, PARA QUE O 
# LOAD DO LIBROSA RETORNE UM ARRAY COM CADA UM DOS SINAIS DOS MICROFONES (O DREGON, POR EXEMPLO, E ASSIM)
# OBS2: OS ARQUIVOS TEM QUE ESTAR NOMEADOS COMO O DREGON: azimutal_elevacao_qqrCoisa.wav

#sys.argv[0] -> delayDatasetCreator.py
#sys.argv[1] -> diretorio onde os audios estao /home/dimi/Downloads/datasets/DREGON_clean_recordings_speech/
#sys.argv[2] -> distancia maxima entre dois microfones (vai ser usado para calcular o delay maximo)

import sys
import time
import numpy as np
from pandas import DataFrame
from librosa import load
from os import listdir

# DEFINICAO DE ALGUMAS FUNCOES
def definirParametrosIniciais(diretorioDeAudios, LMax, velocidadeSom=340):
	# ESSA FUNCAO VAI SER USADA PARA CALCULAR O DELAY MAXIMO POSSIVEL ENTRE DOIS MICROFONES SEGUNDO A GEOMETRIA
	# A RESPOSTA ESTA EM QUANTIDADE DE AMOSTRAS. NAO EM SEGUNDOS 

	# Primeiro, tenho que saber a frequencia de amostragem. Vou pegar so o primeiro arquivo da pasta de audios pra saber esse valor
	for nomeArquivo in listdir(diretorioDeAudios):
		originalAudio, freqAmostragem = load(diretorioDeAudios + nomeArquivo, sr=None, mono=False)
		qtdMics = len(originalAudio)
		break 

	# Agora sim eu faco o calculo
	tempo        = LMax / velocidadeSom
	maxDelay     = freqAmostragem * tempo
	
	return freqAmostragem, int(maxDelay + 1), qtdMics

def verificarDelay(sinalA, sinalB, maxDelay=15):
	
	# Verificando se os dois sinais tem o mesmo tamanho e se maxDelay é compatível
	if len(sinalA) != len(sinalB) or maxDelay >= len(sinalA)-1 :
		return False
	
	# Sei que o maior delay possível (em qtd de amostras) é maxDelay. Portanto, em cada iteração, 
	# a comparação entre os sinais para gerar a correlação se dará com arrays de tamanho 
	# len(sinal) - maxDelay
	tamanho = len(sinalA) - maxDelay
	
	# O delay será a qtd de amostras pra trás ou para frente que há
	# entre o sinal B em relação ao sinal A. Negativo significa que B está adiantado, 0 que não há
	# delay e positivo que está atrasado.
	delay = 0
	melhorCorrelacao = -1    
	inicioA = 0
	fimA    = inicioA + tamanho
	inicioB = len(sinalB) - tamanho
	fimB    = inicioB + tamanho
	
	# Fazendo as iterações e calculando a correlação entre os sinais
	for i in range(-maxDelay, maxDelay+1):
		
		# Calculando a correlação da iteração atual
		corrAtual = np.corrcoef(sinalA[inicioA:fimA], sinalB[inicioB:fimB])[0][1]
		
		# Verificando se encontramos uma correlacao maior ainda para atualizar o delay
		if corrAtual > melhorCorrelacao:
			melhorCorrelacao = corrAtual
			delay = -i
			
		# Fazendo os indexes dos arrays da próxima iteração. De i = -maxDelay até 0, o Sinal A 
		# fica parado e o Sinal B vem vindo pra trás. A partir de i = 1 até maxDelay, o Sinal B
		# fica parado e o Sinal A vai embora.
		if i < 0:
			inicioB -= 1
			fimB     = inicioB + tamanho
		else:
			inicioA += 1
			fimA     = inicioA + tamanho
	
	# Retornando o delay
	return delay

def escreverCabecalho(qtdMics):
	# O CABECALHO DESSE DATASET SERA O DELAY ENTRE DOIS MICROFONES I E J. 
	# VOU FAZER TODAS AS COMBINACOES POSSIVEIS MSM QUE ISSO SEJA REDUNDANTE
	# AO FINAL, O DATASET TB TEM QUE TER A CLASSIFICACAO CORRETA. E A PRIMEIRA 
	# COLUNA VAI SER O NOME DO ARQUIVO

	cabecalhoCSV = ["nomeArquivo"]
	for micI in range(0, 8):
		for micJ in range(micI + 1, 8):
			cabecalhoCSV.append("delay" + str(micI) + str(micJ))

	# TEREMOS DUAS CLASSIFICACOES CORRETAS, O ANGULO AZIMUTAL E O DE ELEVACAO
	cabecalhoCSV.append("azimutalReal") 
	cabecalhoCSV.append("elevacaoReal")

	return cabecalhoCSV

# DEFINICAO DE VARIAVEIS GLOBAIS
diretorioDeAudios                 = sys.argv[1]
LMax                              = float(sys.argv[2])
freqAmostragem, maxDelay, qtdMics = definirParametrosIniciais(diretorioDeAudios, LMax)
matrizDadosCSV                    = []

# DEFININDO O CABECALHO DO CSV
cabecalhoCSV = escreverCabecalho(qtdMics)

# PARA CADA ARQUIVO NA PASTA DE AUDIOS, VOU EXTRAIR AS FEATURES
i = 0
for nomeArquivo in listdir(diretorioDeAudios):
	i += 1
	print("Analisando arquivo ", i)
	#if i == 10:
	#	break

	# PRIMEIRO, ABRO O ARQUIVO COM O LIBROSA
	originalAudio, freqAmostragem = load(diretorioDeAudios + nomeArquivo, sr=None, mono=False)
	
	# PARA CADA COMBINACAO DE DOIS MICROFONES, CALCULO O DELAY E COLOCO NA NOVA LINHA DO DATASET
	# LEMBRANDO QUE A PRIMEIRA COLUNA VAI SER O NOME DO ARQUIVO
	novaLinhaCSV = [nomeArquivo]
	for micI in range(0, 8):
		for micJ in range(micI + 1, 8):
			novaLinhaCSV.append(verificarDelay(originalAudio[micI], originalAudio[micJ], maxDelay))

	# AINDA FALTA COLOCAR AS CLASSIFICACOES CORRETAS NA NOVA LINHA DO CSV (azimutalReal e elevacaoReal)
	nomeArquivoSeparado = nomeArquivo.split("_")
	azimutalReal = nomeArquivoSeparado[0]
	elevacaoReal = nomeArquivoSeparado[1]
	novaLinhaCSV.append(azimutalReal)
	novaLinhaCSV.append(elevacaoReal)

	# AGORA QUE A NOVA LINHA TA PRONTA, VOU COLOCA-LA NA MATRIZ DE DADOS DO CSV
	matrizDadosCSV.append(novaLinhaCSV)

# QUANDO ACABAR O FOR, POSSO FAZER O DATAFRAME E ESCREVE-LO NUM CSV
dataFrame = DataFrame(matrizDadosCSV, columns=cabecalhoCSV)

# ESCREVENDO O CSV
caminho = diretorioDeAudios + str(time.time()) + ".csv"
dataFrame.to_csv(caminho, index=False)
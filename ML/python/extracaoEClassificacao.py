'''
	PARA RODAR ESSE CÓDIGO:
	python3 analiseFeatures.py [bitsProfundidade] [freqAmostragem] [normalizador] [classificador] [FRAME_TIME] [OVERLAP_TIME]

	ONDE:
	bitsProfundidade -> 08 ou 16
	freqAmostragem -> 08, 16, 24, 32, 40 ou 48 
	normalizador -> l1, l2, scale, standardScaler, robustScaler ou 0
	classificador -> KNN, Tree ou SVM
	frame time -> milisegundos
	overlap time -> milisegundos
'''

import numpy
import time
import csv
import sys
import os
import pandas
import librosa
from scipy.stats import skew, kurtosis, mode
from sklearn.preprocessing import normalize, scale, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# ------------------------------ DEFINICAO DE FUNCOES --------------------
def rearranjarDataset(datasetOriginal, featuresSelecionadas, featureAtual):
	
	# ESSA FUNCAO ADICIONA UMA NOVA FEATURE AO DATASET
	colunasTop = ['pasta','arquivo', 'classe'] + featuresSelecionadas + [featureAtual]

	dataset = datasetOriginal.loc[:, datasetOriginal.columns.intersection(colunasTop)]
	
	return dataset

def normalizarDataset(matrizFeatures, normalizador):
	if normalizador == 'l1':
		return pandas.DataFrame(normalize(matrizFeatures, norm='l1'))
	elif normalizador == 'l2':
		return pandas.DataFrame(normalize(matrizFeatures, norm='l2'))
	elif normalizador == 'scale':
		return pandas.DataFrame(scale(matrizFeatures))
	elif normalizador == 'standardScaler':
		return pandas.DataFrame(StandardScaler().fit_transform(matrizFeatures))
	elif normalizador == 'robustScaler':
		return pandas.DataFrame(RobustScaler().fit_transform(matrizFeatures))
	else:
		# MESMO NAO TENDO NORMALIZADOR, PRECISO FAZER AS COLUNAS TEREM INDEXES E NAO NOMES
		return pandas.DataFrame(numpy.array(matrizFeatures))

def escreverCabecalho(writerCSVClassificador, writerCSVReduzido):
	cabecalho = ['qtd', 'ftx', 'acc', 'time_train', 'time_test']
	writerCSVClassificador.writerow(cabecalho)
	writerCSVReduzido.writerow(cabecalho)

def separarXeYTreino(pastaTeste, dataset):
	# SELECIONANDO TODAS AS LINHAS QUE NAO SEJAM DA PASTA TESTE
	dataset = dataset.loc[dataset['pasta'] != pastaTeste]

	xTrain = dataset.drop(['pasta', 'arquivo', 'classe'], axis=1) # TUDO MENOS AS COLUNAS 'PASTA', 'ARQUIVO' E 'CLASSE'
	yTrain = dataset['classe'] # APENAS A COLUNA 'CLASSE'

	return xTrain, yTrain

def atualizarFeaturesSelecionadas(melhorFeature, featuresRestantes, featuresSelecionadas):
	featuresSelecionadas.append(melhorFeature)
	indexPraRemover = numpy.where(featuresRestantes == melhorFeature)
	featuresRestantes = numpy.delete(featuresRestantes, indexPraRemover)

	return featuresRestantes, featuresSelecionadas

def melGibson(paixaoDeCristo):
	mfcc = []

	for linha in paixaoDeCristo:
		mfcc.append(numpy.mean(linha))

	return mfcc

def delta1(coracaoValente):
	delta1 = librosa.feature.delta(coracaoValente, mode='nearest')

	arrayDelta1 = []

	for linha in delta1:
		arrayDelta1.append(numpy.mean(linha))

	return arrayDelta1

def delta2(maquinaMortifera):
	delta2 = librosa.feature.delta(maquinaMortifera, order=2, mode='nearest')

	arrayDelta2 = []

	for linha in delta2:
		arrayDelta2.append(numpy.mean(linha))

	return arrayDelta2

def mediaRMS(y, tamanhoY):
	return numpy.mean(librosa.feature.rms(y=y, frame_length=tamanhoY))

def centroideEspectral(y, sr, tamanhoY):
	return numpy.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=tamanhoY))

def larguraEspectral(y, sr, tamanhoY):
	return numpy.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=tamanhoY))

def planicidadeEspectral(y, tamanhoY): 
	return numpy.mean(librosa.feature.spectral_flatness(y=y, n_fft=tamanhoY))

def rolloff(y, sr, tamanhoY):
	return numpy.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=tamanhoY))

def cruzamentosZero(y, tamanhoY):
	return numpy.mean(librosa.feature.zero_crossing_rate(y, frame_length=tamanhoY))

def assimetria(y):
	return skew(y)

def curtose(y):
	return kurtosis(y)

def variancia(y):
	return numpy.var(y)

def main(datasetOriginal, bitsProfundidade, freqAmostragem, classificador, normalizador, FRAME_LENGTH, OVERLAP_LENGTH):

	# DECLARACAO DE VARIAVEIS
	dataset = datasetOriginal.loc[:, datasetOriginal.columns.intersection(['pasta','arquivo', 'classe'])] #o dataset vai comecar so com as colunas "pasta", "arquivo" e "classe"
	featuresRestantes = numpy.arange(0, datasetOriginal.shape[1] - 3) # do jeito que o datasetOriginal vem, as colunas estão nomeadas de acordo com featuresRestantes, sucesso
	featuresSelecionadas = []
	acuraciaAnterior = 0.001
	contWhile = 0
	arquivoCSV = classificador + "_" + normalizador + "_" + bitsProfundidade + "bits_" + freqAmostragem + "kHz.csv"
	arquivoCSVReduzido = classificador + "_" + normalizador + "_" + bitsProfundidade + "bits_" + freqAmostragem + "kHz_REDUZIDO.csv"

	# ABRINDO OS CSVS DOS RESULTADOS DOS CLASSIFICADORES
	with open(arquivoCSV, 'a') as csvFile:
		with open(arquivoCSVReduzido, 'a') as csvFileReduzido:

			# CRIANDO O OBJETO RESPONSAVEL POR ESCREVER LINHAS NO CSV
			writerCSVClassificador = csv.writer(csvFile)
			writerCSVReduzido = csv.writer(csvFileReduzido)

			#ESCREVENDO O CABECALHO DO CSV DO CLASSIFICADOR
			escreverCabecalho(writerCSVClassificador, writerCSVReduzido)	
		  
			# ENQUANTO HOUVEREM FEATURES A SEREM TESTADAS
			while(len(featuresRestantes) != 0):
			
				contWhile += 1
				print("INICIO:", contWhile)

				# VARIAVEIS QUE VAO GUARDAR AS INFORMACOES DA MELHOR FEATURE DESTA ITERACAO DO WHILE
				melhorAcuracia = 0
				melhorFeature = 1
				tempoTotalTreinoMelhorFeature = 0
				tempoTotalTesteMelhorFeature = 0

				# PARA CADA FEATURE RESTANTE EU A ADICIONO NAS QUE JA ESTAO SENDO UTILIZADAS
				for i, featureAtual in enumerate(featuresRestantes):

					#print("Testando a combinacao de features:", featuresSelecionadas + [featureAtual])

					# REARRANJO O DATASET COM A FEATURE ATUAL
					dataset = rearranjarDataset(datasetOriginal, featuresSelecionadas, featureAtual)

					# AGORA QUE JA TENHO UM DATASET COM AS FEATURES DA VEZ, TENHO QUE CRIAR OS CLASSIFICADORES
					if classificador == 'KNN':
						objClassificador = KNeighborsClassifier(3)
					elif classificador == 'Tree':
						objClassificador = DecisionTreeClassifier()
					else:
						objClassificador = SVC(gamma='scale', decision_function_shape='ovo')
					
					# ESSES ARRAYS VAO GUARDAR AS ACURACIAS DE CADA ITERACAO DO KFOLD, MAS SERAO RESETADOS QUANDO UMA NOVA FEATURE FOR TESTADA
					arrayAcuraciasCadaKFold = []				

					# TEMPO DE TREINAMENTO DAS 10 ITERACOES DO KFOLD
					tempoTotalTreino = 0
					tempoTotalTeste = 0

					# PARA CADA ITERACAO DO KFOLD
					for pastaTeste in range(1,11):

						#print("Nova iteracao do KFold. Pasta de teste:", pastaTeste)

						# SEPARANDO O QUE E X E O QUE E Y DE -> T R E I N O 
						xTrain, yTrain = separarXeYTreino(pastaTeste, dataset)

						# TREINANDO O CLASSIFICADOR
						inicioTreino = time.time()
						objClassificador  = objClassificador.fit(xTrain, yTrain)
						fimTreino = time.time()
						tempoTotalTreino += fimTreino - inicioTreino

						#print("Treinamento finalizado. Tempo total com as iterações anteriores do KFold para essa combinação de features:", tempoTotalTreino)

						# TENHO QUE CONSTRUIR O YTEST E O YPRED
						yTest = []
						yPredict = []

						# PEGANDO ARQUIVO POR ARQUIVO DE TESTE PARA SEGMENTAR E CLASSIFICAR
						for index, dadoTeste in dataset.iterrows():
							if dadoTeste['pasta'] == pastaTeste:

								# ABRINDO O AUDIO ATUAL
								caminhoAudio = "conversoes/" + bitsProfundidade + "bits/" + freqAmostragem + "kHz" + "/" + "fold" + str(pastaTeste) + "/" + dadoTeste['arquivo']
								audioTesteOriginal, sr = librosa.load(caminhoAudio, sr=int(freqAmostragem)*1000)

								#print("Abrindo e segmentando o arquivo", caminhoAudio)

								# CADA AUDIO SERA DIVIDIDO DE ACORDO COM O TAMANHO DA JANELA
								matrizAudioSeparado = librosa.util.frame(audioTesteOriginal, frame_length=FRAME_LENGTH, hop_length=OVERLAP_LENGTH).transpose()

								xTestCadaJanela = []

								# PARA CADA FRAME EU CALCULO AS FEATURES SEPARADAMENTE
								for pedacoAudioNovo in matrizAudioSeparado:

									# VOU PRECISAR DO TAMANHO DA JANELA
									tamanhoY = len(pedacoAudioNovo)									

									# PRA TIRAR O DELTA E O DELTA DELTA, VOU PRECISAR O MFCC EM FORMA DE MATRIZ
									mfccMatriz = librosa.feature.mfcc(y=pedacoAudioNovo, sr=sr, n_mfcc=13, n_fft=tamanhoY)

									# ESSE ARRAY VAI CONTER SO AS FEATURES, SEM CLASSIFICACAO E SEM PASTA, EH COMO O 'X'
									xTestAtual = []
									xTestAtual += melGibson(mfccMatriz)
									#xTestAtual += delta1(mfccMatriz)
									#xTestAtual += delta2(mfccMatriz)
									xTestAtual.append(mediaRMS(pedacoAudioNovo, tamanhoY))
									xTestAtual.append(centroideEspectral(pedacoAudioNovo, sr, tamanhoY))
									xTestAtual.append(larguraEspectral(pedacoAudioNovo, sr, tamanhoY))
									xTestAtual.append(planicidadeEspectral(pedacoAudioNovo, tamanhoY))
									xTestAtual.append(rolloff(pedacoAudioNovo, sr, tamanhoY))
									xTestAtual.append(cruzamentosZero(pedacoAudioNovo, tamanhoY))
									xTestAtual.append(assimetria(pedacoAudioNovo))
									xTestAtual.append(curtose(pedacoAudioNovo))
									xTestAtual.append(variancia(pedacoAudioNovo))

									xTestCadaJanela.append(xTestAtual)

								#print("As features foram extraídas para cada janela.")
								# AQUI EU JA ACABEI DE SEGMENTAR O AUDIO ATUAL E EXTRAIR TODAS AS FEATURES
								# PRECISO NORMALIZAR E APAGAR AS COLUNAS DAS FEATURES QUE NAO ESTAO SENDO UTILIZADAS
								xTestCadaJanela = numpy.array(xTestCadaJanela)

								# NORMALIZANDO
								xTestCadaJanela = normalizarDataset(xTestCadaJanela, normalizador)

								# TIRANDO AS FEATURES QUE NÃO ESTÃO SENDO USADAS
								xTestCadaJanela = xTestCadaJanela.loc[:, xTestCadaJanela.columns.intersection(featuresSelecionadas + [featureAtual])]

								# ARRUMANDO A ORDEM DAS COLUNAS
								xTestCadaJanela = xTestCadaJanela[featuresSelecionadas + [featureAtual]]

								# ESSE AQUI SERA O ARRAY DA PREDICAO DE CADA JANELA DO AUDIO, VOU USA-LO PARA TIRAR A MODA
								inicioTeste = time.time()
								yPredCadaJanela = objClassificador.predict(xTestCadaJanela)

								# PREDICAO FINAL -> MODA DO YPREDCADAJANELA
								# DEPOIS DE FAZER A PREDICAO FINAL, COLOCO O RESULTADO EM YPREDICT PARA TIRAR A ACURACIA DEPOIS								
								yPredict.append(int(mode(yPredCadaJanela)[0]))
								yTest.append(dadoTeste['classe'])
								fimTeste = time.time()
								tempoTotalTeste += fimTeste - inicioTeste

								#print("Classificação finalizada. A moda elegeu a classe", yPredict[-1], "e o áudio pertencia a classe", yTest[-1])

						# AQUI EU JA TENHO OS RESULTADOS DA ITERACAO DO KFOLD, JA POSSO CALCULAR A ACURACIA DESSA ITERACAO DO KFOLD
						# COLOCANDO AS ACURACIAS NOAS ARRAYS
						arrayAcuraciasCadaKFold.append(accuracy_score(yTest, yPredict))						
					  
					# AQUI EU JA ACABEI TODAS AS ITERACOES DO KFOLD COM A COMBINACAO DE FEATURES ATUAL, POSSO TIRAR A MEDIA E SALVAR O RESULTADO
					# MEDIA DAS ACURACIAS DE CADA CLASSIFICADOR PARA ESSA FEATURE
					mediaAcc  = numpy.mean(arrayAcuraciasCadaKFold)
					
					# COLOCANDO AS INFORMACOES NOS CSVS
					# "qtd"  |  "combinacao_ftx" |  "acc"  |  "time_train"  |  "time_test"
					linhaCSV  = [len(featuresSelecionadas) + 1, featuresSelecionadas + [featureAtual], mediaAcc, tempoTotalTreino, tempoTotalTeste]

					writerCSVClassificador.writerow(linhaCSV)

					# Pfor pedaRINTANDO AS INFORMAÇÕES NO TERMINAL
					# são as mesmas informações, mas só com a feature atual
					print(len(featuresSelecionadas) + 1, featureAtual, "%.3f" % mediaAcc, "%.3f" % tempoTotalTreino, "%.3f" % tempoTotalTeste)

					# AGORA JA TENHO A MEDIA DE TODAS AS ACURACIAS DO KFOLD, VOU VER SE COM ESSA FEATURE O RESULTADO FOI MELHOR
					if mediaAcc > melhorAcuracia:
						melhorAcuracia = mediaAcc
						melhorFeature = featureAtual
						tempoTotalTreinoMelhorFeature = tempoTotalTreino
						tempoTotalTesteMelhorFeature = tempoTotalTeste

				# VERIFICANDO SE POSSO PARAR O WHILE COM A CONDICAO DE PARADA
				if(melhorAcuracia/acuraciaAnterior < 1.01):
					print("Critério de parada atingido")
					print("Melhor combinação de features:", featuresSelecionadas, "\n")
					break
				else:
					# ATUALIZACAO DE VARIAVEIS FORA DO WHILE
					acuraciaAnterior = melhorAcuracia

					# DEPOIS QUE ACABAR, EU REMOVO A MELHOR FEATURE DAS FEATURES RESTANTES E COLOCO ELA FIXA NO DATASET
					featuresRestantes, featuresSelecionadas = atualizarFeaturesSelecionadas(melhorFeature, featuresRestantes, featuresSelecionadas)

				# NO FIM DO WHILE EU PRINTO A COMBINAÇÃO DE FEATURES QUE FOI MELHOR E ESCREVO NO CSV REDUZIDO
				linhaCSV  = [len(featuresSelecionadas), featuresSelecionadas, melhorAcuracia, tempoTotalTreinoMelhorFeature, tempoTotalTesteMelhorFeature]
				writerCSVReduzido.writerow(linhaCSV)
				print("Melhor combinação de features:", featuresSelecionadas, "\n")

# ------------------------DEFININDO OS PARAMETROS INICIAIS------------------------

# CAMINHO PARA O CSV DE FEATURES
bitsProfundidade = sys.argv[1]
freqAmostragem = sys.argv[2]
caminhoCSV = "features_"  + bitsProfundidade+"bits_" + freqAmostragem + "kHz.csv"

datasetOriginal = pandas.read_csv(caminhoCSV)

# SALVANDO A COLUNA DE PASTAS, NOME DO ARQUIVO E CLASSIFICACAO
colunaPasta = datasetOriginal["pasta"]
colunaArquivo = datasetOriginal["arquivo"]
colunaClassificacao = datasetOriginal["classe"]

# DELETANDO A PRIMEIRA COLUNA (PASTA), A SEGUNDA (NOMES DOS ARQUIVOS) E A CLASSIFICACAO
datasetOriginal = datasetOriginal.drop(['pasta', 'arquivo', 'classe'], axis=1)

# NORMALIZANDO O DATASET
normalizador = sys.argv[3]
datasetOriginal = normalizarDataset(datasetOriginal, normalizador)

# COLOCANDO AS COLUNAS DE CLASSIFICACAO E PASTA
datasetOriginal.insert(0, "pasta", colunaPasta, True) 
datasetOriginal.insert(1, "arquivo", colunaArquivo, True) 
datasetOriginal.insert(datasetOriginal.shape[1], "classe", colunaClassificacao, True)

# CLASSIFICADOR DESEJADO
classificador = sys.argv[4]

# FRAME LENGTH E OVERLAP LENGTH
FRAME_TIME = int(sys.argv[5]) 	# milissegundos
OVERLAP_TIME = int(sys.argv[6]) 	# milissegundos
FRAME_LENGTH = int(freqAmostragem) * FRAME_TIME		# samples
OVERLAP_LENGTH = int(freqAmostragem) * OVERLAP_TIME	# samples

# RODANDO O CODIGO
main(datasetOriginal, bitsProfundidade, freqAmostragem, classificador, normalizador, FRAME_LENGTH, OVERLAP_LENGTH)
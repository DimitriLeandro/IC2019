'''
	PARA RODAR ESSE CÓDIGO:
	python3 analiseFeatures.py [bitsProfundidade] [freqAmostragem] [normalizador] [classificador]

	ONDE:
	bitsProfundidade -> 08 ou 16
	freqAmostragem -> 08, 16, 24, 32, 40 ou 48 
	normalizador -> l1, l2, scale, standardScaler, robustScaler ou 0
	classificador -> KNN, Tree ou SVM
'''

import numpy
import time
import csv
import sys
from sklearn.preprocessing import normalize, scale, StandardScaler, RobustScaler
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# ------------------------------ DEFINICAO DE FUNCOES --------------------
def kFold(dataset):

	# ESSA FUNCAO RETORNARA UMA MATRIZ COM AS SEGUINTES DIMENSOES
	# matriz[iteracaoKFold][colunaDeTreinoOuTeste][dadoPuro][feature]
	# iteracaoKFold -> vai de 0 a 9 e representa as 10 iteracoes do KFold
	# colunaDeTreinoOuTeste -> vai pegar os dados de treinamento (0) ou os de teste (1)
	# dadoPuro -> dado com varias features, a pasta e a classificacao correta
	# feature -> seleciona uma das 24 features do dado
	
	matriz = []
	
	for pastaTeste in range(1,11):

		rodada = []
		xTrain = []
		xTest  = []

		for dado in dataset:
			if int(dado[0]) == pastaTeste:
				xTest.append(dado)
			else:
				xTrain.append(dado)
		
		rodada.append(xTrain)
		rodada.append(xTest)
		
		matriz.append(rodada)
	
	return matriz

def rearranjarDataset(dataset, datasetOriginal, featureAtual, i):
	
	# ESSA FUNCAO ADICIONA UMA NOVA FEATURE AO DATASET
	
	# REMOVENDO A COLUNA DE CLASSIFICACAO
	dataset = dataset[:,0:-1]

	# SE NAO FOR A PRIMEIRA ITERACAO, EXCLUIR A ULTIMA FEATURE
	if(i != 0):
		dataset = dataset[:,0:-1]

	# ADICIONANDO A FEATURE ATUAL NO DATASET
	colunaNova = datasetOriginal[:,featureAtual]
	dataset = numpy.column_stack((dataset, colunaNova))

	# ADICIONANDO NOVAMENTE A CLASSIFICACAO
	colunaNova = datasetOriginal[:,datasetOriginal.shape[1] - 1]
	dataset = numpy.column_stack((dataset, colunaNova))
	
	return dataset

def novaMelhorFeature(dataset, datasetOriginal, melhorFeature, featuresRestantes):
	
	# ESSA FUNCAO SERVE PARA QUE, DEPOIS DE DEFINIDA A NOVA FEATURE QUE VAI ENTRAR NO DATASET PERMANENTEMENTE, 
	# REARRANJAR O DATASET E REMOVER ESSA FEATURE DO ARRAY DE FEATURES RESTANTES
	
	# REMOVENDO A COLUNA DE CLASSIFICACAO E A ULTIMA FEATURE UTILIZADA
	dataset = dataset[:,0:-2]

	# ADICIONANDO A FEATURE NOVA NO DATASET
	colunaNova = datasetOriginal[:,melhorFeature]
	dataset = numpy.column_stack((dataset, colunaNova))

	# ADICIONANDO NOVAMENTE A CLASSIFICACAO
	colunaNova = datasetOriginal[:,datasetOriginal.shape[1] - 1]
	dataset = numpy.column_stack((dataset, colunaNova))
	
	# REMOVENDO DO ARRAY DE FEATURES RESTANTES
	indexPraRemover = numpy.where(featuresRestantes == melhorFeature)
	featuresRestantes = numpy.delete(featuresRestantes, indexPraRemover)
		
	return dataset, featuresRestantes

def escreverCabecalho(writerCSVClassificador, writerCSVReduzido):
	cabecalho = ['qtd', 'ftx', 'acc', 'time_train', 'time_test']
	writerCSVClassificador.writerow(cabecalho)
	writerCSVReduzido.writerow(cabecalho)

def separarXeY(iteracaoKFold):
	# SEPARANDO OS DADOS DE TREINAMENTO E TESTE E JÁ EXCLUINDO A COLUNA 0 (pasta)
	dadosTreino = numpy.delete(iteracaoKFold[0], 0, axis=1)  
	dadosTeste  = numpy.delete(iteracaoKFold[1], 0, axis=1)

	# SEPARANDO O QUE E X E Y
	xTrain = numpy.delete(dadosTreino, dadosTreino.shape[1] - 1, axis=1) # exclui a coluna do target
	xTest  = numpy.delete(dadosTeste, dadosTeste.shape[1] - 1, axis=1) # exclui a coluna do target
	yTrain = numpy.delete(dadosTreino, numpy.s_[0:dadosTreino.shape[1] - 1], axis=1).ravel() # exclui as colunas menos a ultima
	yTest  = numpy.delete(dadosTeste, numpy.s_[0:dadosTeste.shape[1] - 1], axis=1).ravel() # exclui as colunas menos a ultima

	return xTrain, xTest, yTrain, yTest

def main(datasetOriginal, bitsProfundidade, freqAmostragem, classificador, normalizador):

	# DECLARACAO DE VARIAVEIS
	dataset = numpy.delete(datasetOriginal, numpy.s_[1:-1], axis=1)
	featuresRestantes = numpy.arange(1, datasetOriginal.shape[1] - 1)
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

					# REARRANJO O DATASET COM A FEATURE ATUAL
					dataset = rearranjarDataset(dataset, datasetOriginal, featureAtual, i)

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
					matrizKFold = kFold(dataset)
					for iteracaoKFold in matrizKFold:

						# SEPARANDO O QUE E X E O QUE E Y
						xTrain, xTest, yTrain, yTest = separarXeY(iteracaoKFold)

						# TREINANDO O CLASSIFICADOR
						inicioTreino = time.time()
						objClassificador  = objClassificador.fit(xTrain, yTrain)
						fimTreino = time.time()

						# PREDIZENDO
						inicioTeste = time.time()
						yPredict  = objClassificador.predict(xTest)
						fimTeste = time.time()

						# COLOCANDO AS ACURACIAS NOAS ARRAYS
						arrayAcuraciasCadaKFold.append(accuracy_score(yTest, yPredict))
						
						# SOMANDO NO TEMPO TOTAL DE TREINAMENTO E TESTE
						tempoTotalTreino += fimTreino - inicioTreino
						tempoTotalTeste += fimTeste - inicioTeste
					  
					# MEDIA DAS ACURACIAS DE CADA CLASSIFICADOR PARA ESSA FEATURE
					mediaAcc  = numpy.mean(arrayAcuraciasCadaKFold)
					
					# COLOCANDO AS INFORMACOES NOS CSVS
					# "qtd"  |  "combinacao_ftx" |  "acc"  |  "time_train"  |  "time_test"
					linhaCSV  = [len(featuresSelecionadas) + 1, featuresSelecionadas + [featureAtual], mediaAcc, tempoTotalTreino, tempoTotalTeste]

					writerCSVClassificador.writerow(linhaCSV)

					# PRINTANDO AS INFORMAÇÕES NO TERMINAL
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
					featuresSelecionadas.append(melhorFeature)
					dataset, featuresRestantes = novaMelhorFeature(dataset, datasetOriginal, melhorFeature, featuresRestantes)

				# NO FIM DO WHILE EU PRINTO A COMBINAÇÃO DE FEATURES QUE FOI MELHOR E ESCREVO NO CSV REDUZIDO
				linhaCSV  = [len(featuresSelecionadas), featuresSelecionadas, melhorAcuracia, tempoTotalTreinoMelhorFeature, tempoTotalTesteMelhorFeature]
				writerCSVReduzido.writerow(linhaCSV)
				print("Melhor combinação de features:", featuresSelecionadas, "\n")

# ------------------------DEFININDO OS PARAMETROS INICIAIS------------------------

# CAMINHO PARA O CSV DE FEATURES
bitsProfundidade = sys.argv[1]
freqAmostragem = sys.argv[2]
caminhoCSV = "features_"  + bitsProfundidade+"bits_" + freqAmostragem + "kHz.csv"

# ABRINDO O CSV DE FEATURES COMO UM NUMPY ARRAY E DELETANDO A PRIMEIRA LINHA (CABECALHO)
datasetOriginal = numpy.genfromtxt(caminhoCSV, delimiter=",")
datasetOriginal = numpy.delete(datasetOriginal, 0, 0)

# SALVANDO A COLUNA DE PASTAS E CLASSIFICACAO
colunaPasta = datasetOriginal[:,0]
colunaClassificacao = datasetOriginal[:,datasetOriginal.shape[1] - 1]

# DELETANDO A PRIMEIRA COLUNA (PASTA), A SEGUNDA (NOMES DOS ARQUIVOS) E A CLASSIFICACAO
datasetOriginal = datasetOriginal[:,2:-1]

# NORMALIZANDO O DATASET
if sys.argv[3] == 'l1':
	datasetOriginal = normalize(datasetOriginal, norm='l1')
elif sys.argv[3] == 'l2':
	datasetOriginal = normalize(datasetOriginal, norm='l2')
elif sys.argv[3] == 'scale':
	datasetOriginal = scale(datasetOriginal)
elif sys.argv[3] == 'standardScaler':
	datasetOriginal = StandardScaler().fit_transform(datasetOriginal)
elif sys.argv[3] == 'robustScaler':
	datasetOriginal = RobustScaler().fit_transform(datasetOriginal)
else:
	pass

# COLOCANDO AS COLUNAS DE CLASSIFICACAO E PASTA
datasetOriginal = numpy.column_stack((colunaPasta, datasetOriginal))
datasetOriginal = numpy.column_stack((datasetOriginal, colunaClassificacao))

# CLASSIFICADOR DESEJADO
classificador = sys.argv[4]

# RODANDO O CODIGO
main(datasetOriginal, bitsProfundidade, freqAmostragem, classificador, sys.argv[3])
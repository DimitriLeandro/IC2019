import numpy
import time
import csv
import sys
from sklearn.preprocessing import normalize, scale
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

def escreverCabecalho(writerKNN, writerTree, writerSVM):
	cabecalho = ['qtd', 'ftx', 'acc', 'time_train', 'time_test']
	writerKNN.writerow(cabecalho)
	writerTree.writerow(cabecalho)
	writerSVM.writerow(cabecalho)

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

def main(datasetOriginal, bitsProfundidade, freqAmostragem):

	# DECLARACAO DE VARIAVEIS
	dataset = numpy.delete(datasetOriginal, numpy.s_[1:-1], axis=1)
	featuresRestantes = numpy.arange(1, datasetOriginal.shape[1] - 1)
	featuresSelecionadas = []
	acuraciasGeral = []
	acuraciaAnterior = 0.001
	contWhile = 0
	arquivoCSVKNN = "KNN_" + bitsProfundidade + "bits_" + freqAmostragem + "kHz.csv"
	arquivoCSVTree = "Tree_" + bitsProfundidade + "bits_" + freqAmostragem + "kHz.csv"
	arquivoCSVSVM = "SVM_" + bitsProfundidade + "bits_" + freqAmostragem + "kHz.csv"

	# ABRINDO OS CSVS DOS RESULTADOS DOS CLASSIFICADORES
	with open(arquivoCSVKNN, 'a') as csvFileKNN:
		with open(arquivoCSVTree, 'a') as csvFileTree:
			with open(arquivoCSVSVM, 'a') as csvFileSVM:

				# CRIANDO OS OBJETOS RESPONSAVEIS POR ESCREVER LINHAS NO CSV
				writerKNN = csv.writer(csvFileKNN)
				writerTree = csv.writer(csvFileTree)
				writerSVM = csv.writer(csvFileSVM)

				#ESCREVENDO O CABECALHO DOS CSVS DOS CLASSIFICADORES
				escreverCabecalho(writerKNN, writerTree, writerSVM)				
			  
				# ENQUANTO HOUVEREM FEATURES A SEREM TESTADAS
				while(len(featuresRestantes) != 0):
				
					contWhile += 1
					print("INICIO:", contWhile)

					melhorAcuracia = 0
					melhorFeature = 1

					# PARA CADA FEATURE RESTANTE EU A ADICIONO NAS QUE JA ESTAO SENDO UTILIZADAS
					for i, featureAtual in enumerate(featuresRestantes):

						# REARRANJO O DATASET COM A FEATURE ATUAL
						dataset = rearranjarDataset(dataset, datasetOriginal, featureAtual, i)

						# AGORA QUE JA TENHO UM DATASET COM AS FEATURES DA VEZ, TENHO QUE CRIAR OS CLASSIFICADORES
						knn = KNeighborsClassifier(3)
						tree = DecisionTreeClassifier()
						svm = SVC(gamma='scale', decision_function_shape='ovo')
						
						# ESSES ARRAYS VAO GUARDAR AS ACURACIAS DE CADA ITERACAO DO KFOLD, MAS SERAO RESETADOS QUANDO UMA NOVA FEATURE FOR TESTADA
						arrayAcuraciasCadaKFoldKNN = []
						arrayAcuraciasCadaKFoldTree = []
						arrayAcuraciasCadaKFoldSVM = []						

						# TEMPO DE TREINAMENTO DAS 10 ITERACOES DO KFOLD
						tempoTotalTreinoKNN = 0
						tempoTotalTreinoTree = 0
						tempoTotalTreinoSVM = 0	
						tempoTotalTesteKNN = 0
						tempoTotalTesteTree = 0
						tempoTotalTesteSVM = 0

						# PARA CADA ITERACAO DO KFOLD
						matrizKFold = kFold(dataset)
						for iteracaoKFold in matrizKFold:

							# SEPARANDO O QUE E X E O QUE E Y
							xTrain, xTest, yTrain, yTest = separarXeY(iteracaoKFold)

							# TREINANDO OS CLASSIFICADORES
							inicioTreinoKNN = time.time()
							knn  = knn.fit(xTrain, yTrain)
							fimTreinoKNN = time.time()

							inicioTreinoTree = time.time()
							tree = tree.fit(xTrain, yTrain)
							fimTreinoTree = time.time()

							inicioTreinoSVM = time.time()
							svm  = svm.fit(xTrain, yTrain)
							fimTreinoSVM = time.time()

							# PREDIZENDO
							inicioTesteKNN = time.time()
							yKNN  = knn.predict(xTest)
							fimTesteKNN = time.time()

							inicioTesteTree = time.time()
							yTree = tree.predict(xTest)
							fimTesteTree = time.time()

							inicioTesteSVM = time.time()
							ySVM  = svm.predict(xTest)
							fimTesteSVM = time.time()
						  
							# COLOCANDO AS ACURACIAS NOAS ARRAYS
							arrayAcuraciasCadaKFoldKNN.append(accuracy_score(yTest, yKNN))
							arrayAcuraciasCadaKFoldTree.append(accuracy_score(yTest, yTree))
							arrayAcuraciasCadaKFoldSVM.append(accuracy_score(yTest, ySVM))

							# SOMANDO NO TEMPO TOTAL DE TREINAMENTO E TESTE
							tempoTotalTreinoKNN += fimTreinoKNN - inicioTreinoKNN
							tempoTotalTreinoTree += fimTreinoTree - inicioTreinoTree
							tempoTotalTreinoSVM += fimTreinoSVM - inicioTreinoSVM
							tempoTotalTesteKNN += fimTesteKNN - inicioTesteKNN
							tempoTotalTesteTree += fimTesteTree - inicioTesteTree
							tempoTotalTesteSVM += fimTesteSVM - inicioTesteSVM
						  
						# MEDIA DAS ACURACIAS DE CADA CLASSIFICADOR PARA ESSA FEATURE
						mediaKNN  = numpy.mean(arrayAcuraciasCadaKFoldKNN)
						mediaTree = numpy.mean(arrayAcuraciasCadaKFoldTree)
						mediaSVM  = numpy.mean(arrayAcuraciasCadaKFoldSVM)
						
						# MEDIA ENTRE TODOS OS CLASSIFICADORESS
						mediaAcuracias = (mediaKNN + mediaTree + mediaSVM)/3

						# COLOCANDO AS INFORMACOES NOS CSVS
						# "qtd"  |  "combinacao_ftx" |  "acc"  |  "time_train"  |  "time_test"
						linhaCSVKNN  = [len(featuresSelecionadas) + 1, featuresSelecionadas + [featureAtual], mediaKNN, tempoTotalTreinoKNN, tempoTotalTesteKNN]
						linhaCSVTree = [len(featuresSelecionadas) + 1, featuresSelecionadas + [featureAtual], mediaTree, tempoTotalTreinoTree, tempoTotalTesteTree]
						linhaCSVSVM  = [len(featuresSelecionadas) + 1, featuresSelecionadas + [featureAtual], mediaSVM, tempoTotalTreinoSVM, tempoTotalTesteSVM]

						writerKNN.writerow(linhaCSVKNN)
						writerTree.writerow(linhaCSVTree)
						writerSVM.writerow(linhaCSVSVM)

						# PRINTANDO AS INFORMAÇÕES NO TERMINAL
						# são as mesmas informações, mas só com a feature atual
						print("KNN:\t", len(featuresSelecionadas) + 1, featureAtual, "%.3f" % mediaKNN, "%.3f" % tempoTotalTreinoKNN, "%.3f" % tempoTotalTesteKNN)
						print("Tree:\t", len(featuresSelecionadas) + 1, featureAtual, "%.3f" % mediaTree, "%.3f" % tempoTotalTreinoTree, "%.3f" % tempoTotalTesteTree)
						print("SVM:\t", len(featuresSelecionadas) + 1, featureAtual, "%.3f" % mediaSVM, "%.3f" % tempoTotalTreinoSVM, "%.3f" % tempoTotalTesteSVM)

						# AGORA JA TENHO A MEDIA DE TODAS AS ACURACIAS DO KFOLD, VOU VER SE COM ESSA FEATURE O RESULTADO FOI MELHOR
						if mediaAcuracias > melhorAcuracia:
							melhorAcuracia = mediaAcuracias
							melhorFeature = featureAtual          

					# VERIFICANDO SE POSSO PARAR O WHILE COM A CONDICAO DE PARADA
					if(melhorAcuracia/acuraciaAnterior < 1.01):
						break
					else:
						# ATUALIZACAO DE VARIAVEIS FORA DO WHILE
						acuraciaAnterior = melhorAcuracia
						melhorAcuraciaGeralzona = melhorAcuracia

						# DEPOIS QUE ACABAR, EU REMOVO A MELHOR FEATURE DAS FEATURES RESTANTES E COLOCO ELA FIXA NO DATASET
						featuresSelecionadas.append(melhorFeature)
						dataset, featuresRestantes = novaMelhorFeature(dataset, datasetOriginal, melhorFeature, featuresRestantes)
						
						# TAMBEM TENHO QUE COLOCAR A ACURACIA NO ARRAY DE ACURACIAS GERAIS
						acuraciasGeral.append(melhorAcuracia)

					# NO FIM DO WHILE EU PRINTO A COMBINAÇÃO DE FEATURES QUE FOI MELHOR
					print("Melhor combinação de features:", featuresSelecionadas, "\n")

# ------------------------DEFININDO OS PARAMETROS INICIAIS------------------------

# CAMINHO PARA O CSV DE FEATURES
bitsProfundidade = sys.argv[1]
freqAmostragem = sys.argv[2]
caminhoCSV = "features_"  + bitsProfundidade+"bits_" + freqAmostragem + "kHz.csv"

# ABRINDO O CSV COMO UM NUMPY ARRAY E DELETANDO A PRIMEIRA LINHA (CABECALHO)
datasetOriginal = numpy.genfromtxt(caminhoCSV, delimiter=",")
datasetOriginal = numpy.delete(datasetOriginal, 0, 0)

# SALVANDO A COLUNA DE PASTAS E CLASSIFICACAO
colunaPasta = datasetOriginal[:,0]
colunaClassificacao = datasetOriginal[:,datasetOriginal.shape[1] - 1]

# DELETANDO A PRIMEIRA COLUNA (PASTA), A SEGUNDA (NOMES DOS ARQUIVOS) E A CLASSIFICACAO
datasetOriginal = datasetOriginal[:,2:-1]

# NORMALIZANDO O DATASET
datasetOriginal = normalize(datasetOriginal, norm='l1')
#datasetOriginal = normalize(datasetOriginal, norm='l2')
#datasetOriginal = scale(datasetOriginal)

# COLOCANDO AS COLUNAS DE CLASSIFICACAO E PASTA
datasetOriginal = numpy.column_stack((colunaPasta, datasetOriginal))
datasetOriginal = numpy.column_stack((datasetOriginal, colunaClassificacao))

# RODANDO O CODIGO
main(datasetOriginal, bitsProfundidade, freqAmostragem)
import numpy
import time
import csv
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
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

def escreverCabecalho(writerKNN, writerTree, writerSVM):
	cabecalho = ['qtdDimensoes', 'acc', 'time_train', 'time_test']
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

def main(featuresOriginais, colunaPasta, colunaClassificacao, bitsProfundidade, freqAmostragem):

	# DECLARACAO DE VARIAVEIS
	arquivoCSVKNN = "PCA_KNN_" + bitsProfundidade + "bits_" + freqAmostragem + "kHz.csv"
	arquivoCSVTree = "PCA_Tree_" + bitsProfundidade + "bits_" + freqAmostragem + "kHz.csv"
	arquivoCSVSVM = "PCA_SVM_" + bitsProfundidade + "bits_" + freqAmostragem + "kHz.csv"

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

				# DE 1 DIMENSAO ATE O TOTAL DE FEATURES UTILIZADAS
				for nDimensoes in range(1, len(datasetOriginal[0]) - 2):
					
					# PRECISO APLICAR A REDUCAO DE DIMENSIONALIDADE NA MATRIZ 
					# DE FEATURES E DEPOIS JUNTAR COM AS COLUNAS DE PASTA E CLASSIFICACAO
					featuresPCA = PCA(n_components=nDimensoes).fit_transform(featuresOriginais)
					dataset = numpy.column_stack((colunaPasta, featuresPCA))
					dataset = numpy.column_stack((dataset, colunaClassificacao))

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

					# COLOCANDO AS INFORMACOES NOS CSVS
					# "qtdDimensoes"  |  "acc"  |  "time_train"  |  "time_test"
					linhaCSVKNN  = [nDimensoes, mediaKNN, tempoTotalTreinoKNN, tempoTotalTesteKNN]
					linhaCSVTree = [nDimensoes, mediaTree, tempoTotalTreinoTree, tempoTotalTesteTree]
					linhaCSVSVM  = [nDimensoes, mediaSVM, tempoTotalTreinoSVM, tempoTotalTesteSVM]

					writerKNN.writerow(linhaCSVKNN)
					writerTree.writerow(linhaCSVTree)
					writerSVM.writerow(linhaCSVSVM)

					# PRINTANDO AS INFORMAÇÕES NO TERMINAL
					# são as mesmas informações, mas só com a feature atual
					print("KNN:\t", nDimensoes, "%.3f" % mediaKNN, "%.3f" % tempoTotalTreinoKNN, "%.3f" % tempoTotalTesteKNN)
					print("Tree:\t", nDimensoes, "%.3f" % mediaTree, "%.3f" % tempoTotalTreinoTree, "%.3f" % tempoTotalTesteTree)
					print("SVM:\t", nDimensoes, "%.3f" % mediaSVM, "%.3f" % tempoTotalTreinoSVM, "%.3f" % tempoTotalTesteSVM)
					print("\n")

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
featuresOriginais = datasetOriginal[:,2:-1]

# NORMALIZANDO O DATASET
#datasetOriginal = normalize(datasetOriginal, norm='l1')
#datasetOriginal = normalize(datasetOriginal, norm='l2')
#datasetOriginal = scale(datasetOriginal)

# COLOCANDO AS COLUNAS DE CLASSIFICACAO E PASTA
#datasetOriginal = numpy.column_stack((colunaPasta, datasetOriginal))
#datasetOriginal = numpy.column_stack((datasetOriginal, colunaClassificacao))

# RODANDO O CODIGO
main(featuresOriginais, colunaPasta, colunaClassificacao, bitsProfundidade, freqAmostragem)
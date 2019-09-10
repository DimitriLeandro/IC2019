import numpy
import time
import csv
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

def main(datasetOriginal, bitsProfundidade, freqAmostragem):
	# O DATASET COMECA SO COM A COLUNA 0 (PASTA) E A 23 (CLASSIFICACAO)
	dataset = numpy.delete(datasetOriginal, numpy.s_[1:-1], axis=1)

	# ARRAY DE FEATURES QUE FALTAM, 0 E A PASTA E -1 E A CLASSIFICACAO, POR ISSO NAO ENTRAM
	featuresRestantes = numpy.arange(1, datasetOriginal.shape[1] - 1)
	featuresSelecionadas = []

	# VARIAVEIS DE FORA DO WHILE
	acuraciasGeral = []
	acuraciaAnterior = 0.001
	contWhile = 0
	contIteracoesTotal = 0

	# OBJETOS DE CSVs
	now = datetime.now()
	
	arquivoCSVKNN = "KNN" + bitsProfundidade + freqAmostragem + ".csv"
	arquivoCSVTree = "Tree" + bitsProfundidade + freqAmostragem + ".csv"
	arquivoCSVSVM = "SVM" + bitsProfundidade + freqAmostragem + ".csv"

	cabecalho = ['iteracao', 'qtd features', 'features utilizadas', 'acuracia media kfold', 'tempo de treinamento']

	with open(arquivoCSVKNN, 'a') as csvFileKNN:
		with open(arquivoCSVTree, 'a') as csvFileTree:
			with open(arquivoCSVSVM, 'a') as csvFileSVM:

				writerKNN = csv.writer(csvFileKNN)
				writerKNN.writerow(cabecalho)

				writerTree = csv.writer(csvFileTree)
				writerTree.writerow(cabecalho)

				writerSVM = csv.writer(csvFileSVM)
				writerSVM.writerow(cabecalho)
			  
				# ENQUANTO HOUVEREM FEATURES A SEREM TESTADAS
				while(len(featuresRestantes) != 0):
				
					contWhile += 1
					print("NOVA ITERAÇÃO DO WHILE", contWhile)

					melhorAcuracia = 0
					melhorFeature = 1

					# PARA CADA FEATURE RESTANTE EU A ADICIONO NAS QUE JA ESTAO SENDO UTILIZADAS
					for i, featureAtual in enumerate(featuresRestantes):
					
						print("Testando feature", featureAtual)

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

						# PARA CADA ITERACAO DO KFOLD
						matrizKFold = kFold(dataset)

						# TEMPO DE TREINAMENTO DAS 10 ITERACOES DO KFOLD
						tempoTotalKNN = 0
						tempoTotalTree = 0
						tempoTotalSVM = 0
							
						for iteracaoKFold in matrizKFold:

							# SEPARANDO OS DADOS DE TREINAMENTO E TESTE E JÁ EXCLUINDO A COLUNA 0 (pasta)
							dadosTreino = numpy.delete(iteracaoKFold[0], 0, axis=1)  
							dadosTeste  = numpy.delete(iteracaoKFold[1], 0, axis=1)

							# SEPARANDO O QUE E X E Y
							xTrain = numpy.delete(dadosTreino, dadosTreino.shape[1] - 1, axis=1) # exclui a coluna do target
							xTest  = numpy.delete(dadosTeste, dadosTeste.shape[1] - 1, axis=1) # exclui a coluna do target
							yTrain = numpy.delete(dadosTreino, numpy.s_[0:dadosTreino.shape[1] - 1], axis=1).ravel() # exclui as colunas menos a ultima
							yTest  = numpy.delete(dadosTeste, numpy.s_[0:dadosTeste.shape[1] - 1], axis=1).ravel() # exclui as colunas menos a ultima

							# TREINANDO OS CLASSIFICADORES
							inicioKNN = time.time()
							knn  = knn.fit(xTrain, yTrain)
							fimKNN = time.time()

							inicioTree = time.time()
							tree = tree.fit(xTrain, yTrain)
							fimTree = time.time()

							inicioSVM = time.time()
							svm  = svm.fit(xTrain, yTrain)
							fimSVM = time.time()

							# PREDIZENDO
							yKNN  = knn.predict(xTest)
							yTree = tree.predict(xTest)
							ySVM  = svm.predict(xTest)
						  
							# COLOCANDO AS ACURACIAS NOAS ARRAYS
							arrayAcuraciasCadaKFoldKNN.append(accuracy_score(yTest, yKNN))
							arrayAcuraciasCadaKFoldTree.append(accuracy_score(yTest, yTree))
							arrayAcuraciasCadaKFoldSVM.append(accuracy_score(yTest, ySVM))

							# SOMANDO NO TEMPO TOTAL DE TREINAMENTO
							tempoTotalKNN += fimKNN - inicioKNN
							tempoTotalTree += fimTree - inicioTree
							tempoTotalSVM += fimSVM - inicioSVM
						  
						# MEDIA DAS ACURACIAS DE CADA CLASSIFICADOR PARA ESSA FEATURE
						mediaKNN  = numpy.mean(arrayAcuraciasCadaKFoldKNN)
						mediaTree = numpy.mean(arrayAcuraciasCadaKFoldTree)
						mediaSVM  = numpy.mean(arrayAcuraciasCadaKFoldSVM)
						
						# MEDIA ENTRE TODOS OS CLASSIFICADORESS
						mediaAcuracias = (mediaKNN + mediaTree + mediaSVM)/3

						# COLOCANDO AS INFORMACOES NOS CSVS
						# iteracao, qtdFeatures, features utilizadas, acuracia, tempo
						contIteracoesTotal += 1
						linhaCSVKNN = [contIteracoesTotal, len(featuresSelecionadas) + 1, featuresSelecionadas + [featureAtual], mediaKNN, tempoTotalKNN]
						linhaCSVTree = [contIteracoesTotal, len(featuresSelecionadas) + 1, featuresSelecionadas + [featureAtual], mediaTree, tempoTotalTree]
						linhaCSVSVM = [contIteracoesTotal, len(featuresSelecionadas) + 1, featuresSelecionadas + [featureAtual], mediaSVM, tempoTotalSVM]

						writerKNN.writerow(linhaCSVKNN)
						writerTree.writerow(linhaCSVTree)
						writerSVM.writerow(linhaCSVSVM)

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

# ------------------------DEFININDO OS PARAMETROS INICIAIS------------------------

# CAMINHO PARA O CSV
bitsProfundidade = "8bits"
freqAmostragem = "16k"
caminhoCSV = "conversoes/"+bitsProfundidade+"/"+freqAmostragem+"/features_"+bitsProfundidade+"_"+freqAmostragem+".csv"

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
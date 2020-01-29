'''
	ESSE CODIGO VAI SERVIR PARA VERIFICAR SE O JANELAMENTO DO DATASET PRODUZ MELHORES RESULTADOS
	OS AUDIOS DE TREINAMENTO E TESTE JA DEVEM VIR SEPARADOS NO CSV, PORTANTO, ESSE CODIGO NAO VAI ABRIR ARQUIVOS WAV, EXTRAIR FEATURES NEM NADA DISSO


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
	datasetTreino = dataset.loc[dataset['pasta'] != pastaTeste]

	xTrain = datasetTreino.drop(['pasta', 'arquivo', 'classe'], axis=1) # TUDO MENOS AS COLUNAS 'PASTA', 'ARQUIVO' E 'CLASSE'
	yTrain = datasetTreino['classe'] # APENAS A COLUNA 'CLASSE'	

	return xTrain, yTrain

def selecionarDadosTeste(pastaTeste, dataset):
	# ESSA FUNCAO VAI RETORNAR TODAS AS LINHAS COM NOME DO ARQUIVO, FEATURES SENDO UTILIZADAS E CLASSIFICACAO
	# PQ DEPOIS EU AINDA TENHO QUE VERIFICAR QUAIS LINHAS SAO DE UM MESMO AUDIO PRA TIRAR A MODA DAS CLASSIFICACOES

	# SELECIONANDO TODAS AS LINHAS QUE SEJAM DA PASTA TESTE
	datasetTeste = dataset.loc[dataset['pasta'] == pastaTeste]

	nomesArquivos = datasetTeste['arquivo']
	xTest = datasetTeste.drop(['pasta', 'arquivo', 'classe'], axis=1)
	yTest = datasetTeste['classe']

	return nomesArquivos, xTest, yTest

def atualizarFeaturesSelecionadas(melhorFeature, featuresRestantes, featuresSelecionadas):
	featuresSelecionadas.append(melhorFeature)
	indexPraRemover = numpy.where(featuresRestantes == melhorFeature)
	featuresRestantes = numpy.delete(featuresRestantes, indexPraRemover)

	return featuresRestantes, featuresSelecionadas

def acuraciaPelaModa(nomesArquivosCadaJanela, yTestCadaJanela, yPredCadaJanela):
	# ESSA FUNCAO VAI PEGAR OS ARRAYS QUE TEM AS INFORMACOES SOBRE CADA JANELA DOS AUDIOS DE TESTE E VAI TIRANDO A MODA DAS CLASSIFICACOES DE UM MESMO AUDIO
	# AI VAI VENDO SE ACERTOU OU NAO E VAI CALCULANDO A ACURACIA

	# nomesArquivosCadaJanela -> <class 'pandas.core.series.Series'>
	# yTestCadaJanela -> <class 'pandas.core.series.Series'>
	# yPredCadaJanela -> <class 'numpy.ndarray'>
	yPredCadaJanela = pandas.DataFrame(yPredCadaJanela)

	# CRIANDO OS ARRAYS QUE SERAO UTILIZADOS NO FINAL. CADA LINHA DELES REPESENTA UM ARQUIVO, YTESTFINAL COM A CLASSIFICACAO REAL E YPREDFINAL COM A MODA
	yTestFinal = []
	yPredFinal = []	

	# PEGANDO OS NOMES DOS ARQUIVOS SEM REPETICAO  E TAMBEM SUAS CLASSIFICACOES CORRETAS
	nomesArquivosSemRepeticao = nomesArquivosCadaJanela.unique()

	# PRECISO RESETAR OS INDEXES DESSES ARRAYS PRA CONSEGUIR CONCATENAR DEPOIS, SENAO VAO APARECER VARIOS NaN
	nomesArquivosCadaJanela.reset_index(drop=True, inplace=True)
	yTestCadaJanela.reset_index(drop=True, inplace=True)
	yPredCadaJanela.reset_index(drop=True, inplace=True)

	# CONCATENANDO AS TRES COLUNAS PRA FAZER UMA MATRIZ
	matrizDadosTeste = pandas.concat([nomesArquivosCadaJanela, yPredCadaJanela, yTestCadaJanela], axis=1)

	#print(matrizDadosTeste['classe'])

	# PARA CADA ARQUIVO, SELECIONO TODAS AS LINHAS DESSA MATRIZ matrizDadosTeste QUE SEJAM DESSE ARQUIVO E FAÇO A CLASSIFICACAO
	for arquivoAtual in nomesArquivosSemRepeticao:

		# SELECIONANDO TODAS AS PREDICOES QUE SEJAM DO ARQUVIO ATUAL
		predicoesArquivoAtual = matrizDadosTeste.loc[matrizDadosTeste['arquivo'] == arquivoAtual][0] # O NOME DA COLUNA DE PREDICOES EH 0 MSM

		# COLOCANDO NOS ARRAYS FINAIS
		yPredFinal.append(int(mode(predicoesArquivoAtual)[0]))
		yTestFinal.append(matrizDadosTeste['classe'][0])

		# APAGANDO AS LINHAS DO DATASET QUE SEJAM DO ARQUIVO ATUAL, JA QUE ELE JA FOI USADO
		matrizDadosTeste = matrizDadosTeste.loc[matrizDadosTeste['arquivo'] != arquivoAtual]
		matrizDadosTeste.reset_index(drop=True, inplace=True)

	# AGORA QUE OS ARRAYS JA ESTAO PRONTOS, POSSO APLICAR A ACURACIA
	return accuracy_score(yTestFinal, yPredFinal)


def main(datasetOriginal, bitsProfundidade, freqAmostragem, classificador, normalizador):

	# DECLARACAO DE VARIAVEIS
	dataset = datasetOriginal.loc[:, datasetOriginal.columns.intersection(['pasta','arquivo', 'classe'])] #o dataset vai comecar so com as colunas "pasta", "arquivo" e "classe"
	featuresRestantes = numpy.arange(0, datasetOriginal.shape[1] - 3) # do jeito que o datasetOriginal vem, as colunas estão nomeadas de acordo com featuresRestantes, sucesso
	featuresSelecionadas = []
	acuraciaAnterior = 0.001
	contWhile = 0
	arquivoCSV = classificador + "_" + normalizador + "_" + bitsProfundidade + "bits_" + freqAmostragem + "kHz.csv"
	arquivoCSVReduzido = classificador + "_" + normalizador + "_" + bitsProfundidade + "bits_" + freqAmostragem + "kHz_REDUZIDO.csv"

	# ESSA PARTE E PRA ADICIONAR FEATURES ANTES DO ALGORTIMO COMECAR, PQ JA TINHA RODADO EM OUTRO PC
	# SE FOR RODAR ESSE ALGORITMO DO ZERO, ESSA PARTE DEVE SER EXCLUIDA
	#featuresPrevias = [3, 33, 0, 38, 30, 45, 7, 35, 9, 24, 20]
	#for	melhorFeature in featuresPrevias:
	#	featuresRestantes, featuresSelecionadas = atualizarFeaturesSelecionadas(melhorFeature, featuresRestantes, featuresSelecionadas)

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

						# AGORA TENHO QUE PEGAR TODOS OS DADOS DE TESTE E CLASSIFICAR UM POR UM
						nomesArquivosCadaJanela, xTestCadaJanela, yTestCadaJanela = selecionarDadosTeste(pastaTeste, dataset)
						yPredCadaJanela = objClassificador.predict(xTestCadaJanela)

						# AQUI EU JA TENHO OS RESULTADOS DA ITERACAO DO KFOLD ATUAL, JA POSSO CALCULAR A ACURACIA DESSA ITERACAO DO KFOLD
						# COLOCANDO AS ACURACIAS NOAS ARRAYS

						inicioTeste = time.time()
						arrayAcuraciasCadaKFold.append(acuraciaPelaModa(nomesArquivosCadaJanela, yTestCadaJanela, yPredCadaJanela))
						fimTeste = time.time()
						tempoTotalTeste += fimTeste - inicioTeste				
					  
					# AQUI EU JA ACABEI TODAS AS ITERACOES DO KFOLD COM A COMBINACAO DE FEATURES ATUAL, POSSO TIRAR A MEDIA E SALVAR O RESULTADO
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
				# if(melhorAcuracia/acuraciaAnterior < 1.01):
				if(1==0):
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

# RODANDO O CODIGO
main(datasetOriginal, bitsProfundidade, freqAmostragem, classificador, normalizador)
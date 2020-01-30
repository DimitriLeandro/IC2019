#A classe deverá ser capaz de receber o caminho de um CSV com dados de TREINO, treinar os classificadores e classificar os dados de outros CSVs de TESTE.
#A ideia é criar uma função para receber um único CSV de treino para treinar os classificadores, e eles deverão ficar armazenados na classe, afinal, se a cada CSV de teste eu for fazer o treino antes, o processamento vai ficar pesado. Muito melhor fazer o treinamento uma única vez e depois usar os objetos das classes dos classificadores para testar outros CSVs.

# Toda a implementação da classe está muito bem descrita no jupyter "Implementacao de classe para treinamento e classificacao de CSVs"

# Usando a classe
# 1) Crie o caminho para o CSV de TREINAMENTO e abra-o como um pandas dataframe;
# 2) Utilizando o dataframe de treinamento aberto, crie e treine os classificadores desejados;
# 3) Crie o caminho para o CSV de TESTE, setando as váriaveis diretorio e nomeDataset;
# 4) Classifique o dataset de TESTE aberto, especificando onde o JSON do relátorio deve ser salvo
# Os passos 3 e 4 podem ser repetidos caso se deseje classificar vários datasets de TESTE.

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode 
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import time
import json

class TreinarEClassificar:

	def separarDataframeXeY(self, dataframe):
		# Função para separar o dataframe em X e Y
		# Preciso de uma função que pegue o dataframe devolva dois arrays: xTrain e yTrain. Ai depois é só usar esses arrays para treinar os classificadores.

		# ESSA FUNÇÃO TAMBÉM SERÁ USADA NA PARTE DO TESTE

		# PRIMEIRO, MONTO O X (POSSO EXCLUIR A PRIMEIRA COLUNA "ARQUIVOS" E A ÚLTIMA "CLASSE")
		x = dataframe[dataframe.columns[1:-1]]
		
		# AGORA, MONTO O Y
		y = dataframe[dataframe.columns[-1]]
		
		# CONVERTO TUDO PRA TUPLA NORMAL 
		x = x.values.tolist()
		y = y.values.tolist()
		
		return x, y

	def instanciarClassificadores(self, arrayStringsClassesClassificadores):
		# Função para instanciar classificadores
		# Como eu não sei se vou usar mais classificadores depois, então vou criar um array de classificadores, assim na hora de treinar eu faço um for loop e vou treinando tudo o que tiver dentro.

		# Essa função recebe apenas o nome das classes a serem instanciadas. Por exemplo, envie ["SGDClassifier(params)", "BaggingClassifier(params)"] para receber um array de objetos dessas classes.
		arrayObjClassificadores = []
		
		for classe in arrayStringsClassesClassificadores:
			print("Instanciando um objeto da classe", classe)
			arrayObjClassificadores.append(eval(classe))
			
		return arrayObjClassificadores

	def treinarClassificadores(self, xTrain, yTrain, arrayObjClassificadores):
		# Função para treinar os classificadores
		# Essa função vai receber xTrain, yTrain e o array de classificadores e devolver o array com os classificadores já treinados.
		for i, objClassificador in enumerate(arrayObjClassificadores):
			print("Treinando o", objClassificador.__class__.__name__)
			tempoInicio = time.time()
			arrayObjClassificadores[i].fit(xTrain, yTrain)
			tempoFim = time.time()
			print("Tempo de treinamento do", objClassificador.__class__.__name__, "(segundos):", tempoFim-tempoInicio)
			
		return arrayObjClassificadores

	def criarETreinarClassificadores(self, dataframeTreino, arrayStringsClassesClassificadores):
		# Função para unir as três funções anteriores
		# Essa função vai unir as três funções que foram criadas. Ela vai receber o dataframe de treino e um array de strings contendo os classificadores desejados.
		# Em primeiro lugar, ela vai usar a função separarDataframeXeY para gerar xTrain e yTrain. Depois, vai usar instanciarClassificadores para gerar um array com os classificadores desejados. Por fim, vai treiná-los usando treinarClassificadores().
		# Ela retorna o array de classificadores com todos eles já treinados e prontos para serem usados para predizer dados.

		print("Começando o treinamento dos classificadores")
		xTrain, yTrain          = self.separarDataframeXeY(dataframeTreino)
		arrayObjClassificadores = self.instanciarClassificadores(arrayStringsClassesClassificadores)
		arrayObjClassificadores = self.treinarClassificadores(xTrain, yTrain, arrayObjClassificadores)
		print("Classificadores treinados")
		
		return arrayObjClassificadores

	def calcularModa(self, yPredCadaJanela):
		# Função para calcular a moda das classificações de um único áudio
		# Cada janela de um único áudio será classificada. Depois, para dar um veredito para o áudio como um todo, precisamos tirar a moda das classificações.
		return mode(yPredCadaJanela)[0][0]

	def predizerUnicoAudio(self, xTest, classificador):
		# Função para para predizer um conjunto de dados xTest
		# Essa função receberá uma matriz contendo apenas features de um único áudio e um classificador. Ela vai classificar todas as janelas desse áudio e utilizar a função da moda para dar um veredito sobre esse único áudio.
		# A matriz já vai ter que vir arrumada para essa função. As funções que lidam com o pandas para deixar tudo bonitinho vão vir depois.
		yPredCadaJanela = classificador.predict(xTest).tolist()
		return self.calcularModa(yPredCadaJanela)

	def obterNomesArquivos(self, dataframe):
		# Função para criar um array apenas com os nomes dos arquivos
		# Vou precisar de um array que contenha o nome de cada arquivo de um CSV de teste, sem repetições. Isso será necessário para criar um dataframe contendo apenas linhas que digam respeito às janelas de um único áudio.
		return dataframe[dataframe.columns[0]].unique().tolist()

	def obterDataframeUnicoAudio(self, dataframe, arquivo):
		# Função para criar um dataframe contendo apenas as linhas referentes a um único áudio
		dataframeArquivoSelecionado = dataframe.loc[dataframe[dataframe.columns[0]] == arquivo]
		return dataframeArquivoSelecionado

	def obterYRealYPredUnicoAudio(self, dataframeUnicoAudio, classificador):
		# Função para predizer um único áudio e retornar as classificações real e predita
		# Pois bem, já temos a função de predizer um único áudio, mas ela precisa receber ua matriz bonitinha que só contenha features e que não seja do tipo pandas.
		# A função proposta agora deve receber o dataframe tipo pandas de um único áudio, separar o que é xTest e o que é yReal, mandar classificar xTest e devolver yReal e yPred.

		# PRIMEIRO EU SEPARO O QUE E X E Y
		xTest, arrayYReal = self.separarDataframeXeY(dataframeUnicoAudio)
		
		# FACO YREAL SER UM VALOR UNICO JA QUE A FUNCAO ACIMA RETORNA UM ARRAY Y
		yReal = arrayYReal[0]
		
		# PREDIZENDO A CLASSE DO AUDIO EM QUESTAO (A FUNCAO ABAIXO JA RETORNA A MODA)
		yPred = self.predizerUnicoAudio(xTest, classificador)
		
		return yReal, yPred  

	def classificarDataframeCompleto(self, dataframeTeste, classificador, verbose=False):
		# Função para classificar um dataframe de teste inteiro
		# A função abaixo vai receber um dataframe e um classificador. Depois disso ela vai:
		# 1) Obter um array com o nome dos arquivos desse dataframe utilizando a função obterNomesArquivos;
		# 2) Para cada arquivo no dataframe, ela vai criar um novo dataframe contendo as janelas apenas desse único áudio, utilizando, para isso, a função obterDataframeUnicoAudio;
		# 3) Obter a classificação real e a predita para cada um dos arquivos utilizando a função obterYRealYPredUnicoAudio;
		# 4) Com tudo isso, ela vai montar os arrays yTest e yPred;
		# 5) Retornar o classification report do sklearn como um DICT (depois fica bem facil de transformar em JSON) com todas as informações de precisão, recall, acurácia e tal.
		# A função que vai salvar o classification report, convertendo-o para um JSON será implementada a diante. Lembrando que essa função também vai ficar responsável por colocar o nome do classificador utilizado no classification report.
		
		print("Classificando com o ", classificador.__class__.__name__)
		
		# CRIANDO OS ARRAYS GERAIS
		yRealCadaAudio = []
		yPredCadaAudio = []
		
		# PEGANDO O NOME DOS ARQUIVOS QUE ESTAO NO DATAFRAME
		arrayNomesArquivos = self.obterNomesArquivos(dataframeTeste)
		totalArquivos = len(arrayNomesArquivos)
		
		# PARA CADA ARQUIVO NO DATAFRAME DE TESTE
		for i, arquivo in enumerate(arrayNomesArquivos):
			
			if verbose == True:
				print("Classificando " + arquivo + ". Arquivo", i+1, "de", totalArquivos, "->", str(100*((i+1)/totalArquivos))+"%")
			
			# EU CRIO UM DATAFRAME CONTENDO APENAS AS LINHAS DO AUDIO ATUAL
			dataframeAudioAtual = self.obterDataframeUnicoAudio(dataframeTeste, arquivo)
			
			# E OBTENHO, PARA ESSE AUDIO, A CLASSIFICACAO REAL E A PREDITA
			yRealAtual, yPredAtual = self.obterYRealYPredUnicoAudio(dataframeAudioAtual, classificador)
			
			# COLOCO O RESULTADO NOS ARRAYS GERAIS
			yRealCadaAudio.append(yRealAtual)
			yPredCadaAudio.append(yPredAtual)
			
		# CRIANDO O CLASSIFICATION REPORT. PARA FAZER ELE PARECER UM JSON, TEM QUE COLOCAR output_dict=True
		# ASSIM ELE RETORNA UM DICT QUE DEPOIS VAI SER CONVERTIDO PRA JSON E SER SALVO EM ALGUM LUGAR
		dictRelatorio = classification_report(yRealCadaAudio, yPredCadaAudio, digits=3, output_dict=True)
		
		# COLOCANDO O NOME DO CLASSIFICADOR NO DICIONARIO
		dictRelatorio['classificador'] = classificador.__class__.__name__
		
		return dictRelatorio

	def salvarRelatorioClassificacao(self, dictRelatorio, diretorioOndeSalvar, nomeDatasetTreino):
		# Função para salvar o classification report
		# Essa função vai receber um classification report como um DICT, converte-lo para JSON e salvar num diretório. Dentro desse dicionário já vai vir escrito qual foi o classificador utilizado, pois a função classificarDataframeCompleto coloca essa informação antes de retornar esse dicionário do relatório de classificação.
		nomeJSON = "relatorio_" + nomeDatasetTreino + "_" + str(time.time()) + ".json"
		
		print("Salvando as informações num JSON:", nomeJSON)
		
		with open(diretorioOndeSalvar + nomeJSON, 'w') as json_file:
			json.dump(dictRelatorio, json_file)

	def classificarDataframe(self, dataframeTeste, arrayObjClassificadores, diretorioOndeSalvar, nomeDatasetTreino, verbose=False):
		# Função para classificar um dataset com todos os classificadores e gerar os relatorios
		# Essa é a função principal da parte de treinamento de dataset, pois ela vai receber um dataset e o array de classificadores já treinados e, para cada classificador, ela vai garar um relátorio de classificação.
		print("Começando a classificar o dataset", nomeDatasetTreino)
		
		# PARA CADA CLASSIFICADOR
		for classificadorAtual in arrayObjClassificadores:
			
			# EU GERO O RELATORIO DE CLASSIFICACAO COMO UM DICIONARIO
			dictRelatorioAtual = self.classificarDataframeCompleto(dataframeTeste, classificadorAtual, verbose)

			print("Acurácia:", dictRelatorioAtual["accuracy"])
			
			# E SALVO AS INFORMACOES COMO JSON
			self.salvarRelatorioClassificacao(dictRelatorioAtual, diretorioOndeSalvar, nomeDatasetTreino)

		print("O teste com todos os classificadores foi finalizado")

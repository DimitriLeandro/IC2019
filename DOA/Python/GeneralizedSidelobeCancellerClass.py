# O objetivo desse jupyter é criar uma classe capaz de receber um array de sinais e retornar:
# 
# 1) sinalSemBeamforming: soma dos sinais do array sem nenhuma alteração;
# 
# 2) sinalBeamformado: sinal resultante do beamforming padrão, sem uso do GSC
# 
# 3) sinalFinalGSC: sinal final após todo o algoritmo do GSC
#
# TODO O CODIGO IMPLEMENTADO AQUI ESTA MUITO BEM DESCRITO NO JUPYTER "implementando classe para GSC"
# A ideia aqui é fazer uma classe que recebe apenas os arrays com os sinais originais. Simular um array de microfones vai ser uma coisa a parte.


import numpy as np
import time
from scipy.optimize import least_squares

class GeneralizedSidelobeCanceller:

	sinalSemBeamforming    = []
	sinalBeamformado       = []
	sinalFinalGSC          = []
	arraySinaisRuidosos    = []
	tempoProcessamentoGSC  = 0

	def __init__(self, arraySinaisOriginais=None, arrayDelays=None):
	
		# DEVO CONTINUAR O ALGORITMO E FAZER TUDO AUTOMATICAMENTE?
		if arraySinaisOriginais != None:

			# GERANDO O SINAL SEM BEAMFORMING
			self.sinalSemBeamforming = self.gerarSinalSemBeamforming(arraySinaisOriginais)

			# COMECANDO A CALCULAR O TEMPO DE PROCESSAMENTO DO ALGORITMO
			tempoInicio = time.time()
			
			# GERANDO O ARRAY DE DELAYS CASO AINDA NAO TENHA SIDO ENVIADO NO 
			# MEMENTO EM QUE SE INSTANCIOU A CLASSE 
			if arrayDelays == None:
				arrayDelays = self.obterArrayDelays(arraySinaisOriginais)
				
			# GERANDO O SINAL BEAMFORMADO E O ARRAY DE SINAIS DEFASADOS
			self.sinalBeamformado, arraySinaisDefasados = self.gerarSinalBeamformado(arraySinaisOriginais, arrayDelays)
		
			# GERANDO A BLOCKING MATRIX
			blockingMatrix = self.gerarBlockingMatrix(len(arraySinaisDefasados))

			# GERANDO OS SINAIS RUIDOSOS
			self.arraySinaisRuidosos = self.obterSinaisRuidosos(blockingMatrix, arraySinaisDefasados)

			# OBTENDO OS PESOS IDEAIS
			arrayPesos = self.obterArrayPesos(len(self.arraySinaisRuidosos))

			# GERANDO O SINAL FINAL GSC
			self.sinalFinalGSC = self.gerarSinalFinalGSC(self.sinalBeamformado, arrayPesos, self.arraySinaisRuidosos)

			# TERMINANDO DE CONTAR O TEMPO DE PROCESSAMENTO
			tempoFim = time.time()
			self.tempoProcessamentoGSC = tempoFim - tempoInicio
			print("Tempo total gasto para processar o GSC completo (segundos):", self.tempoProcessamentoGSC)


	def obterSinaisResultantes(self):
		return self.sinalSemBeamforming, self.sinalBeamformado, self.sinalFinalGSC

	def verificarDelay(self, sinalA, sinalB, maxDelay=15):
		#Para fazer o beamforming eu vou precisar saber qual é a defasagem entre os microfones. Pois bem, essa função recebe dois sinais e retorna a defasagem entre eles de acordo com a correlação.
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

	def obterArrayDelays(self, arraySinaisOriginais):
	
		# Função para gerar um vetor com o delay entre o n-ésimo microfone e o primeiro (referência)
		# Essa função vai receber o array de sinais e gerar o array de delays entre os sinais. O primeiro sinal do array de sinais sempre será a referência, e o array de delays conterá o delay dos microfones em relação somente à esse microfone de referência.
		# 
		# Além disso, é necessário - para as próximas funções - que o array de delays contenha o delay entre a referência e a própria referência. Ou seja, no começo do array de delays tem que ter um 0, já que a defasagem entre um sinal e ele mesmo é 0.

		print("Calculando os delays entre os microfones")

		arrayDelays = [0] # comeca com o delay entre o mic0 e o proprio mic0
		
		for i, sinalAtual in enumerate(arraySinaisOriginais):
			
			# se estivermos na iteracao 0 eu pulo, pois nao vou gastar processamento 
			# pra descobrir que o delay entre mic0 e mic0 e 0
			if i == 0:
				continue
				
			arrayDelays.append(self.verificarDelay(arraySinaisOriginais[0], sinalAtual))

		return arrayDelays

	def gerarSinalBeamformado(self, arraySinaisOriginais, arrayDelays):
		# Função para fazer um beamforming simples
		# A função abaixo vai pegar o array de sinais e o array de delays para realizar a defasagem necessária em cada sinal a fim de produzir um sinal final resultante da soma dos sinais originais devidamente defasados. Ao final, o sinal beamformado será dividido pela quantidade de microfones para que não fique muito alto.
		# 
		# Além disso, eu vou precisar de um array com os sinais defasados de acordo com os delays encontrados. Então, a função abaixo também vai retornar um o arraySinaisDefasados. A soma de todos os sinais desse array produz o sinalBeamformado. Isso, na verdade, é algo que vai ficar diferente do que foi feito no jupyter que tem o GSC genético. Por lá, eu implementei de um jeito que os sinais originais iam pra blocking matrix, e não os defasados. Mas a melhor forma de fazer é enviar os sinais defasados.

		# O VETOR DE DELAYS TAMBEM PRECISA TER O DELAY00
		
		# SINAL QUE VAI CONTER TODOS OS OUTROS SOMADOS
		sinalBeamformado = np.zeros(len(arraySinaisOriginais[0]))
		
		# ARRAY DE SINAIS DEFASADOS
		arraySinaisDefasados = []
		
		# PARA CADA SINAL
		for index, sinalAtual in enumerate(arraySinaisOriginais):
			
			# DELAY ENTRE O SINAL ATUAL E O SINAL 0
			delayAtual = arrayDelays[index]
			
			# SE FOR POSITIVO, EU COMO UMA PARTE DO SINAL NO COMECO E PREENCHO COM ZEROS NO FINAL
			if delayAtual >= 0:
				sinalAtualDefasado = np.concatenate((sinalAtual[delayAtual:], np.zeros(delayAtual)))
				
			# SE FOR NEGATIVO, EU COLOCO ZEROS NO COMECO E CORTO DO FINAL
			else:
				sinalAtualDefasado = np.concatenate((np.zeros(-delayAtual), sinalAtual[:delayAtual]))
				
			# COLOCO O SINAL ATUAL DEFASADO NO ARRAY DE SINAIS DEFASADOS E SOMO NO SINAL BEAMFORMADO
			sinalBeamformado += sinalAtualDefasado
			arraySinaisDefasados.append(sinalAtualDefasado)
				
		# NO FINAL EU AINDA DIVIDO O SINAL BEAMFORMADO PELO NUMERO DE MICS PRA NAO FICAR MTO ALTO
		sinalBeamformado /= len(arraySinaisOriginais)
		
		return sinalBeamformado, arraySinaisDefasados

	def gerarBlockingMatrix(self, qtdMics):
	
		blockingMatrix = np.zeros((qtdMics-1, qtdMics))
		
		for cont, linha in enumerate(blockingMatrix):
			linha[cont]   = 1
			linha[cont+1] = -1
		
		return blockingMatrix

	def calcularEnergiaSinal(self, sinal):
		return sum(sinal**2)

	def gerarSinalSemBeamforming(self, arraySinaisOriginais):
		return np.sum(arraySinaisOriginais, axis=0)/len(arraySinaisOriginais)

	def obterSinaisRuidosos(self, blockingMatrix, arraySinaisDefasados):
		# Função que passa os sinais defasados pela blocking matrix
		# Preciso de uma função que faça a multiplicação matricial entre os sinais defasados e a blocking matrix e retorne um array com os sinais que restaram após essa multiplicação. 
		# Os sinais depois da blocking matrix são quase puramente ruidosos, por isso, vou chamar o array resultante de arraySinaisRuidosos.

		return np.matmul(blockingMatrix, arraySinaisDefasados)

	def calcularEnergiaSinalFinal(self, arrayPesos):
		# Função para ser minimizada
		# O objetivo do Least Mean Squares é encontrar os parâmetros que minimizem uma função. Pois bem, a ideia aqui é minimizar a energia do sinal final quando for feita a operação

		# sinalGSC = sinalBeamformado - (pesos0 * sinaisRuidoso0 + ... + pesosN * sinaisRuidosoN)

		# Para isso, eu preciso de uma função que receba APENAS os pesos e retorne a energia do sinal após a operação acima, pois ela vai servir para eu usar o LMS a fim de encontrar os pesos que minimizem a energia do sinal final.

		# Ou seja, se essa função vai fazer a operação acima, que precisa do sinal beamformado e dos sinais ruidosos, mas ela só recebe os pesos, então ela vai precisar que o sinal beamformado e os sinais ruidosos sejam variáveis globais da classe!
		sinalFinalRuidoso = 0
		for index, pesoAtual in enumerate(arrayPesos):
			sinalFinalRuidoso += pesoAtual * self.arraySinaisRuidosos[index]
		
		sinalFinalLMS = self.sinalBeamformado - sinalFinalRuidoso
		
		return self.calcularEnergiaSinal(sinalFinalLMS)

	def obterArrayPesos(self, qtdSinaisRuidosos):
		# Função para encontrar os pesos minimizando a energia do sinal final
		# Essa é a função que vai usar o LMS para minimizar a função calcularEnergiaSinalFinal(arrayPesos) e retornar o array de pesos.
		# 
		# Ela só vai precisar saber a quantidade de pesos que ela vai ter que encontrar, e como é um peso pra cada sinal ruidoso, basta ela receber a qtd de sinais ruidosos.
		# 
		# Para o LMS rodar, ele precisa de um chute inicial e limites inferiores e superiores dos pesos. Esses valores vão ser fixos.
		# DEFININDO OS PARAMETROS INICIAIS
		chuteInicial      = np.full(qtdSinaisRuidosos, 0).tolist()
		limitesInferiores = np.full(qtdSinaisRuidosos, -1000).tolist()
		limitesSuperiores = np.full(qtdSinaisRuidosos, 1000).tolist()

		print("Iniciando a filtragem adaptativa com LMS")
		
		# RODANDO O LMS. ELE RETORNA UM OBJETO CHEIO DE ATRIBUTOS
		objRespostaLMS = least_squares(fun=self.calcularEnergiaSinalFinal, x0=chuteInicial, bounds=[limitesInferiores, limitesSuperiores], verbose=0)
		
		# O UNICO ATRIBUTO QUE ME INTERESSA E ESSE "X", QUE E A RESPOSTA ENCONTRADA DE FATO
		return objRespostaLMS.x
	
	def gerarSinalFinalGSC(self, sinalBeamformado, arrayPesos, arraySinaisRuidosos):
		# Função para construir o sinal final após o GSC
		# Nesse ponto, já vamos ter o sinal beamformado, os pesos e os sinais ruidosos. É só fazer acontecer!
		sinalFinalRuidoso = 0
		for index, pesoAtual in enumerate(arrayPesos):
			sinalFinalRuidoso += pesoAtual * arraySinaisRuidosos[index]
		
		sinalFinalGSC = sinalBeamformado - sinalFinalRuidoso
		
		return sinalFinalGSC	
	
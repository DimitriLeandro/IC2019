from joblib import Parallel, delayed
import numpy as np

class DaSParallel:

	delayMax             = 0
	arrayDelays          = []
	arraySinaisMics      = [] # SINAIS ORIGINAIS DOS MICROFONES
	arraySinaisAjustados = [] # SINAIS COM AS DEFASAGENS CORRIGIDAS
	sinalDaS             = [] # SINAL APOS O DELAY AND SUM

	def __init__(self, arraySinaisMics, delayMax=25, fazerMedia=True):
		# COLOCANDO OS PARAMETROS GLOBAIS DA CLASSE
		self.delayMax        = 25
		self.arraySinaisMics = arraySinaisMics

		# CALCULANDO O ARRAY DE DELAYS
		self.calcularArrayDelays()

		# CONSTRUINDO O SINAL BEAMFORMADO
		self.executarDaS(fazerMedia)

	def calcularCorrelacao(self, a, b):
		return np.corrcoef(a, b)[0][1]

	def ajustarSinaisDadoDelay(self, a, b, delayDesejado):
		# Função para ajustar dois sinais dado um delay a ser testado
		# Supondo que eu determine que um delay entre dois sinais é -4. Quero reconstruir os sinais para testar a correlação entre eles com esse delay.
		inicioB = self.delayMax - delayDesejado
		fimB    = -self.delayMax - delayDesejado
		if fimB == 0:
			fimB = None
		return a[self.delayMax:-self.delayMax], b[inicioB:fimB]

	def calcularDefasagemEntreDoisSinais(self, sinalA, sinalB):
		# Função para calcular a defasagem entre dois sinais quaisquer
		# Como foi observado que o gráfico Correlação x Delay é sempre quadrático com concavidade para baixo, essa função tenta calcular o menor número de correlações quanto possível.
		# Em primeiro lugar, ela calcula as correlações para os delays -1, 0 e 1. Com isso, ela vai saber em qual direção seguir. Isto é, os delays aumentam pra direita ou para a esquerda do gráfico? Sabendo isso, ela segue na direção em que as correlações aumentam e só para quando uma das correlações for mais baixa que a iteração anterior, pois, dessa forma, sabe-se que o pico foi encontrado.

		# COMECO CALCULANDO COM OS DELAYS -1, 0 E 1. DEPOIS EU VEJO PRA QUAL DIRECAO EU VOU
		sinalAAjustado, sinalBAjustado = self.ajustarSinaisDadoDelay(sinalA, sinalB, -1)
		correlacaoMenosUm = self.calcularCorrelacao(sinalAAjustado, sinalBAjustado)
		sinalAAjustado, sinalBAjustado = self.ajustarSinaisDadoDelay(sinalA, sinalB, 0)
		correlacaoZero    = self.calcularCorrelacao(sinalAAjustado, sinalBAjustado)
		sinalAAjustado, sinalBAjustado = self.ajustarSinaisDadoDelay(sinalA, sinalB, 1)
		correlacaoMaisUm  = self.calcularCorrelacao(sinalAAjustado, sinalBAjustado)
		
		# VERIFICANDO PRA QUAL DIRECAO EU VOU
		if correlacaoZero > correlacaoMenosUm and correlacaoZero > correlacaoMaisUm:
			# SE ENTROU AQUI E PQ DELAY 0 DA A MAIOR CORRELACAO
			return 0
		elif correlacaoMaisUm > correlacaoZero and correlacaoMaisUm > correlacaoMenosUm:
			# SE ENTROU AQUI A GNT VAI PRA CIMA
			maiorCorrelacao      = correlacaoMaisUm
			delayMaiorCorrelacao = 1
			rangeDelays          = np.arange(2, self.delayMax + 1, 1)
		elif correlacaoMenosUm > correlacaoZero and correlacaoMenosUm > correlacaoMaisUm:
			# SE ENTROU AQUI A GNT VAI PRA BAIXO
			maiorCorrelacao      = correlacaoMenosUm
			delayMaiorCorrelacao = -1
			rangeDelays          = np.arange(-2, -self.delayMax - 1, -1)
		else:
			# NAO E POSSIVEL QUE ENTRE AQUI MEU DEUS, MAS SO PRA GARANTIR NE...
			return None
		
		# SE A FUNCAO CHEGAR ATE AQUI E PQ AINDA NAO HOUVE RETORNO. VAMOS CONTINUAR TESTANDO OS DELAYS
		for delayAtual in rangeDelays:
			
			# CALCULO A CORRELACAO COM O DELAY ATUAL
			sinalAAjustado, sinalBAjustado = self.ajustarSinaisDadoDelay(sinalA, sinalB, delayAtual)
			correlacaoAtual = self.calcularCorrelacao(sinalAAjustado, sinalBAjustado)
			
			# JA QUE ESTAMOS INDO NA DIRECAO EM QUE A CORRELACAO AUMENTA, 
			# SE ELA DIMINUIR E PQ A GNT TINHA CHEGADO NO PICO NA ITERACAO ANTERIOR
			if correlacaoAtual < maiorCorrelacao:
				break
				
			# SE NAO DIMINUIU ENTAO SEGUE O BAILE
			maiorCorrelacao      = correlacaoAtual
			delayMaiorCorrelacao = delayAtual
			
		return delayMaiorCorrelacao

	def calcularArrayDelays(self):
		# Essa função vai usar a função anterior para calcular o delay entre cada microfofe e o microfone de origem. Ela deverá ser paralelizada, assim, cada núcleo fica responsável por uma combinação.
		# PARA CADA MIC QUE NAO SEJA O REFERENCIAL
		arrayDelaysParcial = Parallel(n_jobs=-1)(delayed(self.calcularDefasagemEntreDoisSinais)(self.arraySinaisMics[0], sinalAtual) for sinalAtual in self.arraySinaisMics[1:])
		
		# TEM QUE TER O 0 NO COMECO PQ EH O DELAY ENTRE O REFERENCIAL E ELE MESMO
		self.arrayDelays = np.array([0] + arrayDelaysParcial)

	def executarDaS(self, fazerMedia=True):
		# AJUSTANDO OS SINAIS PARA QUE POSSAM SER SOMADOS DEPOIS
		resposta = Parallel(n_jobs=-1)(delayed(self.ajustarSinaisDadoDelay)(self.arraySinaisMics[0], sinalAtual, delayAtual) for sinalAtual, delayAtual in zip(self.arraySinaisMics, self.arrayDelays))
		__, self.arraySinaisAjustados = zip(*resposta)
		
		# SOMANDO OS SINAIS AJUSTADOS
		self.sinalDaS = np.sum(self.arraySinaisAjustados, axis=0)
		
		# FAZENDO A MEDIA CASO DESEJADO
		if fazerMedia == True:
			self.sinalDaS /= len(self.arraySinaisMics)
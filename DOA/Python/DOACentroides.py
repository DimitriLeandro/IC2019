import numpy as np
from math import sqrt as sqrt
from scipy.spatial import distance

# Essa classe pode receber uma gravação dos 4 microfones (Respeaker), 
# extrair as features (delays os microfones) e comparar com centroides
# para predizer de qual direção um evento sonoro veio.
class DOACentroides:

	dado           = []
	centroides     = []
	freqAmostragem = 0
	L              = 0
	velocidadeSom  = 0
	maxDelay       = 0
	predicao       = -1

	# Quando um objeto dessa classe for instanciado, ele deverá mandar no construtor
	# os sinais dos quatro microfones e a frequência de amostragem, assim, o contrutor
	# vai extrair as features, transformar os sinais em um dado (vetor de features) e
	# predizer de qual direção o evento sonoro veio.
	def __init__(self, sinalMic1, sinalMic2, sinalMic3, sinalMic4, freqAmostragem=44100, L=0.058, velocidadeSom=340.29):
		# Em primeiro lugar, devo atribuir os valores às variáveis globais da classe
		self.freqAmostragem = freqAmostragem
		self.L              = L
		self.velocidadeSom  = velocidadeSom

		# Depois, devo encontrar o maior delay possível de acordo com a 
		# frequência de amostragem e das características físicas do Respeaker
		self.maxDelay = self.definirMaxDelay()

		# Definindo os centróides
		self.centroides = self.definirCentroides()

		# Agora sim eu faço a predição, verificando cada uma das features para compor 
		# o dado (vetor). Se o retorno da função verificarDelayEntreSinais for False
		# é porque ou os sinais têm tamanhos diferentes ou maxDelay é incompatível
		featureDelay12 = self.verificarDelayEntreSinais(sinalMic1, sinalMic2)
		featureDelay13 = self.verificarDelayEntreSinais(sinalMic1, sinalMic3)
		featureDelay14 = self.verificarDelayEntreSinais(sinalMic1, sinalMic4)
		featureDelay23 = self.verificarDelayEntreSinais(sinalMic2, sinalMic3)
		featureDelay24 = self.verificarDelayEntreSinais(sinalMic2, sinalMic4)
		featureDelay34 = self.verificarDelayEntreSinais(sinalMic3, sinalMic4)
		self.dado = np.array([
			featureDelay12, 
			featureDelay13, 
			featureDelay14, 
			featureDelay23, 
			featureDelay24, 
			featureDelay34
		])

		# Finalmente, predizendo de qual direção o envento sonoro veio
		self.predicao = self.predizerDirecao()


	# Essa função descobre qual é o delay máximo entre um microfone e outro 
	# (em quantidade de amostras). O delay máximo dar-se-á na diagonal dos microfones
	def definirMaxDelay(self):	    
		diagonal     = sqrt(2 * (self.L**2))
		tempo        = diagonal / self.velocidadeSom
		qtdAmostras  = self.freqAmostragem * tempo
		
		# Adicionando mais 2 amostras pra garantir
		return int(qtdAmostras + 2)


	# Essa função calcula teoricamente os centróides de cada posição de onde o
	# som pode ter vindo.
	def definirCentroides(self):  
		# Calculando o delay teórico entre o microfone 1 e o microfone 2 
		# supondo que o som veio da posição 1. Depois, os outros delays serão
		# advindos deste.
		meiaDiagonal = sqrt(2 * (self.L**2)) / 2
		tempo        = meiaDiagonal / self.velocidadeSom
		qtdAmostras  = self.freqAmostragem * tempo

		# Completando as features - POSIÇÃO 1
		delay12 = qtdAmostras
		delay13 = 2 * qtdAmostras
		delay14 = qtdAmostras
		delay23 = qtdAmostras
		delay24 = 0
		delay34 = -1 * qtdAmostras
		centroidePosicao1 = [delay12, delay13, delay14, delay23, delay24, delay34]
		
		# Completando as features - POSIÇÃO 2
		delay12 = -1 * qtdAmostras
		delay13 = 0
		delay14 = qtdAmostras
		delay23 = qtdAmostras
		delay24 = 2 * qtdAmostras
		delay34 = qtdAmostras
		centroidePosicao2 = [delay12, delay13, delay14, delay23, delay24, delay34]
		
		# Completando as features - POSIÇÃO 3
		delay12 = -1 * qtdAmostras
		delay13 = -2 * qtdAmostras
		delay14 = -1 * qtdAmostras
		delay23 = -1 * qtdAmostras
		delay24 = 0
		delay34 = qtdAmostras
		centroidePosicao3 = [delay12, delay13, delay14, delay23, delay24, delay34]
		
		# Completando as features - POSIÇÃO 4
		delay12 = qtdAmostras
		delay13 = 0
		delay14 = -1 * qtdAmostras
		delay23 = -1 * qtdAmostras
		delay24 = -2 * qtdAmostras
		delay34 = -1 * qtdAmostras
		centroidePosicao4 = [delay12, delay13, delay14, delay23, delay24, delay34]
		
		# Fazendo o array de centroides e retornando-o
		arrayCentroides = np.array([
			centroidePosicao1, 
			centroidePosicao2, 
			centroidePosicao3, 
			centroidePosicao4
		])
		return arrayCentroides


	# Essa função verifica o delay entre dois sinais a partir da correlação.
	def verificarDelayEntreSinais(self, sinalA, sinalB):
		# Verificando se os dois sinais tem o mesmo tamanho e se maxDelay é compatível
		if len(sinalA) != len(sinalB) or self.maxDelay >= len(sinalA)-1 :
			return False
		
		# Sei que o maior delay possível (em qtd de amostras) é maxDelay. Portanto, em cada iteração, 
		# a comparação entre os sinais para gerar a correlação se dará com arrays de tamanho 
		# len(sinal) - maxDelay
		tamanho = len(sinalA) - self.maxDelay
		
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
		for i in range(-self.maxDelay, self.maxDelay+1):
			
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


	# Essa função recebe um vetor de features e compara as distâncias desse dado até 
	# cada centroide. O centroide mais próximo indica a direção de onde o som veio.
	def predizerDirecao(self):
		# Verificando se os centroides e o sinal têm 6 features
		if self.dado.shape[0] != 6 or self.centroides.shape[1] != 6:
			return False
		
		# Para começar, o centroides[0] será o mais próximo
		melhorCentroide = 0
		melhorDistancia = distance.euclidean(self.dado, self.centroides[0])
		
		# Iterando cada um dos centroides
		for index, centroideAtual in enumerate(self.centroides):
			distanciaAtual = distance.euclidean(self.dado, centroideAtual)
			if distanciaAtual <= melhorDistancia:
				melhorDistancia = distanciaAtual
				melhorCentroide = index
				
		# Retornando o centroide mais próximo do sinal
		return melhorCentroide

import numpy as np
from math import sqrt as sqrt
from scipy.spatial import distance

class DOACentroides:

	def verificarDelay(sinalA, sinalB, maxDelay=15):
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


	def definirCentroides(freqAmostragem=44100, L=0.058, velocidadeSom=340.29):  
		# Calculando o delay entre o microfone 1 e o microfone 2 supondo que o som veio da posição 1
		meiaDiagonal = sqrt(2 * (L**2)) / 2
		tempo        = meiaDiagonal / velocidadeSom
		qtdAmostras  = freqAmostragem * tempo

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
		
		# Fazendo o array de centroides
		centroides = np.array([centroidePosicao1, centroidePosicao2, centroidePosicao3, centroidePosicao4])
		return centroides


	def centroideMaisProximo(dado, centroides):
	    # Verificando se os centroides e o sinal têm 6 features
	    if dado.shape[0] != 6 or centroides.shape[1] != 6:
	        return False
	    
	    # Para começar, o centroides[0] será o mais próximo
	    melhorCentroide = 0
	    melhorDistancia = distance.euclidean(dado, centroides[0])
	    
	    # Iterando cada um dos centroides
	    for index, centroideAtual in enumerate(centroides):
	        distanciaAtual = distance.euclidean(dado, centroideAtual)
	        if distanciaAtual <= melhorDistancia:
	            melhorDistancia = distanciaAtual
	            melhorCentroide = index
	            
	    # Retornando o centroide mais próximo do sinal
	    return melhorCentroide


	def definirDelayMaximo(freqAmostragem=44100, L=0.058, velocidadeSom=340.29): 
	    # O delay máximo (em qtd de amostras) dar-se-á na diagonal dos microfones
	    diagonal = sqrt(2 * (L**2))
	    tempo        = diagonal / velocidadeSom
	    qtdAmostras  = freqAmostragem * tempo
	    
	    # Adicionando mais 3 amostras pra garantir
	    return int(qtdAmostras + 5)


	def transformarSinais(sinal1, sinal2, sinal3, sinal4, maxDelay=15):
	    # O objetivo é transformar 4 sinais em um dado de 6 dimensões
	    # Extraindo as 6 features:
	    delay12 = verificarDelay(sinal1, sinal2, maxDelay)
	    delay13 = verificarDelay(sinal1, sinal3, maxDelay)
	    delay14 = verificarDelay(sinal1, sinal4, maxDelay)
	    delay23 = verificarDelay(sinal2, sinal3, maxDelay)
	    delay24 = verificarDelay(sinal2, sinal4, maxDelay)
	    delay34 = verificarDelay(sinal3, sinal4, maxDelay)
	    
	    # Criando o vetor o retornando-o
	    novoDado = np.array([delay12, delay13, delay14, delay23, delay24, delay34])
	    return novoDado
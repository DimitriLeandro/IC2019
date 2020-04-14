# Implementação de uma classe para simular um array de microfones
# A ideia desse jupyter é implementar uma classe que receba o caminho para um arquivo de áudio WAV e realize todo o processamento para simular um array de microfones com ou sem ruído.
# O construtor da classe deverá realizar todo o processamento, setando os arrays de sinais e de delays, que serão públicos. Deverá haver um método que retorna essas dois arrays.

# PARA USAR A CLASE:

# EXEMPLO----------------------------------------------

# IMPORTANDO A CLASSE DE SIMULACAO DE ARRAY DE MICROFONES
# from SimularArrayMicrofonesClass import SimularArrayMicrofones

# # DEFININDO OS PARAMETROS INICIAIS
# caminhoArquivo   = '/home/dimi/Programming/IC2019/DOA/Gravacoes/Simulacoes/Gravacao3/sinalPuroMono2Segundos.wav'
# cooredenadasMics = Null para coordenadas padrão ou uma matriz onde as linhas são os microfones
# energiaRuido     = 0.01

# # INSTANCIANDO UM OBJETO DA CLASSE 
# objSimularArrayMics = SimularArrayMicrofones(caminhoArquivo, energiaRuido, coordenadas)

# # OBTENDO O ARRAY DE SINAIS E DE DELAYS E A FREQ DE AMOSTRAGEM
# arraySinaisSimulados, arrayDelays, freqAmostragem = objSimularArrayMics.obterResultado()
#-----------------------------------------------------------

import librosa
import numpy as np
import math
from random import randint

class SimularArrayMicrofones:

	arraySinaisSimulados = []
	arrayDelays          = []
	freqAmostragem       = 0

	def __init__(self, caminhoArquivo, energiaRuido=None, normalizarEnergia=False, azimutalRad=None, elevacaoRad=None, coordenadasMics=[[0,0,0],[0,0.04137,0.04137],[0.0585,0.04137,0.04137],[0.0585,0,0]]):    
		# ABRINDO O ARQUIVO MONO PURO
		sinalPuroMono, self.freqAmostragem = librosa.load(caminhoArquivo, sr=None, mono=True)

		# NORMALIZO PELA ENERGIA CASO SE QUEIRA
		if normalizarEnergia == True:
			sinalPuroMono = (sinalPuroMono-np.mean(sinalPuroMono))/np.std(sinalPuroMono)
		
		# GERANDO UM DELAY ENTRE OS MICROFONES
		self.arrayDelays = self.obterArrayDelays(coordenadasMics, self.freqAmostragem, azimutalRad, elevacaoRad)

		
		# GERANDO OS SINAIS DE CADA MICROFONE DE ACORDO COM OS DELAYS GERADOS
		self.arraySinaisSimulados = self.simularArrayMicrofones(sinalPuroMono, self.arrayDelays)
		
		# ADICIONANDO RUIDO
		if energiaRuido != None:
			self.arraySinaisSimulados = self.adicionarRuido(self.arraySinaisSimulados, energiaRuido)

	def obterResultado(self):
		return self.arraySinaisSimulados, self.arrayDelays, self.freqAmostragem

	def gerarRuidoBranco(self, qtdAmostras, energiaRuido):

		media        = 0
		desvioPadrao = 1
		
		return np.random.normal(media, desvioPadrao, size=qtdAmostras) * energiaRuido**(1/2)

	def calcularProdutoInterno(self, vetorA, vetorB):
		soma = 0	    
		for i in range(len(vetorA)):
			soma += vetorA[i] * vetorB[i]      
		return soma

	def grausParaRad(self, angRad):
		return (angRad * math.pi)/180

	def radParaGraus(self, angGraus):
		return (angGraus * 180)/math.pi

	def segundosParaAmostras(self, segundos, freqAmostragem):
		return freqAmostragem * segundos

	def obterArrayDelays(self, coordenadasMics, freqAmostragem, azimutalRad=None, elevacaoRad=None, velocidadeSom=340.29):
		# Função para gerar um array de delays entre os microfones
		# Essa função precisará receber as coordenadas dos microfones para gerar os delays entre eles de acordo com os angulos azimutal e de elevação, que serão aleatórios.
		# Para achar os delays entre os microfones basta fazer o algoritmo do produto interno.
		
		arrayDelays = []
		
		# GERANDO AZIMUTAL E ELEVACAO ALEATORIOS (EM RAD)
		if azimutalRad == None:
			azimutalRad = self.grausParaRad(randint(0, 359))
		if elevacaoRad == None:
			elevacaoRad = self.grausParaRad(randint(-90, 90))
		
		# CONVERTENDO PARA UM VETOR UNITARIO
		vetorWavefront = [
			math.cos(elevacaoRad) * math.cos(azimutalRad),
			math.cos(elevacaoRad) * math.sin(azimutalRad),
			math.sin(elevacaoRad)
		]
		
		# PARA CADA MICROFONE
		for coordenadasOriginaisMicAtual in coordenadasMics:
			
			# EU FACO A DIFERENCA DAS COORDENADAS ATUAIS PRAS COORDENADAS DO MICROFONE DA ORIGEM
			coordenadasModificadasMicAtual = np.array(coordenadasOriginaisMicAtual) - np.array(coordenadasMics[0])
			
			# CALCULO DELAY USANDO O PRODUTO INTERNO DAS COORDENADAS DO MIC ATUAL COM O VETOR WAVEFRONT
			delayAtual = self.calcularProdutoInterno(vetorWavefront, coordenadasModificadasMicAtual)/velocidadeSom
			delayAtual = self.segundosParaAmostras(delayAtual, freqAmostragem)
			delayAtual = int(round(delayAtual))
			arrayDelays.append(delayAtual)
			
		return arrayDelays

	def simularArrayMicrofones(self, sinalPuroMono, arrayDelays):
		# Função para gerar um array de microfones e o delay entre eles
		# A função abaixo vai receber um sinal mono e vai gerar um array de sinais. Basta copiar e colar o sinal mono e defasar aleatoriamente. Essa função vai devolver o array de sinais e o delay entre eles.
		# ARRAY DE SINAIS
		arraySinaisSimulados = []
		
		# PARA CADA DELAY ENTRE OS MICROFONES
		for delayAtual in arrayDelays:
			
			# SE FOR POSITIVO, EU COMO UMA PARTE DO SINAL NO COMECO E PREENCHO COM ZEROS NO FINAL
			if delayAtual >= 0:
				sinalMicAtual = np.concatenate((sinalPuroMono[delayAtual:], np.zeros(delayAtual)))
				
			# SE FOR NEGATIVO, EU COLOCO ZEROS NO COMECO E CORTO DO FINAL
			else:
				sinalMicAtual = np.concatenate((np.zeros(-delayAtual), sinalPuroMono[:delayAtual]))
				
			# COLOCO O SINAL COPIADO E DEFASADO NO ARRAY DE MICROFONES
			arraySinaisSimulados.append(sinalMicAtual)
		
		return arraySinaisSimulados
	
	def adicionarRuido(self, arraySinaisSimulados, energiaRuido):
		# Função para adicionar ruido a cada um dos microfones
		# O ruído não poderá ser defasado, pois ele vem de todas as direções. Essa função vai utilizar a função que gera o ruído.
		ruido = self.gerarRuidoBranco(len(arraySinaisSimulados[0]), energiaRuido)
		
		for i, sinalAtual in enumerate(arraySinaisSimulados):
			arraySinaisSimulados[i] = sinalAtual + ruido
		
		return arraySinaisSimulados
	
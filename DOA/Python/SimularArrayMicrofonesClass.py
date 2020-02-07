# Implementação de uma classe para simular um array de microfones
# A ideia desse jupyter é implementar uma classe que receba o caminho para um arquivo de áudio WAV e realize todo o processamento para simular um array de microfones com ou sem ruído.
# O construtor da classe deverá realizar todo o processamento, setando os arrays de sinais e de delays, que serão públicos. Deverá haver um método que retorna essas dois arrays.

# PARA USAR A CLASE:

# EXEMPLO----------------------------------------------

# IMPORTANDO A CLASSE DE SIMULACAO DE ARRAY DE MICROFONES
# from SimularArrayMicrofonesClass import SimularArrayMicrofones

# # DEFININDO OS PARAMETROS INICIAIS
# caminhoArquivo = '/home/dimi/Programming/IC2019/DOA/Gravacoes/Simulacoes/Gravacao3/sinalPuroMono2Segundos.wav'
# qtdMics        = 4
# energiaRuido   = 0.01

# # INSTANCIANDO UM OBJETO DA CLASSE 
# objSimularArrayMics = SimularArrayMicrofones(caminhoArquivo, qtdMics, energiaRuido)

# # OBTENDO O ARRAY DE SINAIS E DE DELAYS E A FREQ DE AMOSTRAGEM
# arraySinaisSimulados, arrayDelays, freqAmostragem = objSimularArrayMics.obterResultado()
#-----------------------------------------------------------

import librosa
import numpy as np
from random import randint

class SimularArrayMicrofones:

	arraySinaisSimulados = []
	arrayDelays          = []
	freqAmostragem       = 0

	def __init__(self, caminhoArquivo, qtdMics, energiaRuido=None):
		# Função construtora para unir tudo
		# Abaixo, vou criar a função construtora que apenas recebe o caminho para um arquivo WAV, a quantidade de microfones desejada e se deve adicionar ruido branco nos microfones.
		# ABRINDO O ARQUIVO MONO PURO
		sinalPuroMono, self.freqAmostragem = librosa.load(caminhoArquivo, sr=None, mono=True)
		
		# COLOCANDO A DEFASAGEM ENTRE OS MICROFONES
		self.arraySinaisSimulados, self.arrayDelays = self.simularArrayMicrofones(sinalPuroMono, qtdMics)
		
		# ADICIONANDO RUIDO
		if energiaRuido != None:
			self.arraySinaisSimulados = self.adicionarRuido(self.arraySinaisSimulados, energiaRuido)

	def obterResultado(self):
		return self.arraySinaisSimulados, self.arrayDelays, self.freqAmostragem

	def gerarRuidoBranco(self, qtdAmostras, energiaRuido):

		media        = 0
		desvioPadrao = 1
		
		return np.random.normal(media, desvioPadrao, size=qtdAmostras) * energiaRuido**(1/2)

	def simularArrayMicrofones(self, sinalPuroMono, qtdMics):
		# Função para gerar um array de microfones e o delay entre eles
		# A função abaixo vai receber um sinal mono e vai gerar um array de sinais. Basta copiar e colar o sinal mono e defasar aleatoriamente. Essa função vai devolver o array de sinais e o delay entre eles.
		# ARRAY DE SINAIS
		arraySinaisSimulados = []
		arrayDelays          = [0]
		
		# VOU CRIAR qtdMics SINAIS QUE SAO O PROPRIO SINAL PURO, MAS DEFASADOS ALEATORIAMENTE
		for i in range(qtdMics):
			
			# O PRIMEIRO MICROFONE NAO VAI TER DEFASAGEM EM RELACAO AO SINAL PURO
			if i == 0:
				arraySinaisSimulados.append(sinalPuroMono)
				continue
			
			# GERANDO UM DELAY (em amostras, nao em segundos) ALEATORIO PARA O MIC DA VEZ
			delayAleatorio = randint(-30,30)
			arrayDelays.append(delayAleatorio)
			
			# SE FOR POSITIVO, EU COMO UMA PARTE DO SINAL NO COMECO E PREENCHO COM ZEROS NO FINAL
			if delayAleatorio >= 0:
				sinalMicAtual = np.concatenate((sinalPuroMono[delayAleatorio:], np.zeros(delayAleatorio)))
				
			# SE FOR NEGATIVO, EU COLOCO ZEROS NO COMECO E CORTO DO FINAL
			else:
				sinalMicAtual = np.concatenate((np.zeros(-delayAleatorio), sinalPuroMono[:delayAleatorio]))
				
			# COLOCO O SINAL COPIADO E DEFASADO NO ARRAY DE MICROFONES
			arraySinaisSimulados.append(sinalMicAtual)
		
		return arraySinaisSimulados, arrayDelays
	
	def adicionarRuido(self, arraySinaisSimulados, energiaRuido):
		# Função para adicionar ruido a cada um dos microfones
		# O ruído não poderá ser defasado, pois ele vem de todas as direções. Essa função vai utilizar a função que gera o ruído.
		ruido = self.gerarRuidoBranco(len(arraySinaisSimulados[0]), energiaRuido)
		
		for i, sinalAtual in enumerate(arraySinaisSimulados):
			arraySinaisSimulados[i] = sinalAtual + ruido
		
		return arraySinaisSimulados
	
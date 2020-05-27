import numpy as np
import time
from scipy.optimize import least_squares

# Como uma classe para fazer apenas o DaS foi criada, o GSC também vai precisar de uma classe nova. A ideia é usar os sinais dos microfones e esse sinal que vai vir dessa classe do DaS.

class GSC:

	arraySinaisMics     = [] # ARRAY COM OS SINAIS ORIGINAIS CAPTADOS PELOS MICS
	blockingMatrix      = [] # NAO TEM MTO O QUE COMENTAR AQUI
	arraySinaisRuidosos = [] # ARRAY COM OS SINAIS QUE PASSARAM PELA BLOCKING MATRIX
	sinalDaS            = [] # SINAL QUE A CLASSE RECEBE NO INIT QUE VEIO DO DaS
	arrayPesos          = [] # ARRAY COM OS PESOS QUE MULTIPLICAM OS SINAIS RUIDOSOS
	sinalGSC            = [] # SINAL QUE VAI TER O RESULTADO DE TODO O GSC
	tempoProcessamento  = 0  # TEMPO DE PROCESSAMENTO EM SEGUNDOS -> ESSE TEMPO E APENAS DA PARTE ADAPTATIVAAAAA O SINAL DO DELAY SUM VAI SER FEITO EM OUTRA CLASSE. PRA SABER O TEMPO REAL DO GSC TEM QUE SOMAR OS TEMPOS! (DAS + PARTE ADAPTATIVA DO GSC)

	def __init__(self, arraySinaisMics, sinalDaS):
		# PARAMETROS INICIAIS
		self.arraySinaisMics     = self.cortarSinais(arraySinaisMics, len(sinalDaS))
		self.blockingMatrix      = self.gerarBlockingMatrix(len(arraySinaisMics))
		self.sinalDaS            = sinalDaS

		# INICIANDO A MEDICAO DE TEMPO DE PROCESSAMENTO
		tempoInicio = time.time()

		# RODANDO A PARTE ADAPTATIVA DO GSC
		self.arraySinaisRuidosos = self.obterSinaisRuidosos()		
		self.arrayPesos          = self.obterArrayPesos(len(self.arraySinaisRuidosos))
		self.sinalGSC            = self.gerarSinalFinalGSC()

		# FINALIZANDO A MEDICAO DE TEMPO DE PROCESSAMENTO
		tempoFim = time.time()
		self.tempoProcessamento = tempoFim - tempoInicio

	def cortarSinais(self, sinais, tamanho):
		# Função para cortar um pedacinho dos sinais dos microfones
		# O DaS implementado come um pouquinho do sinal captado pelos microfones (bem pouquinho). Pras operações que eu vou fazer em seguida darem certo, preciso deixar o tamanho dos sinais igual.
		return sinais[:,0:tamanho]

	def gerarBlockingMatrix(self, qtdMics):
		bm = np.zeros((qtdMics-1, qtdMics))
		for cont, linha in enumerate(bm):
			linha[cont]   = 1
			linha[cont+1] = -1
		return bm

	def obterSinaisRuidosos(self):
		# Função para passar os sinais dos mics pela blocking matrix
		return np.matmul(self.blockingMatrix, self.arraySinaisMics)

	def calcularEnergiaSinalFinal(self, pesos):   
		# Função que será minimizada pelo LMS 
		sinalFinalRuidoso = np.zeros(len(self.sinalDaS))
		for index, pesoAtual in enumerate(pesos):
			sinalFinalRuidoso += pesoAtual * self.arraySinaisRuidosos[index]
		sinalFinal = self.sinalDaS - sinalFinalRuidoso
		return np.var(sinalFinal)

	def obterArrayPesos(self, qtdSinaisRuidosos):
		chuteInicial      = np.full(qtdSinaisRuidosos, 0).tolist()
		limitesInferiores = np.full(qtdSinaisRuidosos, -100).tolist()
		limitesSuperiores = np.full(qtdSinaisRuidosos, 100).tolist()

		# RODANDO O LMS. ELE RETORNA UM OBJETO CHEIO DE ATRIBUTOS
		objRespostaLMS = least_squares(fun=self.calcularEnergiaSinalFinal, x0=chuteInicial, bounds=[limitesInferiores, limitesSuperiores], verbose=0)

		# O UNICO ATRIBUTO QUE ME INTERESSA E ESSE "X", QUE E A RESPOSTA ENCONTRADA DE FATO
		return objRespostaLMS.x

	def gerarSinalFinalGSC(self):
		sinalFinalRuidoso = np.zeros(len(self.sinalDaS))
		for index, pesoAtual in enumerate(self.arrayPesos):
			sinalFinalRuidoso += pesoAtual * self.arraySinaisRuidosos[index]
		sinalFinal = self.sinalDaS - sinalFinalRuidoso

		return sinalFinal
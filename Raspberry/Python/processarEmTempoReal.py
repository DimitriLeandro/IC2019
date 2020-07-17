import sys
import pyaudio
import numpy as np
import warnings
import threading
from time import time, sleep
from delayAndSum import delayAndSum as DaS
from extracaoFeatures import extrairFeaturesUnicoFrame as ExtrairFeatures
from iniciarNormalizadorEClassificador import main as IniciarObjetos
warnings.filterwarnings("ignore")


# DEFINICAO DE FUNCOES ---------------------------------------------------------------------------
def processarJanela(arraySinais, idJanela, freqAmostragem, objNormalizador, objClassificador):
	# Função para realizar todo o processamento em uma janela
	# A função receberá uma janela gravada, aplicará o Delay and Sum, extrairá as features do sinal beamformado, normalizará o vetor de features obtido, e irá classificar o dado gerado.

    # INICIANDO A MEDICAO DE TEMPO
    tempoInicio = time()
    stringID = "(id:" + str(idJanela) + ")"
    print(stringID, "Iniciando o processamento")

    # DELAY AND SUM
    sinalDaS = DaS(arraySinais)

    # EXTRACAO DE FEATURES
    features = ExtrairFeatures(sinalDaS, freqAmostragem)

    # NORMALIZANDO AS FEATURES
    features = objNormalizador.transform([features])[0]

    # CLASSIFICACAO
    predicao = objClassificador.predict([features])[0]

    # FINALIZANDO A MEDICAO DE TEMPO
    tempoFim = time()
    print(stringID, "Classificado como", predicao)
    print(stringID, "Processamento finalizado em", tempoFim - tempoInicio, "s")


def gravarJanela(idJanela, objNormalizador, objClassificador, idDispositivoGravacao, tempoJanela, freqAmostragem, profundidadeBytes, qtdCanais):
	#Função para gravar uma janela com o dispositivo
	#A função abaixo deverá gravar uma janela e depois enviá-la para a função anterior. A função anterior deve ser rodada como uma thread.

    #assert profundidadeBytes == 2, "Profundidade de bytes diferente de 2! Você deve alterar o sistema manualmente."

    # DEFININDO O TAMANHO DA JANELA EM AMOSTRAS
    tamanhoJanela = int(tempoJanela * freqAmostragem)

    # INSTANCIANDO UM OBJ PY AUDIO E MANDANDO OS PARAMETROS INICIAIS
    objPyAudio = pyaudio.PyAudio()
    stream = objPyAudio.open(
        input_device_index = idDispositivoGravacao,
        rate               = freqAmostragem,
        format             = objPyAudio.get_format_from_width(profundidadeBytes),
        channels           = qtdCanais,
        input              = True
    )

    # VERBOSE
    stringID = "(id:" + str(idJanela) + ")"
    print(stringID, "Iniciando gravação. Timestamp:", time())

    # GRAVANDO A JANELA
    janelaBinaria = stream.read(tamanhoJanela)

    # VERBOSE
    print(stringID, "Gravação finalizada. Timestamp:", time())

    # MATANDO OS OBJETOS PRA LIMPAR MEMORIA
    stream.stop_stream()
    stream.close()
    objPyAudio.terminate()

    # CONVERTO A JANELA PRA INT 16
    janelaInt16 = np.fromstring(janelaBinaria, dtype=np.int16)

    # COMO TEM OS 4 MICS NA JANELA, VOU DAR UM RESHAPE PRA CADA MIC FICAR EM UMA LINHA
    janelaInt16 = janelaInt16.reshape((tamanhoJanela, qtdCanais)).T

    # GARANINDO QUE A DIMENSIONALIDADE ESTA CORRETA
    #assert janelaInt16.shape[0] == qtdCanais and janelaInt16.shape[1] == tamanhoJanela, "Erro na dimensionalidade da janela gravada!"

    # MANDANDO A JANELA GRAVADA APRA O PROCESSAMENTO
    objThread = threading.Thread(target=processarJanela, args=(janelaInt16, idJanela, freqAmostragem, objNormalizador, objClassificador))
    objThread.start()

# RODANDO O ALGORITMO -----------------------------------------------

# PARAMETROS INCIAIS
idDispositivoGravacao = 0
tempoJanela           = 0.200
freqAmostragem        = 16000
profundidadeBytes     = 2
qtdCanais             = 4
caminhoCSVDataset     = "/home/pi/Datasets/SESA_v2_16kHz_16bits_200ms_58features_desescalonado_remocaoSilencio.csv"

# INSTANCIANDO OS OBJETOS DE NORMALIZACAO E CLASSIFICACAO
objClassificador, objNormalizador = IniciarObjetos(caminhoCSVDataset, classificador=None, verbose=True)

# VAMO QUE VAMO
i = 1
while True:

    # INICIANDO JANELA PRINCIPAL
    objThread = threading.Thread(target=gravarJanela, args=(i, objNormalizador, objClassificador, idDispositivoGravacao, tempoJanela, freqAmostragem, profundidadeBytes, qtdCanais))
    objThread.start()

    # DELAY PARA COMECAR A JANELA DE SOBREPOSICAO
    i += 1
    sleep(tempoJanela/2)

    # INICIANDO JANELA DE SOBREPOSICAO
    objThread = threading.Thread(target=gravarJanela, args=(i, objNormalizador, objClassificador, idDispositivoGravacao, tempoJanela, freqAmostragem, profundidadeBytes, qtdCanais))
    objThread.start()

    # DELAY PARA COMECAR A JANELA PRICIPAL NA PROXIMA ITERACAO
    i += 1
    sleep(tempoJanela/2)

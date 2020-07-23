# PARA RODAR ESSE COMANDO: (venv1)$ python Programming/IC2019/Raspberry/Python/processarEmTempoReal.py 2>/dev/null
# COLOCAR 2>/dev/null NO FINAL SUPRIME OS ALERTAS DO ALSA

import warnings
warnings.filterwarnings("ignore")
import pyaudio
import numpy as np
import threading
from time import time, sleep
from delayAndSum import delayAndSum as DaS
from extracaoFeatures import extrairFeaturesUnicoFrame as ExtrairFeatures
from iniciarNormalizadorEClassificador import main as IniciarObjetos

# DEFINICAO DE FUNCOES --------------------------------------------------

def processarJanelaCompleta(idJanela, metadeInicialJanelaAtual, metadeFinalJanelaAtual):
    
    objFile.write("(id:"+str(idJanela)+"): Inicio processamento. Timestamp "+str(time())+"\n")
    
    # COPIANDO OS ARRAYS NA MEMORIA PRA GARANTIR QUE NAO VAI DAR BOSTA
    metadeInicial = np.copy(metadeInicialJanelaAtual)
    metadeFinal   = np.copy(metadeFinalJanelaAtual)
    
    # CONVERTO A JANELA PRA INT 16
    janelaInt16_A = np.fromstring(metadeInicial, dtype=np.int16)
    janelaInt16_B = np.fromstring(metadeFinal, dtype=np.int16)
    
    # COMO TEM OS 4 MICS NA JANELA, VOU DAR UM RESHAPE PRA CADA MIC FICAR EM UMA LINHA
    janelaInt16_A = janelaInt16_A.reshape((metadeTamanhoJanela, qtdCanais)).T   
    janelaInt16_B = janelaInt16_B.reshape((metadeTamanhoJanela, qtdCanais)).T
    
    # DELAY AND SUM
    sinalDaS_A = DaS(janelaInt16_A)
    sinalDaS_B = DaS(janelaInt16_B)
    
    # JUNTO AS METADES
    janelaCompleta = np.concatenate((sinalDaS_A, sinalDaS_B))
    
    # EXTRACAO DE FEATURES
    features = ExtrairFeatures(janelaCompleta, freqAmostragem)

    # NORMALIZANDO AS FEATURES
    features = objNormalizador.transform([features])[0]

    # CLASSIFICACAO
    predicao = objClassificador.predict([features])[0]
    
    print("(id:"+str(idJanela)+"):", predicao)
    objFile.write("(id:"+str(idJanela)+"): Fim processamento. " + str(predicao) + ". Timestamp "+str(time())+"\n")

# RODANDO O ALGORITMO -----------------------------------------------
print("Iniciando a aplicacao...")

# PARAMETROS INICIAIS
executarAteIdJanela   = 1000
idDispositivoGravacao = 0
tempoJanela           = 0.200
freqAmostragem        = 16000
profundidadeBytes     = 2
qtdCanais             = 4
metadeTamanhoJanela   = int((tempoJanela * freqAmostragem)/2)
caminhoCSVDataset     = "/home/pi/Datasets/SESA_ReSpeaker_16kHz_16bits_200ms_58features_desescalonado_2.csv"
caminhoGravarLog      = "/home/pi/Programming/IC2019/Raspberry/Resultados/Logs/logProcessamento_"+str(time())+".txt"

# INICIANDO OS OBJETOS NECESSARIOS (CLASSIFICADOR, NORMALIZADOR E ESCREVER EM TXT)
objClassificador, objNormalizador = IniciarObjetos(caminhoCSVDataset, verbose=True)
objFile = open(caminhoGravarLog, "w")

# INICIANDO O PYAUDIO E STREAM
objPyAudio = pyaudio.PyAudio()
stream = objPyAudio.open(
    input_device_index = idDispositivoGravacao,
    rate               = freqAmostragem,
    format             = objPyAudio.get_format_from_width(profundidadeBytes),
    channels           = qtdCanais,
    input              = True
)

# LOOP
idJanela = 1
while idJanela <= executarAteIdJanela:

    # GRAVANDO
    janelasParciais = []
    for i in range(5):
        
        if i == 0:
            objFile.write("(id:"+str(idJanela)+"): Inicio gravacao. Timestamp "+str(time())+"\n")
            stream.start_stream()
            janelasParciais.append(stream.read(metadeTamanhoJanela))
            stream.stop_stream()            
            idJanela += 1
            
        elif i == 1:
            objFile.write("(id:"+str(idJanela)+"): Inicio gravacao. Timestamp "+str(time())+"\n")
            stream.start_stream()
            janelasParciais.append(stream.read(metadeTamanhoJanela))
            stream.stop_stream()
            objFile.write("(id:"+str(idJanela-1)+"): Fim gravacao. Timestamp "+str(time())+"\n")            
            objThread = threading.Thread(target=processarJanelaCompleta, args=(idJanela-1, janelasParciais[i-1], janelasParciais[i]))
            objThread.start()            
            idJanela += 1
            
        elif i == 2:
            objFile.write("(id:"+str(idJanela)+"): Inicio gravacao. Timestamp "+str(time())+"\n")
            stream.start_stream()
            janelasParciais.append(stream.read(metadeTamanhoJanela))
            stream.stop_stream()
            objFile.write("(id:"+str(idJanela-1)+"): Fim gravacao. Timestamp "+str(time())+"\n")            
            objThread = threading.Thread(target=processarJanelaCompleta, args=(idJanela-1, janelasParciais[i-1], janelasParciais[i]))
            objThread.start()            
            idJanela += 1
            
        elif i == 3:
            objFile.write("(id:"+str(idJanela)+"): Inicio gravacao. Timestamp "+str(time())+"\n")
            stream.start_stream()
            janelasParciais.append(stream.read(metadeTamanhoJanela))
            stream.stop_stream()
            objFile.write("(id:"+str(idJanela-1)+"): Fim gravacao. Timestamp "+str(time())+"\n")            
            objThread = threading.Thread(target=processarJanelaCompleta, args=(idJanela-1, janelasParciais[i-1], janelasParciais[i]))
            objThread.start()
            
        else:
            stream.start_stream()
            janelasParciais.append(stream.read(metadeTamanhoJanela))
            stream.stop_stream()
            objFile.write("(id:"+str(idJanela)+"): Fim gravacao. Timestamp "+str(time())+"\n")            
            objThread = threading.Thread(target=processarJanelaCompleta, args=(idJanela, janelasParciais[i-1], janelasParciais[i]))
            objThread.start()            
            idJanela += 1

# PARANDO O AMBIENTE
stream.stop_stream()
stream.close()
objPyAudio.terminate()
del objPyAudio
del stream

sleep(5)
print("Finalizando a aplicação...")
objFile.close()

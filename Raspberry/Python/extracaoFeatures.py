#----------------------------------------------------------------------
# ESSE ARQUIVO NAO IMPLEMENTA UMA CLASSE, VAMOS USAR FUNCOES SOLTAS
#----------------------------------------------------------------------
import librosa
import numpy as np


# ESSA E A FUNCAO QUE RECEBE UMA JANELA E EXTRAI TODAS AS SUAS FEATURES
def extrairFeaturesUnicoFrame(sinal, freqAmostragem):
    
    # PARA IMPEDIR QUE O LIBROSA CONTINUE FAZENDO O JANELAMENTO DO ÁUDIO MESMO QUE frameLength 
    # SEJA DO TAMANHO DO ÁUDIO, O PARÂMETRO DE OVERLAP DEVE SER MAIOR QUE frameLength
    frameLength   = len(sinal)
    overlapLength = frameLength + 1

    # CRIANDO O ARRAY DE FEATURES DO FRAME EM QUESTAO
    arrayFeaturesFrame = []

    #PRIMEIRO, VOU EXTRAIR AS FEATURES UNITARIAS
    arrayFeaturesFrame.append(float(extrairRMS(sinal, frameLength, overlapLength)))
    arrayFeaturesFrame.append(float(extrairCentroideEspectral(sinal, freqAmostragem, frameLength, overlapLength)))
    arrayFeaturesFrame.append(float(extrairLarguraBanda(sinal, freqAmostragem, frameLength, overlapLength)))
    arrayFeaturesFrame.append(float(extrairPlanicidade(sinal, frameLength, overlapLength)))
    arrayFeaturesFrame.append(float(extrairRolloff(sinal, freqAmostragem, frameLength, overlapLength)))
    arrayFeaturesFrame.append(float(extrairZCR(sinal, frameLength, overlapLength)))
    arrayFeaturesFrame.extend(extrairMFCCs(extrairMatrizMFCC(sinal, freqAmostragem)))
    arrayFeaturesFrame.extend(extrairMelEspectrograma(sinal, freqAmostragem))
    arrayFeaturesFrame.extend(extrairCromagramas(sinal, freqAmostragem, frameLength, overlapLength))
    
    return np.array(arrayFeaturesFrame)



#----------------------------------------------------------------------
# DAQUI PRA BAIXO SAO AS FUNCOES QUE EXTRAEM CADA UMA DAS FEATURES
#----------------------------------------------------------------------

def extrairRMS(sinal, frameLength, overlapLength):
    return librosa.feature.rms(y=sinal, frame_length=frameLength, hop_length=overlapLength)

def extrairCentroideEspectral(sinal, freqAmostragem, frameLength, overlapLength):
    return librosa.feature.spectral_centroid(y=sinal, sr=freqAmostragem, n_fft=frameLength, hop_length=overlapLength)

def extrairLarguraBanda(sinal, freqAmostragem, frameLength, overlapLength):
    return librosa.feature.spectral_bandwidth(y=sinal, sr= freqAmostragem, n_fft=frameLength, hop_length=overlapLength)

def extrairPlanicidade(sinal, frameLength, overlapLength):
    return librosa.feature.spectral_flatness(y=sinal, n_fft=frameLength, hop_length=overlapLength)

def extrairRolloff(sinal, freqAmostragem, frameLength, overlapLength):
    return librosa.feature.spectral_rolloff(y=sinal, sr= freqAmostragem, n_fft=frameLength, hop_length=overlapLength)

def extrairZCR(sinal, frameLength, overlapLength):
    return librosa.feature.zero_crossing_rate(y=sinal, frame_length=frameLength, hop_length=overlapLength)

def extrairMatrizMFCC(sinal, freqAmostragem):
    return librosa.feature.mfcc(y=sinal, sr=freqAmostragem)

def extrairMFCCs(matrizMFCC):
    
    arrayMFCCs = []

    for linha in matrizMFCC:
        arrayMFCCs.append(np.mean(linha))

    return arrayMFCCs

def extrairMelEspectrograma(sinal, freqAmostragem):    
    matrizMelEspectrograma = librosa.feature.melspectrogram(y=sinal, sr=freqAmostragem, n_mels=20)

    arrayMelEspectrograma = []

    for linha in matrizMelEspectrograma:
        arrayMelEspectrograma.append(np.mean(linha))

    return arrayMelEspectrograma

def extrairCromagramas(sinal, freqAmostragem, frameLength, overlapLength):

    matrizCromagramas = librosa.feature.chroma_stft(y=sinal, sr=freqAmostragem, n_fft=frameLength, hop_length=overlapLength)

    arrayCromagramas = []

    for linha in matrizCromagramas:
        arrayCromagramas.append(np.mean(linha))

    return arrayCromagramas











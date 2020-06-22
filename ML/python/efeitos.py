import librosa
import numpy as np

def normalizarEnergia(sinalOriginal):
	return sinalOriginal/np.std(sinalOriginal)

def adicionarRuidoBrancoAleatoriamente(sinalOriginal, snrMinima=15, snrMaxima=30):
    # Função para adição de ruído branco com SNR aleatória
	# A distribuição da SNR será uniforme entre 10 dB e 20 dB.
    snr          = np.random.uniform(snrMinima, snrMaxima)
    energiaRuido = np.var(sinalOriginal)/(10**(snr/10))
    ruido        = np.random.normal(0, 1, size=len(sinalOriginal)) * energiaRuido**(1/2)
    return sinalOriginal + ruido, snr

def deslocarTempoAleatoriamente(sinalOriginal, percentualMinimo=0.10, percentualMaximo=0.90):
	# A distribuição do percentual do tamanho do sinal original que será deslocado também será uniforme. O percentual pode ser negativo ou positivo, indicando se o sinal resultante será adiantado ou atrasado em relação ao original. A distribuição ficará entre -0.4 e +0.4.
    percentualDeslocamento = np.random.uniform(percentualMinimo, percentualMaximo)
    qtdAmostrasDeslocadas  = int(np.absolute(percentualDeslocamento) * len(sinalOriginal))
    return np.append(sinalOriginal[qtdAmostrasDeslocadas:], sinalOriginal[0:qtdAmostrasDeslocadas]), percentualDeslocamento

def alterarVelocidadeAleatoriamente(sinalOriginal, velocidadeMinima=0.9, velocidadeMaxima=1.1):
    # A velocidade será uniforme entre 0.5 e 2.
    velocidade = np.random.uniform(velocidadeMinima, velocidadeMaxima)
    while velocidade > 0.975 and velocidade < 1.025:
    	velocidade = np.random.uniform(velocidadeMinima, velocidadeMaxima)
    return librosa.effects.time_stretch(sinalOriginal, velocidade), velocidade

def mudarPitchAleatoriamente(sinalOriginal, freqAmostragem, pitchMinimo=-2, pitchMaximo=2):
    # O pitch terá distribuição uniforme entre -7 e 7.
    pitch = np.random.uniform(pitchMinimo, pitchMaximo)
    while pitch > -0.5 and pitch < 0.5:
    	pitch = np.random.uniform(pitchMinimo, pitchMaximo)
    return librosa.effects.pitch_shift(sinalOriginal, freqAmostragem, pitch), pitch

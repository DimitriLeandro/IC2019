import csv
import os
import librosa
import numpy
from scipy.stats import skew, kurtosis

#-----------------------------------------------------------------------------------
# DEFININDO ONDE SERA FEITA A EXTRACAO DE FEATURES
profundidadeBits = "8bits"
freqAmostragem = 8

#-----------------------------------------------------------------------------------
# DEFINICAO DE FUNCOES
def melGibson(y, sr):
	melGibson = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
	mfcc = []

	for linha in melGibson:
		mfcc.append(numpy.mean(linha))

	return mfcc

def mediaRMS(y):
	return numpy.mean(librosa.feature.rms(y=y))

def centroideEspectral(y, sr):
	return numpy.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

def larguraEspectral(y, sr):
	return numpy.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

def contrasteEspectral(y, sr):
	array_contrastes = []
	fmin = 0.5 * sr * 2**(-6)
	
	S = numpy.abs(librosa.stft(y))
	spectral_contrast = librosa.feature.spectral_contrast(S=S, sr=sr, fmin=fmin)
	
	for linha in spectral_contrast:
		array_contrastes.append(numpy.mean(linha))
		
	return array_contrastes

def planicidadeEspectral(y): 
	return numpy.mean(librosa.feature.spectral_flatness(y=y))

def rolloff(y, sr):
	return numpy.mean( librosa.feature.spectral_rolloff(y=y, sr=sr))

def cruzamentosZero(y):
	return numpy.mean(librosa.feature.zero_crossing_rate(y))

def assimetria(y):
	return skew(y)

def curtose(y):
	return kurtosis(y)

def variancia(y):
	return numpy.var(y)

def main():
	# ABRINDO O CSV COM OS NOMES DOS AUDIOS
	with open("metadata/UrbanSound8K.csv") as metadata:
		objReadCSV = csv.reader(metadata, delimiter=',')
		next(objReadCSV)
		
		# ABRINDO O CSV ONDE FICARAO AS FEATURES
		caminhoCSV = "conversoes/"
		caminhoCSV += profundidadeBits
		caminhoCSV += "/"
		caminhoCSV += str(freqAmostragem) + "k"
		caminhoCSV += "/features_"
		caminhoCSV += profundidadeBits
		caminhoCSV += "_"
		caminhoCSV += str(freqAmostragem) + "k"
		caminhoCSV += ".csv"
		
		# REMOVENDO ALGUM CSV SE JA EXISTIR
		if os.path.exists(caminhoCSV):
			os.remove(caminhoCSV)
		
		with open(caminhoCSV, 'a') as csvFeatures:
			objWriteCSV = csv.writer(csvFeatures)
			
			# ESCREVENDO O CABEÇALHO DO CSV
			cabecalho = []
			cabecalho.append("pasta")
			cabecalho.append("arquivo")
			cabecalho.append("mfcc0")
			cabecalho.append("mfcc1")
			cabecalho.append("mfcc2")
			cabecalho.append("mfcc3")
			cabecalho.append("mfcc4")
			cabecalho.append("mfcc5")
			cabecalho.append("mfcc6")
			cabecalho.append("mfcc7")
			cabecalho.append("mfcc8")
			cabecalho.append("mfcc9")
			cabecalho.append("mfcc10")
			cabecalho.append("mfcc11")
			cabecalho.append("mfcc12")
			cabecalho.append("rms")
			cabecalho.append("centroide")
			cabecalho.append("largura")
			cabecalho.append("contraste0")
			cabecalho.append("contraste1")
			cabecalho.append("contraste2")
			cabecalho.append("contraste3")
			cabecalho.append("contraste4")
			cabecalho.append("contraste5")
			cabecalho.append("contraste6")
			cabecalho.append("planicidade")
			cabecalho.append("rolloff")
			cabecalho.append("zcr")
			cabecalho.append("assimetria")
			cabecalho.append("curtose")
			cabecalho.append("variancia")
			cabecalho.append("classe")
			
			objWriteCSV.writerow(cabecalho)
		
			# PRA CADA AUDIO EU VERIFICO SE E DA PASTA QUE EU QUERO
			cont = 0
			for audio in objReadCSV:
				if audio[6] != '0' and audio[6] != '2':

					cont += 1

					# PEGANDO O NOME DO ARQUIVO E A CLASSE ATRIBUIDA
					arquivo = "conversoes/"
					arquivo += profundidadeBits
					arquivo += "/"
					arquivo += str(freqAmostragem) + "k"
					arquivo += "/"
					arquivo += "fold"
					arquivo += audio[5]
					arquivo += "/"
					arquivo += audio[0]
					
					classe = audio[6]
					
					print("Abrindo o arquivo", cont, "de 6732")
					
					# ABRINDO O AUDIO ATUAL
					y, sr = librosa.load(arquivo, sr=freqAmostragem*1000)

					# EXTRAINDO AS FEATURES
					arrayFeatures = []
					
					arrayFeatures.append(audio[5])
					arrayFeatures.append(audio[0])
					arrayFeatures += melGibson(y, sr)
					arrayFeatures.append(mediaRMS(y))
					arrayFeatures.append(centroideEspectral(y, sr))
					arrayFeatures.append(larguraEspectral(y, sr))
					arrayFeatures += contrasteEspectral(y, sr)
					arrayFeatures.append(planicidadeEspectral(y))
					arrayFeatures.append(rolloff(y, sr))
					arrayFeatures.append(cruzamentosZero(y))
					arrayFeatures.append(assimetria(y))
					arrayFeatures.append(curtose(y))
					arrayFeatures.append(variancia(y))
					arrayFeatures.append(classe)
					
					arrayFeatures = numpy.array(arrayFeatures)
					
					objWriteCSV.writerow(arrayFeatures)

main()
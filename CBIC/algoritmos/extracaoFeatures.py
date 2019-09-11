import csv
import os
import librosa
import numpy
from scipy.stats import skew, kurtosis

#-----------------------------------------------------------------------------------
# DEFININDO O CAMINHO PARA OS ARQUIVOS E OS CSVs

# CAMINHO PARA O CSV QUE VEIO JUNTO COM O DATASET, 
# PRECISO DELE PRA LER O NOME DOS ARQUIVOS E DAS PASTAS
metadataOriginal = "metadata/UrbanSound8K.csv" 

# ESSAS VARIAVEIS SERAO USADAS PARA COMPOR O PATH DE ALGUMAS COISAS
# A FREQ DE AMOSTRAGEM TEM QUE ESTAR DIVIDIDA POR MIL
profundidadeBits = "16bits"
freqAmostragem = 48

FRAME_TIME = 200 	# milissegundos
OVERLAP_TIME = 100 	# milissegundos

FRAME_LENGTH = freqAmostragem * FRAME_TIME		# samples
OVERLAP_LENGTH = freqAmostragem * OVERLAP_TIME	# samples

# O CAMINHO PRO CSV COM AS FEATURES SERA ESCRITO NESSA VARIAVEL
caminhoCSV = "conversoes/"
caminhoCSV += profundidadeBits
caminhoCSV += "/"
caminhoCSV += str(freqAmostragem) + "k"
caminhoCSV += "/features_"
caminhoCSV += profundidadeBits
caminhoCSV += "_"
caminhoCSV += str(freqAmostragem) + "k"
caminhoCSV += ".csv"

#-----------------------------------------------------------------------------------
# DEFINICAO DE FUNCOES
def melGibson(paixaoDeCristo):
	mfcc = []

	for linha in paixaoDeCristo:
		mfcc.append(numpy.mean(linha))

	return mfcc

def melGibsonStd(paixaoDeCristo):
	mfcc_std = []

	for linha in paixaoDeCristo:
		mfcc_std.append(numpy.std(linha))

	return mfcc_std

def delta1(coracaoValente):
	delta1 = librosa.feature.delta(coracaoValente, mode='nearest')

	arrayDelta1 = []

	for linha in delta1:
		arrayDelta1.append(numpy.mean(linha))

	return arrayDelta1

def delta1Std(coracaoValente):
	delta1 = librosa.feature.delta(coracaoValente, mode='nearest')

	arrayDelta1_std = []

	for linha in delta1:
		arrayDelta1_std.append(numpy.std(linha))

	return arrayDelta1_std

def delta2(maquinaMortifera):
	delta2 = librosa.feature.delta(maquinaMortifera, order=2, mode='nearest')

	arrayDelta2 = []

	for linha in delta2:
		arrayDelta2.append(numpy.mean(linha))

	return arrayDelta2

def delta2Std(maquinaMortifera):
	delta2 = librosa.feature.delta(maquinaMortifera, order=2, mode='nearest')

	arrayDelta2_std = []

	for linha in delta2:
		arrayDelta2_std.append(numpy.mean(linha))

	return arrayDelta2_std

def mediaRMS(y):
	return numpy.mean(librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=OVERLAP_LENGTH))

def centroideEspectral(y, sr):
	return numpy.mean(librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=OVERLAP_LENGTH))

def larguraEspectral(y, sr):
	return numpy.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=OVERLAP_LENGTH))

def contrasteEspectral(y, sr):
	array_contrastes = []
	fmin = 0.5 * sr * 2**(-6)
	
	S = numpy.abs(librosa.stft(y))
	spectral_contrast = librosa.feature.spectral_contrast(S=S, sr=sr, fmin=fmin, n_fft=FRAME_LENGTH, hop_length=OVERLAP_LENGTH)
	
	for linha in spectral_contrast:
		array_contrastes.append(numpy.mean(linha))
		
	return array_contrastes

def planicidadeEspectral(y): 
	return numpy.mean(librosa.feature.spectral_flatness(y=y, n_fft=FRAME_LENGTH, hop_length=OVERLAP_LENGTH))

def rolloff(y, sr):
	return numpy.mean( librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=OVERLAP_LENGTH))

def cruzamentosZero(y):
	return numpy.mean(librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=OVERLAP_LENGTH))

def assimetria(y):
	return skew(y)

def curtose(y):
	return kurtosis(y)

def variancia(y):
	return numpy.var(y)


# FUNCAO MAIN QUE VAI EXTRAIR AS FEATURES E CRIAR O CSV------------------
def main():
	# ABRINDO O CSV COM OS NOMES DOS AUDIOS
	with open(metadataOriginal) as metadata:
		objReadCSV = csv.reader(metadata, delimiter=',')
		next(objReadCSV)
		
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
			cabecalho.append("mfcc0_std")
			cabecalho.append("mfcc1_std")
			cabecalho.append("mfcc2_std")
			cabecalho.append("mfcc3_std")
			cabecalho.append("mfcc4_std")
			cabecalho.append("mfcc5_std")
			cabecalho.append("mfcc6_std")
			cabecalho.append("mfcc7_std")
			cabecalho.append("mfcc8_std")
			cabecalho.append("mfcc9_std")
			cabecalho.append("mfcc10_std")
			cabecalho.append("mfcc11_std")
			cabecalho.append("mfcc12_std")
			cabecalho.append("Delta 0")
			cabecalho.append("Delta 1")
			cabecalho.append("Delta 2")
			cabecalho.append("Delta 3")
			cabecalho.append("Delta 4")
			cabecalho.append("Delta 5")
			cabecalho.append("Delta 6")
			cabecalho.append("Delta 7")
			cabecalho.append("Delta 8")
			cabecalho.append("Delta 9")
			cabecalho.append("Delta 10")
			cabecalho.append("Delta 11")
			cabecalho.append("Delta 12")
			cabecalho.append("Delta 0_std")
			cabecalho.append("Delta 1_std")
			cabecalho.append("Delta 2_std")
			cabecalho.append("Delta 3_std")
			cabecalho.append("Delta 4_std")
			cabecalho.append("Delta 5_std")
			cabecalho.append("Delta 6_std")
			cabecalho.append("Delta 7_std")
			cabecalho.append("Delta 8_std")
			cabecalho.append("Delta 9_std")
			cabecalho.append("Delta 10_std")
			cabecalho.append("Delta 11_std")
			cabecalho.append("Delta 12_std")
			cabecalho.append("Delta Delta 0")
			cabecalho.append("Delta Delta 1")
			cabecalho.append("Delta Delta 2")
			cabecalho.append("Delta Delta 3")
			cabecalho.append("Delta Delta 4")
			cabecalho.append("Delta Delta 5")
			cabecalho.append("Delta Delta 6")
			cabecalho.append("Delta Delta 7")
			cabecalho.append("Delta Delta 8")
			cabecalho.append("Delta Delta 9")
			cabecalho.append("Delta Delta 10")
			cabecalho.append("Delta Delta 11")
			cabecalho.append("Delta Delta 12")
			cabecalho.append("Delta Delta 0_std")
			cabecalho.append("Delta Delta 1_std")
			cabecalho.append("Delta Delta 2_std")
			cabecalho.append("Delta Delta 3_std")
			cabecalho.append("Delta Delta 4_std")
			cabecalho.append("Delta Delta 5_std")
			cabecalho.append("Delta Delta 6_std")
			cabecalho.append("Delta Delta 7_std")
			cabecalho.append("Delta Delta 8_std")
			cabecalho.append("Delta Delta 9_std")
			cabecalho.append("Delta Delta 10_std")
			cabecalho.append("Delta Delta 11_std")
			cabecalho.append("Delta Delta 12_std")
			cabecalho.append("rms")
			cabecalho.append("centroide")
			cabecalho.append("largura")
			#cabecalho.append("contraste0")
			#cabecalho.append("contraste1")
			#cabecalho.append("contraste2")
			#cabecalho.append("contraste3")
			#cabecalho.append("contraste4")
			#cabecalho.append("contraste5")
			#cabecalho.append("contraste6")
			cabecalho.append("planicidade")
			cabecalho.append("rolloff")
			cabecalho.append("zcr")
			cabecalho.append("assimetria")
			cabecalho.append("curtose")
			cabecalho.append("variancia")
			cabecalho.append("classe")
			
			objWriteCSV.writerow(cabecalho)
		
			# PRA CADA AUDIO EU EXTRAIO AS FEATURES, DESSA VEZ 
			# VAMOS PEGAR TODAS AS CLASSES
			cont = 0
			for audio in objReadCSV:
				cont += 1
				print("Arquivo", cont, "de 8732")

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
				
				# ABRINDO O AUDIO ATUAL
				y, sr = librosa.load(arquivo, sr=freqAmostragem*1000)

				# EXTRAINDO AS FEATURES
				arrayFeatures = []

				# PRA TIRAR O DELTA E O DELTA DELTA, VOU PRECISAR O MFCC EM FORMA DE MATRIZ
				mfccMatriz = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=FRAME_LENGTH, hop_length=OVERLAP_LENGTH)
				
				arrayFeatures.append(audio[5])
				arrayFeatures.append(audio[0])
				arrayFeatures += melGibson(mfccMatriz)
				arrayFeatures += melGibsonStd(mfccMatriz)
				arrayFeatures += delta1(mfccMatriz)
				arrayFeatures += delta1Std(mfccMatriz)
				arrayFeatures += delta2(mfccMatriz)
				arrayFeatures += delta2Std(mfccMatriz)
				arrayFeatures.append(mediaRMS(y))
				arrayFeatures.append(centroideEspectral(y, sr))
				arrayFeatures.append(larguraEspectral(y, sr))
				#arrayFeatures += contrasteEspectral(y, sr)
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

# ESSE ARQUIVO TEM COMO OBJETIVO ENTRAR NA PASTA COM OS TESTES FEITOS COM CADA SNR PARA RENOMEAR OS CSVS E COLOCA-LOS NA PASTA DA IC
# NAO TA NEM UM POUCO BONITO 
# NAO ME JULGUE, EU SO QUERIA QUE FUNCONASSE LOGO, NAO QUE FICASSE OTIMIZADO OU BONITO

import os
from shutil import copyfile

# DEFININDO A PASTA ONDE ESTAO OS CSVS E PARA ONDE ELES DEVERAO SER COPIADOS
diretorioOrigem  = "/home/dimi/Downloads/Datasets/SESA/SESA_Normalizado/outros/RBRANCO/"
diretorioDestino = "/home/dimi/Programming/IC2019/ML/datasets/SESA/SESA_Normalizado/outros/RBRANCO/" 

# LISTANDO O QUE TEM DENTRO DO DIRETORIO DE ORIGEM (QUERO AS PASTAS "SNR_AdB")
diretoriosSNRsOrigem = os.listdir(diretorioOrigem)

# VOU PASSAR POR CADA COISA QUE TIVER DENTRO DO DIRETORIO DE ORIGEM
for diretorioSNRAtual in diretoriosSNRsOrigem:

	# SE FOR PASTA, EU PROSSIGO COM O ALGORITMO
	if os.path.isdir(diretorioOrigem + diretorioSNRAtual) == True:

		# EU VOU PRECISAR COPIAR OS ARQUIVOS PRA PASTA DE DESTINO, VOU VERIFICAR SE LA JA TEM UM DIRETORIO COM A SNR ATUAL
		if os.path.isdir(diretorioDestino + diretorioSNRAtual):
			print("O diretorio", diretorioSNRAtual, " já existe na pasta de destino. Pulando essa SNR.")
			continue

		# SE CHEGOU AQUI EU POSSO CRIAR O DIRETORIO NO DESTINO
		os.mkdir(diretorioDestino + diretorioSNRAtual)
		
		# ENTRO NA PASTA SEM BEAMFORMING------------------------------------------------------------------------------------------------------------------------------------
		diretorioSemBeamforming = diretorioOrigem + diretorioSNRAtual + "/testeSemBeamforming/"

		# PEGO O CSV
		arquivoCSV = [filename for filename in os.listdir(diretorioSemBeamforming) if filename.endswith(".csv")]
		
		# VERIFICANDO SE SO TEM UM CSV MESMO
		if len(arquivoCSV) != 1:
			print("O diretorio", diretorioSNRAtual, " sem beamforming possui uma quantidade não esperada de CSVs. Esperando 1 arquivo apenas. Pulando essa pasta.")
			continue

		# RENOMEANDO O CSV DESSA PASTA
		novoNome = "teste_semBeamforming_" + diretorioSNRAtual + "_semPCA.csv"
		os.rename(diretorioSemBeamforming + str(arquivoCSV[0]), diretorioSemBeamforming + novoNome)

		# COPIANDO O ARQUIVO RENOMEADO PARA O DESTINO
		copyfile(diretorioSemBeamforming + novoNome, diretorioDestino + diretorioSNRAtual + "/" + novoNome)

		# REPETINDO A MESMA COISA PARA A PASTA COM BEAMFORMING-----------------------------------------------------------------------------------------------------------------
		diretorioBeamforming = diretorioOrigem + diretorioSNRAtual + "/testeBeamforming/"
		arquivoCSV = [filename for filename in os.listdir(diretorioBeamforming) if filename.endswith(".csv")]
		if len(arquivoCSV) != 1:
			print("O diretorio", diretorioSNRAtual, "com beamforming possui uma quantidade não esperada de CSVs. Esperando 1 arquivo apenas. Pulando essa pasta.")
			continue
		novoNome = "teste_beamforming_" + diretorioSNRAtual + "_semPCA.csv"
		os.rename(diretorioBeamforming + str(arquivoCSV[0]), diretorioBeamforming + novoNome)
		copyfile(diretorioBeamforming + novoNome, diretorioDestino + diretorioSNRAtual + "/" + novoNome)

		# AGORA PRO GSC---------------------------------------------------------------------------------------------------------------------------------------------------------
		diretorioGSC = diretorioOrigem + diretorioSNRAtual + "/testeGSC/"
		arquivoCSV = [filename for filename in os.listdir(diretorioGSC) if filename.endswith(".csv")]
		if len(arquivoCSV) != 1:
			print("O diretorio", diretorioSNRAtual, "com GSC possui uma quantidade não esperada de CSVs. Esperando 1 arquivo apenas. Pulando essa pasta.")
			continue
		novoNome = "teste_GSC_" + diretorioSNRAtual + "_semPCA.csv"
		os.rename(diretorioGSC + str(arquivoCSV[0]), diretorioGSC + novoNome)
		copyfile(diretorioGSC + novoNome, diretorioDestino + diretorioSNRAtual + "/" + novoNome)

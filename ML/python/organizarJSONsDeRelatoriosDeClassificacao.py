# ESSE ARQUIVO TEM COMO OBJETIVO ORGANIZAR OS JSONS DE RELATORIOS DE CLASSIFICACAO
# QUANDO EU RODO O JUPYTER QUE CRIA E USA OS CLASSIFICADORES, OS JSONS COM OS RESULTADOS
# FICAM NA PASTA ONDE ESTA O CSV DO DATASET
# A IDEIA AQUI E PEGAR ESSE MONTE DE JSON E COLOCAR NA PASTA DE RELATORIOS DE CLASSIFICACAO 
# PRA FICAR MAIS ORGANIZADO

import os
from shutil import copyfile

# DEFININDO A PASTA ONDE ESTAO OS JSONS E PARA ONDE ELES DEVERAO SER COPIADOS
diretorioOrigem  = "/home/dimi/Programming/IC2019/ML/datasets/SESA/SESA_Normalizado/outros/RBRANCO/"
diretorioDestino = "/home/dimi/Programming/IC2019/ML/relatorios_classificacao/SESA_Normalizado/data_augmentation/ruido_branco/com_gridsearch/" 

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

		# PEGO OS JSONS NA ORIGEM
		arquivosJSON = [filename for filename in os.listdir(diretorioOrigem + diretorioSNRAtual) if filename.endswith(".json")]
		
		# PARA CADA ARQUIVO
		for jsonAtual in arquivosJSON:

			# COPIANDO O ARQUIVO PARA O DESTINO
			copyfile(diretorioOrigem + diretorioSNRAtual + "/" + jsonAtual, diretorioDestino + diretorioSNRAtual + "/" + jsonAtual)

			# DELETANDO O ARQUIVO NA ORIGEM
			os.remove(diretorioOrigem + diretorioSNRAtual + "/" + jsonAtual)
		
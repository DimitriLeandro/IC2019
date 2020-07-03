#----------------------------------------------------------------------
# ESSE ARQUIVO NAO IMPLEMENTA UMA CLASSE, VAMOS USAR FUNCOES SOLTAS
#----------------------------------------------------------------------
# VOU APENAS ADAPTAR O CODIGO DA CLASSE DaSParallel PARA NAO SER MAIS 
# UMA CLASSE E TAMBEM PARA NAO PARALELIZAR O DELAY AND SUM, JA QUE OS
# TESTES REVELARAM QUE A PARALELIZACAO NA VERDADE ESTAVA DEMORANDO MAIS
import numpy as np


# ESSA E A FUNCAO PRINCIPAL QUE VAI RECEBER O ARRAY COM OS SINAIS
# DE CADA MICROFONE E RETORNAR O SINAL BEAMFORMADO
def delayAndSum(arraySinaisMics, delayMax=20, fazerMedia=True):

    # CALCULANDO O ARRAY DE DELAYS
    arrayDelays = calcularArrayDelays(arraySinaisMics, delayMax)

    # CONSTRUINDO O SINAL BEAMFORMADO
    sinalDaS = executarDaS(arraySinaisMics, arrayDelays, delayMax, fazerMedia)
    
    return sinalDaS
    
    
    
#----------------------------------------------------------------------
# DAQUI PRA BAIXO SAO AS FUNCOES AUXILIARES
#----------------------------------------------------------------------

def calcularCorrelacao(sinalA, sinalB):
    return np.corrcoef(sinalA, sinalB)[0][1]

def ajustarSinaisDadoDelay(sinalA, sinalB, delayDesejado, delayMax):
    # Função para ajustar dois sinais dado um delay a ser testado
    # Supondo que eu determine que um delay entre dois sinais é -4. 
    # Quero reconstruir os sinais para testar a correlação entre eles com esse delay.
    inicioB = delayMax - delayDesejado
    fimB    = -delayMax - delayDesejado
    
    if fimB == 0:
        fimB = None
        
    return sinalA[delayMax:-delayMax], sinalB[inicioB:fimB]
  
    
def calcularDefasagemEntreDoisSinais(sinalA, sinalB, delayMax):
    # Função para calcular a defasagem entre dois sinais quaisquer
    # Como foi observado que o gráfico Correlação x Delay é sempre 
    # quadrático com concavidade para baixo, essa função tenta calcular
    # o menor número de correlações quanto possível.
    # Em primeiro lugar, ela calcula as correlações para os delays -1, 0 e 1. 
    # Com isso, ela vai saber em qual direção seguir. Isto é, os delays 
    # aumentam pra direita ou para a esquerda do gráfico? 
    # Sabendo isso, ela segue na direção em que as correlações aumentam e 
    # só para quando uma das correlações for mais baixa que a iteração anterior, 
    # pois, dessa forma, sabe-se que o pico foi encontrado.

    # COMECO CALCULANDO COM OS DELAYS -1, 0 E 1. DEPOIS EU VEJO PRA QUAL DIRECAO EU VOU
    sinalAAjustado, sinalBAjustado = ajustarSinaisDadoDelay(sinalA, sinalB, -1, delayMax)
    correlacaoMenosUm              = calcularCorrelacao(sinalAAjustado, sinalBAjustado)
    
    sinalAAjustado, sinalBAjustado = ajustarSinaisDadoDelay(sinalA, sinalB, 0, delayMax)
    correlacaoZero                 = calcularCorrelacao(sinalAAjustado, sinalBAjustado)
    
    sinalAAjustado, sinalBAjustado = ajustarSinaisDadoDelay(sinalA, sinalB, 1, delayMax)
    correlacaoMaisUm               = calcularCorrelacao(sinalAAjustado, sinalBAjustado)

    # VERIFICANDO PRA QUAL DIRECAO EU VOU
    if correlacaoZero > correlacaoMenosUm and correlacaoZero > correlacaoMaisUm:
        # SE ENTROU AQUI E PQ DELAY 0 DA A MAIOR CORRELACAO
        return 0
    elif correlacaoMaisUm > correlacaoZero and correlacaoMaisUm > correlacaoMenosUm:
        # SE ENTROU AQUI A GNT VAI PRA CIMA
        maiorCorrelacao      = correlacaoMaisUm
        delayMaiorCorrelacao = 1
        rangeDelays          = np.arange(2, delayMax + 1, 1)
    else:
        #correlacaoMenosUm > correlacaoZero and correlacaoMenosUm > correlacaoMaisUm:
        # SE ENTROU AQUI A GNT VAI PRA BAIXO
        maiorCorrelacao      = correlacaoMenosUm
        delayMaiorCorrelacao = -1
        rangeDelays          = np.arange(-2, -delayMax -1, -1)

    # SE A FUNCAO CHEGAR ATE AQUI E PQ AINDA NAO HOUVE RETORNO. VAMOS CONTINUAR TESTANDO OS DELAYS
    for delayAtual in rangeDelays:

        # CALCULO A CORRELACAO COM O DELAY ATUAL
        sinalAAjustado, sinalBAjustado = ajustarSinaisDadoDelay(sinalA, sinalB, delayAtual, delayMax)
        correlacaoAtual = calcularCorrelacao(sinalAAjustado, sinalBAjustado)

        # JA QUE ESTAMOS INDO NA DIRECAO EM QUE A CORRELACAO AUMENTA, 
        # SE ELA DIMINUIR E PQ A GNT TINHA CHEGADO NO PICO NA ITERACAO ANTERIOR
        if correlacaoAtual < maiorCorrelacao:
            break

        # SE NAO DIMINUIU ENTAO SEGUE O BAILE
        maiorCorrelacao      = correlacaoAtual
        delayMaiorCorrelacao = delayAtual

    return delayMaiorCorrelacao    
    
    
def calcularArrayDelays(arraySinaisMics, delayMax):
    # Essa função vai usar a função "calcularDefasagemEntreDoisSinais" 
    # para calcular o delay entre cada microfofe e o microfone de origem.
    
    # PARA CADA MIC QUE NAO SEJA O REFERENCIAL
    arrayDelays = [0]
    for sinalAtual in arraySinaisMics[1:]:
        arrayDelays.append(calcularDefasagemEntreDoisSinais(arraySinaisMics[0], sinalAtual, delayMax))

    return arrayDelays


def executarDaS(arraySinaisMics, arrayDelays, delayMax, fazerMedia=True):
    
    # AJUSTANDO OS SINAIS PARA QUE POSSAM SER SOMADOS DEPOIS    
    arraySinaisAjustados = []
    for sinalAtual, delayAtual in zip(arraySinaisMics, arrayDelays):
        __, sinalBAjustado = ajustarSinaisDadoDelay(arraySinaisMics[0], sinalAtual, delayAtual, delayMax)
        arraySinaisAjustados.append(sinalBAjustado)

    # SOMANDO OS SINAIS AJUSTADOS
    sinalDaS = np.sum(arraySinaisAjustados, axis=0)

    # FAZENDO A MEDIA CASO DESEJADO
    if fazerMedia == True:
        sinalDaS = sinalDaS/len(arraySinaisMics)
        
    return sinalDaS   
    
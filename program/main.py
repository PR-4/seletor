#Seletor
#programa principal que calcula o passey para seleção de amostras da seção geradora
#Autor: Carreira(2023)

#Pacotes
import numpy as np
import pandas as pd
import time as t
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image
#Pacotes internos
import sys
sys.path.insert(0,'../modules')
import debug as d 
import equations as cot


#-------------------------------------------#
#    Pré-processamento  (Análise de dados)  #
#-------------------------------------------#


# leitura do dado de entrada e criação do dataframe

poco = pd.read_csv('../input/1BRSA1007RJS.csv', sep=';', low_memory = False, header=0, index_col=1) 

# Retira a segunda linha
poco.drop(axis=0)


# Mostra as 5 primeiras linhas do arquivo de entrada

print(poco.head(10))


################## Dicionário de variáveis ###################
# wellName: nome do poço
# dasetName: frame ou corrida associada à aquisição de dados
# TDEP: profundidade [m]
# GR: raio-gama [gAPI]
# DT: tempo de trânsito [-]
# MDT: tempo de trânsito [us/ft]
# RHOB: densidade da formação [g/cm³]
# RT06 -> RT90: resistividade da formação [Ohm.m]
#############################################################

# Seleciona a geradora pré-sal do poço (G. Lagoa Feia - F. Atafona)
topo = 5420.0
base = 5600.0
poco=poco[(poco['TDEP'].astype(float) >= topo) & (poco['TDEP'].astype(float) <= base)] # alvo! 

# Retira dados advindos de desmoronamento
#Parametros do poço:
# diametro = 9 pol
# lam = +/- 2 pol (limite aceitavel maximo)
#dim = 9
#delta = 2
#ls = dim + delta
#li = dim - delta

# filtragem via caliper:

#poco=poco[(df['Cali'] >= li) & (poco['Cali'] <= ls)]#informação baseada diametro do poco

# Identifica o volume de dados do dataframe

print('Dimensão =', poco.shape)
print("Número de entradas =", poco.shape[0])
print("Número de variáveis =", poco.shape[1])

# Verifica os tipos de variáveis do dataframe
#print(poco.dtypes)

# Conta Nans e retorna porcentagens em ordem decrescente (ordenar em ordem decrescente as variáveis por seus valores ausentes)
print((poco.isnull().sum()/poco.shape[0]).sort_values(ascending=False)*100)


# Substitui os Nans pelo valor da mediana já que o descarte de mais de 50% dos dados pode significar o descarte de informações importantes. 
(poco.isnull().sum()/poco.shape[0]).sort_values(ascending=False)

poco.reviews_per_month.fillna(df.reviews_per_month.median(),inplace=True)


# Estatística básica (histograma, média dos canais, correlação, heatmap)

# histograma
poco.hist(bins=15,figsize=(15,10));


# media


# correlação


# heatmap



#------------------------#
#    Processamento       #
#------------------------#


# Cálculo do DlogR


# Cálculo do COT



#-----------------------#
#        Saída          #
#-----------------------#

# Salva os arquivos de saída








#Seletor
#programa que seleciona uma lista de IDs de poços e armazena em um dataframe

#Pacotes
import numpy as np
import pandas as pd
import time as t



ini = t.time()

# leitura da tabela da ANP
ANP = pd.read_excel(open('../input/tabela-de-pocos.xlsx', 'rb') ,index_col= 0, sheet_name='Planilha1',header=0)

# leitura do arquivo de seleção
lista = pd.read_excel(open('../input/controle_de_entrega_de_dados_v2.xlsx', 'rb'), sheet_name="Lara (11301 e 11302)",header=0)
#lista = ['3-FQ-21-ES','7-CAM-114-RN','4-LD-7-ES']

# seletor

seletor = ANP.loc[lista['POCO'].to_list()]
seletor = seletor.reset_index()
print(seletor)


#saida
seletor.to_excel('../output/selecao.xlsx', index=False, header=True )

fim = t.time()
print('Tempo de processamento=',(fim-ini)/60,'minutos')


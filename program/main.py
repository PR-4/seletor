#Seletor
#programa principal que calcula o passey para seleção de amostras da seção geradora
#Autor: Carreira(2023)

#Pacotes
import numpy as np
import pandas as pd
import time as t
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as st
import random
import pytz
from IPython.display import Image

#Pacotes internos
import sys
sys.path.insert(0,'../modules')
import debug as d 
import equations as eq


#-------------------------------------------#
#    Pré-processamento  (Análise de dados)  #
#-------------------------------------------#


# leitura do dado de entrada e criação do dataframe

poco = pd.read_csv('../input/1BRSA1007RJS.csv', sep=';', low_memory = False, header=0)

# Retira a segunda linha
poco.drop(0, inplace=True)

# Retira a coluna MDT
poco.drop('MDT', axis=1, inplace=True)

# Substitui decimal "," por "."

poco = poco.replace(',','.',regex=True)

# Converte strings to numeric
columns_to_convert = ['TDEP', 'DT', 'GR', 'RHOB', 'RT06', 'RT10', 'RT20', 'RT30', 'RT60', 'RT90']
poco[columns_to_convert] = poco[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Média dos canais de resistividade
columns_to_average = ['RT06','RT10', 'RT20', 'RT30', 'RT60', 'RT90']
poco['RT'] = poco[columns_to_average].mean(axis=1)

# Substitui os Nans pelo valor da mediana já que o descarte de mais de 50% dos dados pode significar o descarte de informações importantes. 
(poco.isnull().sum()/poco.shape[0]).sort_values(ascending=False)
poco.DT.fillna(poco.DT.median(),inplace=True)
poco.RHOB.fillna(poco.RHOB.median(),inplace=True)
poco.GR.fillna(poco.GR.median(),inplace=True)
poco.RT06.fillna(poco.RT06.median(),inplace=True)
poco.RT10.fillna(poco.RT10.median(),inplace=True)
poco.RT20.fillna(poco.RT20.median(),inplace=True)
poco.RT30.fillna(poco.RT30.median(),inplace=True)
poco.RT60.fillna(poco.RT60.median(),inplace=True)
poco.RT90.fillna(poco.RT90.median(),inplace=True)



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

# Retira valores -999.999
poco=poco[(poco['GR'] != -9999) &  (poco['GR'] != -999999.9999)]
#poco=poco[(poco['RT06'] != -9999)]
# Identifica o volume de dados do dataframe

print('Dimensão =', poco.shape)
print("Número de entradas =", poco.shape[0])
print("Número de variáveis =", poco.shape[1])

# Verifica os tipos de variáveis do dataframe
#print(poco.dtypes)

# Conta Nans e retorna porcentagens em ordem decrescente (ordenar em ordem decrescente as variáveis por seus valores ausentes)
print((poco.isnull().sum()/poco.shape[0]).sort_values(ascending=False)*100)


# Mostra as 5 primeiras linhas do arquivo de entrada

print(poco.head(5))


# Estatística básica (histograma, média dos canais, correlação, heatmap)

# histograma
poco.drop('wellName', axis=1, inplace=True)
poco.drop('datasetName', axis=1, inplace=True)
# Convert the column to a numeric data type
poco['GR'] = pd.to_numeric(poco['GR'], errors='coerce')
poco['GR'].plot.hist(bins=15)
plt.show()


# media
# Calculate the mean values
mean_values = poco.mean()

# Plot the mean values as a line plot
#mean_values.plot(kind='line')

# Alternatively, plot the mean values as a bar plot
mean_values.plot(kind='bar')
#plt.show()

# correlação
# Calculate the correlation matrix
correlation_matrix = poco.corr()

# Plot the correlation map using a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Set plot title
plt.title('Correlation Map')
#plt.show()

# heatmap

sns.heatmap(poco)
#plt.show()

#--------------------------------------------------------------#
#                         Processamento                        #
#--------------------------------------------------------------#


# Baseline
GRbaseline = np.min(poco.GR)
DTbaseline = np.min(poco.DT)
RTbaseline = np.min(poco.RT30)
RHOBbaseline = np.min(poco.RHOB)

# calcula os parâmetros da Eq. via mahmoud
cab = ['DR', 'DT', 'GR', 'RHO', 'COT']
data = pd.read_csv('../input/mahmoud.txt', sep='\\s+',skiprows=1, names = cab, usecols=(0,1,2,3,4))
dfm = pd.DataFrame(data)
print(len(dfm))
#separando as informações do dado em variáveis distintas
DRm = dfm[dfm.columns[0]]
DTm = dfm[dfm.columns[1]]
GRm = dfm[dfm.columns[2]]
RHOm = dfm[dfm.columns[3]]
COTm = dfm[dfm.columns[4]]

# Valores máximos e mínimos de cada coluna baseados em mahmoud
dtmax = np.argmax(np.array(dfm['DT']))
dtmin = np.argmin(np.array(dfm['DT']))
###############################
grmax = np.argmax(np.array(dfm['GR']))
grmin = np.argmin(np.array(dfm['GR']))
###############################
resmax = np.argmax(np.array(dfm['DR'])) 
resmin = np.argmin(np.array(dfm['DR']))
###############################
rhomax = np.argmax(np.array(dfm['RHO']))
rhomin = np.argmax(np.array(dfm['RHO']))
###############################
cotmax = np.argmax(np.array(dfm['COT']))
cotmin = np.argmin(np.array(dfm['COT']))


dr = [DRm[0],DRm[1],DRm[5],DRm[6],DRm[10],DRm[11],DRm[15],DRm[16]]
dt = [DTm[0],DTm[1],DTm[5],DTm[6],DTm[10],DTm[11],DTm[15],DTm[16]]
gr = [GRm[0],GRm[1],GRm[5],GRm[6],GRm[10],GRm[11],GRm[15],GRm[16]]
rho = [RHOm[0],RHOm[1],RHOm[5],RHOm[6],RHOm[10],RHOm[11],RHOm[15],RHOm[16]]
cot = [COTm[0],COTm[1],COTm[5],COTm[6],COTm[10],COTm[11],COTm[15],COTm[16]]


# O baseline é geralmente determinado na formação que não é rica em MO e aonde nessa rocha tem as curvas de DT e GR se cruzando. 

resbaseline = RTbaseline # min(dr)/2   #np.sum(dr)/len(dr)
dtbaseline  = DTbaseline # min(dt)/2    #np.sum(dt)/len(dt)
grbaseline  = GRbaseline # min(gr)/2    #np.sum(gr)/len(gr)
rhobbaseline = RHOBbaseline
#print(resbaseline, grbaseline, rhobbaseline)


# Cálculo do dlogR via passey et al. (1990)

dLOGr90 = np.zeros(len(poco.GR))

for i in range (len(dr)):
   dLOGr90[i] = eq.dlogr90(dr[i],resbaseline,rho[i],rhobbaseline)
   #print(dLOGr90[i])



# Monte Carlo com caixa adaptativa por porcentagem variada

# Variáveis do modelo de monte carlo!
p = [1.08,0.95,1.0,0.005,440]
nbox=30
porcentagem = 50
porcentagem = porcentagem/100
npts = 90000
nloop = npts
alfa_otimo=1.08
beta_otimo=0.95
delta_otimo=1.0
eta_otimo=0.005

#Alocação dinâmica
otimo=np.zeros((nbox,5))
beta  = np.zeros(npts)
eta   = np.zeros(npts)
alfa  = np.zeros(npts)
delta = np.zeros(npts)
cot_c = np.zeros((nloop,len(dr)))
phi = np.zeros(nloop)
PHI = np.copy(phi)
cot_verdadeiro = np.copy(cot)



# O monte carlo
for k in range(nbox):
    alfamin  = (alfa_otimo - porcentagem*alfa_otimo) 
    alfamax  = (alfa_otimo + porcentagem*alfa_otimo) 
    betamin  = (beta_otimo - porcentagem*beta_otimo) 
    betamax  = (beta_otimo + porcentagem*beta_otimo) 
    deltamin = (delta_otimo - porcentagem*delta_otimo) 
    deltamax = (delta_otimo + porcentagem*delta_otimo)
    etamin   = (eta_otimo - porcentagem*eta_otimo) 
    etamax   = (eta_otimo + porcentagem*eta_otimo) 
    phi = np.zeros(nloop)
    
    for i in range(nloop):
        beta[i]  = random.uniform(betamin, betamax)
        alfa[i]  = random.uniform(alfamin, alfamax)
        eta[i]   = random.uniform(etamin, etamax)
        delta[i] = random.uniform(deltamin, deltamax)
        #print(alfa[i],beta[i],eta[i],delta[i])
        for j in range(len(dr)):
            cot_c[i,j] = eq.passey16(dLOGr90[j],alfa[i],beta[i],delta[i],eta[i],p[4],gr[j],grbaseline)
            #print(cot_c[i])
            # funcao phi:
            phi[i] += (cot_verdadeiro[j] - cot_c[i,j])**2
            PHI[i] += np.sqrt(np.sum((cot_verdadeiro[j]-cot_c[i,j])**2)/len(cot))
            
    imelhor = np.argmin(phi)  # localizar o indice do menor elemento do vetor phi
    # parametros finais:
    alfa_otimo = alfa[imelhor]
    beta_otimo = beta[imelhor]
    delta_otimo = delta[imelhor]
    eta_otimo = eta[imelhor]
    cot_otimo = cot_c[imelhor]
    print(k,alfa_otimo,beta_otimo,delta_otimo,eta_otimo,phi[imelhor])
     # cria matriz de parâmetros 
    P = np.stack((alfa_otimo,beta_otimo,delta_otimo,eta_otimo))# com o último parâmetro
    otimo[k] = (alfa_otimo,beta_otimo,delta_otimo,eta_otimo,phi[imelhor])# com todos os ótimos parâmetros


#Limites da caixa
print("Caixa eta->",etamin,etamax,eta_otimo)
print("Caixa delta->",deltamin,deltamax, delta_otimo)
print("Caixa beta->",betamin,betamax, beta_otimo)
print("Caixa alfa->",alfamin,alfamax, alfa_otimo)


PHI_abs = np.sqrt(np.sum((cot_verdadeiro-cot_otimo)**2)/len(cot))
print(PHI_abs)


plt.plot(range(nbox),otimo[:,4], 'bo-',label='phi')# valores de eta por phi
plt.title('Convergência')
plt.xlabel('Iterações')
plt.ylabel('Phi')
plt.legend()
plt.savefig('../output/Convergencia.png')


otimo[:,0]


#fig, ax = plt.subplots(2,2, figsize=(15, 8), sharex=True, sharey=True)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=False)
fig.suptitle('Análise dos parâmetros')

ax1.plot(range(nbox),otimo[:,0],'k.-')
ax1.plot(range(nbox),np.full_like(range(nbox),alfamin),'b-',linewidth=10)
ax1.plot(range(nbox),np.full_like(range(nbox),alfamax),'r-',linewidth=2)
ax1.plot(alfa_otimo,'*y')
ax1.set_ylabel('Alfa')

ax2.plot(range(nbox),otimo[:,1],'k.-')
ax2.plot(range(nbox),np.full_like(range(nbox),betamin),'b-',linewidth=10)
ax2.plot(range(nbox),np.full_like(range(nbox),betamax),'r-',linewidth=2)
ax2.plot(beta_otimo,'*y')
ax2.set_ylabel('Beta')

ax3.plot(range(nbox),otimo[:,2],'k.-')
ax3.plot(range(nbox),np.full_like(range(nbox),deltamin),'b-',linewidth=10)
ax3.plot(range(nbox),np.full_like(range(nbox),deltamax),'r-',linewidth=2)
ax3.plot(delta_otimo,'*y')
ax3.set_xlabel('Valor ótimo')
ax3.set_ylabel('Delta')

ax4.plot(range(nbox),otimo[:,3], 'k.-')
ax4.plot(range(nbox),np.full_like(range(nbox),etamin),'b-',linewidth=10)
ax4.plot(range(nbox),np.full_like(range(nbox),etamax),'r-',linewidth=2)
ax4.plot(eta_otimo,'*y')
ax4.set_xlabel('Valor ótimo')
ax4.set_ylabel('Eta')

#plt.legend()
plt.savefig('../output/parametrosMC.png')






# Poço 

deltaLR = eq.dlogr90(poco.RT30,RTbaseline,poco.RHOB,RHOBbaseline)



# Cálculo do COT


cot_calc = eq.passey16(deltaLR,alfa_otimo,beta_otimo,delta_otimo,eta_otimo,p[4],poco.GR,GRbaseline)

mincot=np.min(cot_calc)

#-----------------------#
#        Saída          #
#-----------------------#

# Salva os arquivos de saída


plt.figure(figsize=(20,3))
plt.plot(poco.TDEP,cot_calc-mincot,'b*', markersize=4,label='Monte Carlo')
plt.plot(poco.TDEP,np.zeros(len(poco.TDEP)),color='red')
plt.ylabel('COT')
plt.xlabel('Profundidade (m)')
plt.gca().invert_yaxis()   
plt.legend()
plt.show()
plt.savefig('../image/1BRSA1007RJS.png')





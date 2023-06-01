#!/usr/bin/env python
# coding: utf-8

# ### referencias úteis

# * [Monte Carlo](https://pbpython.com/monte-carlo.html)
# * [(Sukanta,2014)](https://www.ias.ac.in/article/fulltext/reso/019/08/0713-0739)
# 
# * [Feiguin](https://github.com/afeiguin/comp-phys/blob/master/10_01_montecarlo_integration.ipynb) 
# 
# * [Variância](https://mathworld.wolfram.com/Variance.html)

# # Calculando o COT via DlogR por Monte Carlo de caixa adaptável

# ## Importação de módulos externos

# In[1]:


import random
import numpy as np
import math
import random
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import clear_output
from datetime import datetime
import pytz



#Configurando figura e o rc parameters
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")
plt.rcParams["figure.figsize"] = (10,5)
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size = 20)


# ### Funções de trabalho

# In[2]:


#### Funções utilizadas #####
def dlogr(res,resb,x,xb,m):
    '''Função que determina o Delta log R dos pares ordenados de propriedades
    Resistividade e Sônico ou Resistividade ou Densidade. 
    Entradas:
    res, dados de resistividade
    reb, baseline
    x, canal de densidade ou sônico
    xb, baseline da densidade ou sônico
    m, coeficiente de cimentação
    Saída:
    DlogR, Delta Log R'''
    import math
    
    #Recurso computacional para eliminar os zeros:
    dummy = 1e-100000
    
    if np.size(res) > 1:
        dado  = len(res)
        DlogR = np.zeros(dado)
        res   = np.array(res)
        x     = np.array(x)
        resb  = np.min(res)
        xb    = np.median(x)
        for i in range(dado):
            DlogR[i]=math.log10(res[i]/(resb+dummy))+((1/np.log(10))*(m/(x[i]-xb))*(x[i]-xb))
            if x[i]/xb < 0:
                print(x[i]-xb)
                if res[i]/resb < 0:
                    print("Cuidado! Log negativo!",res[i]-resb)
     
    
    
    else:
        res = float(res)
        resb = float(resb)
        x = float(x)
        xb = float(xb)
        DlogR=math.log10(res/(resb+dummy))+((1/np.log(10))*(m/(x-xb))*(x-xb))
        
        
    return DlogR

def dlogr90(res,resb,x,xb):
    
    
    if np.size(res) > 1:
        dado  = len(res)
        DlogR = np.zeros(dado)
        res   = np.array(res)
        x     = np.array(x)
        resb  = np.min(res)
        xb    = np.median(x)
        for i in range(dado):
            DlogR[i]=math.log10(res[i]/(resb))+(0.02*(x[i]-xb))
            if x[i]/xb < 0:
                print(x[i]-xb)
                if res[i]/resb < 0:
                    print("Cuidado! Log negativo!",res[i]-resb)
    else:
        res = float(res)
        resb = float(resb)
        x = float(x)
        xb = float(xb)
        DlogR=math.log10(res/(resb))+(0.02*(x-xb))
        
    return DlogR

#############################

def passey16(drlog,alfa,beta,delta,eta,Tmax,gr,grb):
    '''Função que determina COT via delta log R
        Entradas:
    drlog,parâmetro calculado
    alfa, parâmetro estimado
    beta, parâmetro estimado
    delta, parâmetro estimado
    eta, parâmetro estimado
    Tmax, indicador de maturidade em oC
    gr, canal raio gama
    Saída:
    COT, Conteúdo orgânico total
    '''
    
    if np.size(drlog) > 1:
        dado = len(gr)
        COT  = np.zeros(dado)
        gr   = np.array(gr)
        grb  = np.median(gr) 
        for i in range(dado):
            COT[i] = (alfa*drlog[i] + beta*(gr[i]-grb))*10**(delta-eta*Tmax)
            #print(COT[i],delta-eta*Tmax)
    else:
        gr = float(gr)
        grb = float(grb)
        COT = (alfa*drlog + beta*(gr-grb))*10**(delta-eta*Tmax)
        
    return COT

def DensidadeAparenteSeca(RHOB,NPHI):
    NPHI= NPHI / 100 # Fator de conversão
    DAS=RHOB-NPHI
    return DAS


# ## O método $\Delta_{log}R$ para determinação do COT sintético 

# O método publicado por Passey et al. (1990) foi desenvolvido e testado pela
# Exxon/Esso em 1979 e aplicado desde então em diversos poços pelo mundo. Passey
# et al. (1990) partem do princípio de que as rochas são compostas por três componentes: a matriz, a matéria orgânica sólida e os fluidos que preenchem os poros. 
# 
# O método se baseia no fato de que uma rocha geradora possui respostas nos perfis
# diferentes, mas proporcionais à sua maturação e ao seu conteúdo orgânico. Ou seja,
# o perfil é sensível à variação de COT(Perfil Sônico) e a transformação da matéria
# orgânica sólida em fluido(Perfil de Resistividade).
# Logo, para aplicação do método são necessárias as curvas de perfil resistividade
# e perfil sônico. O perfil sônico deve estar ajustado na escala de -100µs/pé para cada dois ciclos logarítmicos de resistividade. 
# 
# Após isso, uma linha base deverá ser
# definida em um intervalo de uma rocha de granulometria fina não geradora, onde
# as curvas de resistividade e sônico estejam sobrepostas. A separação das
# curvas de Sônico e Resistividade em intervalos de rocha geradora é denominada de
# ∆LogR, que é linearmente relacionada ao teor de COT em função da maturidade
# (PASSEY et al., 1990). 

# ### Determinação do $\Delta_{log}R$ via método de Passey (1990)

# Passey et al (1990) propuseram um método que estima o COT sintético baseada curvas que indicam a escala de porosidade da rocha alvo. Essas informações de porosidade são retiradas de dois perfis o perfil sônico que mede o tempo de trânsito e o perfil de resistividade que infere a resistividade na rocha estudada. As informações de porosidade assiciada a leitura desses dois perfis indicam a aviabilidade de ocorrência de rochas com alto conteúdo de matéria orgância. O conteúdo de querogênio associado ao alto conteúdo de matéria orgânica ocasiona um efeito nos dois perfis mensionados acima. Os efeitos são: baixa no valor de densidade e baixa no valor de tempo de trânsito. Além disso o perfil de resistividade apresenta uma alta na presença do kerogênio em rochas com alta porosoidade. 
# 

# \begin{equation}
# \Delta_{log}R = log_{10} \big{(} \dfrac{R}{R_{baseline}} \big{)} + 0.02 \times (DT - DT_{baseline})
# \end{equation}

# ### Solução genérica da função de Passey e Wang via método para determinação do $\Delta_{log}R$. 

# Wang revisitou a equação de Passey e com base nos teores de vitrinita e curvas de R, DT, RHO e GR cria uma nova relação empírica para os folhelhos devonianos. Note que é possível calcular o $\Delta_{log}R $ de duas maneiras possíveis.
# 
# \begin{equation}
# \Delta_{log}R = log_{10} \big{(} \dfrac{R}{R_{baseline}} \big{)} + \dfrac{1}{ln 10} \dfrac{m}{(DT - DT_{m})} \times (DT - DT_{baseline})
# \end{equation}
# 
# \begin{equation}
# \Delta_{log}R = log_{10} \big{(} \dfrac{R}{R_{baseline}} \big{)} + \dfrac{1}{ln 10 } \dfrac{m}{(RHO_{m} - RHO)} \times (RHO  - RHO_{baseline})
# \end{equation}
# 
# \begin{equation}
# COT = [ \alpha \Delta_{log}R + \beta (GR - GR_{baseline}) ] \times 10^{(\delta-\eta Tmax)}
# \end{equation}
# 
# 
# Onde $DT_{m}$ é o canal sônico medida em tempo de trânsito ($\mu s/ft$), $\textbf{m}$ representa o coeficiente de cimentação, $RHO$ e $RHO_{baseline}$ representam o canal de densidades e sua média em ($g/cm^{3}$), $R$ e $R_{baseline}$ são  o canal de resistividade e o valor de base correspondente, $[\alpha, \beta,\delta,\eta]$ sãos constantes da equação que varia de acordo com a formação geológica e $T_{max}$ é o indicador de maturidade ($^{\circ} C$), $GR$ é o canal de raio gamma e o $GR_{baseline}$ é a média do canal raio gama (API).
# 
# 

# ### Tabela de coeficientes (Mahmud,2015)

# A tabela original publicada é composta 4 subtabelas geradas por quatro modelos de inteligência artificial. Abaixo é apresentada uma dessas tabelas. Esses dados foram utilizados para compilar uma nova tabela para o cálculo dos coeficientes via monte carlo. Esse arquivo foi denominado dado.txt

# Dados (n = 671) |   R  | DT   | GR  | RHOB | COT |
# ----------------|-------|------|-----|------|-----|
# Mínimo          | 4.97  |50.95 |27.37| 2.39 | 0.76| 
# Máximo          |163.6  |97.1  |146.9| 2.7  |  5.1|
# Alcance         |158.6  |  46.1|119.6| 0.3  | 4.4 |
# Desvio Padrão   |39.81  |8.20  |21.63|0.07  |0.96 | 
# Variância       |1585   | 67   |468  |0.0044|0.916|

# # 1) Dados de entrada

# ## 1.1) Poços: 
# 
# Cálculo da COT baseado no chute inicial com os coeficientes retirados dos dados de poços

# In[3]:


#dadode entrada
cab = ['DEPT','RHO8a','RHOZa','RHO8b','RHOZb','ECGRa','ECGRb','SVEL','HRDUa','HRDUb','NPHIa','NPHIb']
datareal = pd.read_excel('../inputs/BaciaCampos/1ESS95AES/1ESS95AES.xlsx',
                       skiprows=1,names = cab, usecols=(0,1,2,3,4,5,6,8,9,10,11), index_col = "DEPT")
df = pd.DataFrame(datareal)
df1 = df.drop_duplicates()
df2 = df1.dropna(how='all')
df3a = df2['RHO8a'].replace([np.nan], df2['RHO8a'].mean())
df3b = df2['RHOZa'].replace([np.nan], df2['RHOZa'].mean())
df3c = df2['RHO8b'].replace([np.nan], df2['RHO8b'].mean())
df3d = df2['RHOZb'].replace([np.nan], df2['RHOZb'].mean())
df3e = df2['ECGRa'].replace([np.nan], df2['ECGRa'].mean())
df3f = df2['ECGRb'].replace([np.nan], df2['ECGRb'].mean())
df3g = df2['HRDUa'].replace([np.nan], df2['HRDUa'].mean())
df3h = df2['HRDUb'].replace([np.nan], df2['HRDUb'].mean())
df3i = df2['NPHIa'].replace([np.nan], df2['NPHIa'].mean())

df3 = pd.concat([df3a, df3b, df3c, df3d, df3e, df3f, df3g, df3h, df3i], axis=1)
print(df3)


# In[4]:


# Entradas:
z = df3.index
RHOB = df3.RHO8a #(df3.RHO8a+df3.RHO8b)/2
RHOZ = df3.RHOZa#(df3.RHOZa+df3.RHOZb)/2
GR = df3.ECGRb#(df3.ECGRa+df3.ECGRb)/2
LLD = df3.HRDUa#(df3.HRDUa+df3.HRDUb)/2
NPHI = df3.NPHIa


# In[5]:


#replace dos nans
#RHOB = RHOB.replace([np.nan], RHOB.mean())
#RHOZ = RHOZ.replace([np.nan], RHOZ.mean())
#GR = GR.replace([np.nan], GR.mean())
#ILD = ILD.replace([np.nan], ILD.mean())
print(GR)


# In[6]:


# CAlculo do baseline para o dado real
GRbaseline = np.min(GR)
RHOBbaseline = np.min(RHOB)
LLDbaseline = np.min(LLD)

print(LLDbaseline)


# ## 1.2) Tabela (2022):
# 
# Dados utilizados para a definição do cálculo dos parâmetros ótimos da inversão via MC.

# In[7]:


# lendo os dados da tabela:
cab = ['DR', 'DT', 'GR', 'RHO', 'COT']
data = pd.read_csv('../inputs/dadoMC.txt', sep='\\s+',skiprows=1, names = cab, usecols=(0,1,2,3,4))
dfm = pd.DataFrame(data)
print(len(dfm))
#separando as informações do dado em variáveis distintas
DRm = dfm[dfm.columns[0]]
DTm = dfm[dfm.columns[1]]
GRm = dfm[dfm.columns[2]]
RHOm = dfm[dfm.columns[3]]
COTm = dfm[dfm.columns[4]]
#print(type(DR))
#df


# In[8]:


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

print(resmin,grmin, rhomin)


# In[9]:


dr = [DRm[0],DRm[1],DRm[5],DRm[6],DRm[10],DRm[11],DRm[15],DRm[16]]
dt = [DTm[0],DTm[1],DTm[5],DTm[6],DTm[10],DTm[11],DTm[15],DTm[16]]
gr = [GRm[0],GRm[1],GRm[5],GRm[6],GRm[10],GRm[11],GRm[15],GRm[16]]
rho = [RHOm[0],RHOm[1],RHOm[5],RHOm[6],RHOm[10],RHOm[11],RHOm[15],RHOm[16]]
cot = [COTm[0],COTm[1],COTm[5],COTm[6],COTm[10],COTm[11],COTm[15],COTm[16]]


# #### O baseline é geralmente determinado na formação que não é rica em MO e aonde nessa rocha tem as curvas de DT e GR se cruzando. 

# In[10]:


resbaseline = LLDbaseline # min(dr)/2   #np.sum(dr)/len(dr)
#dtbaseline  = DTbaseline # min(dt)/2    #np.sum(dt)/len(dt)
grbaseline  = GRbaseline # min(gr)/2    #np.sum(gr)/len(gr)
rhobbaseline = RHOBbaseline
print(resbaseline, grbaseline, rhobbaseline)


# # 2) Calculando o DrLog usando os valores tabelados

# In[11]:


#joga na funcao do deltalogR

dLOGr = dlogr(resmax,resbaseline,rhomax,rhobbaseline,2)

#Cálculo do dlogR via passey et al. (1990)

dLOGr90 = np.zeros(len(GR))

for i in range (len(dr)):
   dLOGr90[i] = dlogr90(dr[i],resbaseline,rho[i],rhobbaseline)
   print(dLOGr90[i])


# # 3) Monte Carlo com caixa adaptativa por porcentagem variada

# In[12]:


# Variáveis do modelo de monte carlo!
# definir primeiramente as caixas eta e delta!!!!!!!!
# PEGAR O VALOR DE TMAX DA TABELA DO ANDRE
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


# In[13]:


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
            cot_c[i,j] = passey16(dLOGr90[j],alfa[i],beta[i],delta[i],eta[i],p[4],gr[j],grbaseline)
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


# In[14]:


#Limites da caixa
print("Caixa eta->",etamin,etamax,eta_otimo)
print("Caixa delta->",deltamin,deltamax, delta_otimo)
print("Caixa beta->",betamin,betamax, beta_otimo)
print("Caixa alfa->",alfamin,alfamax, alfa_otimo)


# In[15]:


PHI_abs = np.sqrt(np.sum((cot_verdadeiro-cot_otimo)**2)/len(cot))
print(PHI_abs)


# In[16]:


#for i in range (len(cot)):
#    print(np.array(dLOGr90[i]),np.array(GR[i]-grbaseline),cot_otimo[i])


# # 4) Avaliação do método de monte carlo via cálculo do resíduo quadrático

# In[17]:


plt.plot(range(nbox),otimo[:,4], 'bo-',label='phi')# valores de eta por phi
plt.title('Convergência')
plt.xlabel('Iterações')
plt.ylabel('Phi')
plt.legend()
plt.savefig('../outputs/Convergencia.png')


# In[18]:


otimo[:,0]


# In[19]:


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
plt.savefig('../outputs/parametrosMC.png')


# In[20]:


#Salva o log file e figura
# data atual
minimo = alfa_otimo,beta_otimo,delta_otimo,eta_otimo,phi[imelhor]
local = datetime.now()
H = local.strftime("%d%m%Y%H%M%S")
# Salva o melhor valor em um arquivo de log
minimo = pd.DataFrame(minimo)
minimo.to_csv('../log/MC'+H+'.txt',sep=' ', index=False)
plt.savefig('../outputs/MC'+H+'.png')


# # 5) Calculando o COT para os dados da Bacia de Campos
# 

# ## 5.1) Poço 1ESS95AES

# In[21]:


deltaLR = dlogr90(LLD,LLDbaseline,RHOB,RHOBbaseline)


# In[22]:


cot_calc = passey16(deltaLR,alfa_otimo,beta_otimo,delta_otimo,eta_otimo,p[4],GR,GRbaseline)


# In[23]:


mincot=np.min(cot_calc)


# In[24]:


plt.figure(figsize=(20,3))
plt.plot(z,cot_calc-mincot,'b*', markersize=4,label='Monte Carlo')
plt.plot(z,np.zeros(len(z)),color='red')
plt.ylabel('COT')
plt.xlabel('Profundidade (m)')
plt.gca().invert_yaxis()   
plt.legend()
plt.savefig('../images/1ESS95AES.png')


# ## Cálculo da densidade aparente seca:

# In[25]:


DBD = DensidadeAparenteSeca(RHOB,NPHI)
print(DBD)


# # Arquivos de saída COT

# In[27]:


inputk = pd.DataFrame({'Depth(m)':z,'COT': cot_calc-mincot ,'RHOB':RHOB,'GR':GR, 'DBD':DBD})#,
                      #'NPHI':NPHI,'RHOB':RHOB})
inputk.to_csv('../outputs/1ESS95AES_COT.txt', sep=' ', index=False) 


# # FIM

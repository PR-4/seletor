import numpy as np

#-------------Content---------------#
# Total Organic Content calculations#
#-----------------------------------#

class COT:
    pass
def dlogr(res,x,m):
    '''Função que determina o Delta log R dos pares ordenados de propriedades
    Resistividade e Sônico ou Resistividade ou Densidade. 
    Entradas:
    res, dados de resistividade
    x, canal de densidade ou sônico
    m, coeficiente de cimentação
    Saída:
    DlogR, Delta Log R'''
    
    import math
    dado  = len(res)
    DlogR = np.zeros(dado)
    res   = np.array(res)
    x     = np.array(x)
    resb  = np.min(res)
    xb    = np.median(x)
    
    #Recurso computacional para eliminar os zeros:
    dummy = 1e-100000
    
    for i in range(dado):
        DlogR[i]=math.log10(res[i]/(resb+dummy))+((1/np.log(10))*(m/(x[i]-xb))*(x[i]-xb))
        if x[i]/xb < 0:
            print(x[i]-xb)
        if res[i]/resb < 0:
            print("Cuidado! Log negativo!",res[i]-resb)
     
        
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

def passey16(drlog,alfa,beta,delta,eta,Tmax,gr):
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
    dado = len(gr)
    COT  = np.zeros(dado)
    gr   = np.array(gr)
    grb  = np.median(gr) 
    
    for i in range(dado):
        COT[i] = (alfa*drlog[i] + beta*(gr[i]-grb))*10**(delta-eta*Tmax)
        #print(COT[i],delta-eta*Tmax)
            
    return COT

def empirico(pp,pf,zm,sr):
    '''
    Ressurgencia project (2023) empirical equation based on lake parameters that consider a preservation factor. Preservation factor takes into account geochemistry parameters such as O2, pH, carbon preservation factor in lake sediments. The sedimentation rate is a conditional that changes sedimentation's rate of position.   
    Input:
      -pp, primary productivity
      -sr, sedimentation rate
      -pf, preservation factor
      -zm, depth in meters

    Output:
      -COT: discrete total organic content
    '''
    if 1 <= sr <= 0:
        COT = (pp*pf*zm)/sr
    else:
        COT = pp*pf*zm*sr

    return COT
 

def igor(pp,pf,dbd,sr):
    '''
    Igor(2023) prediction TOC formulae.
    Inputs:
      -pp, primary productivity
      -sr, sedimentation rate
      -pf, preservation factor
      -dbd, dry bulk density
   Output:
      -COT: discrete total organic content
      '''
    COT = (pp * pf)/(dbd * sr)

    return COT

def muller(rho_sed):
    '''
    Müller (2005) definition of the empirical prediction of total organic content (TOC) by sediment density.
    
    Input:
     -rho_sed: grain sediment density array
    
    Output:
     - COT: or TOC total organic content array
    '''
    
    COT = (rho_sed - 2.65/0.0523)

    return COT

def dean(rho_sed):
    '''
    Dean and Gorham (1998) definition of the empirical prediction of total organic content (TOC) by sediment density.
    
    Input:
     -rho_sed: grain sediment density array
    
    Output:
     - COT: or TOC total organic content array
    '''

    COT = (rho_sed/1.665)**(1/-0.887)

    return COT

def stein(rho_sed,sr):
    '''
    Stein(1986) definition of the empirical prediction of total organic content (TOC) by sediment density.
    
    Input:
     -rho_sed: grain sediment density array
     -sr: sedimentation rate
     -pp: primary production
    Output:
     - COT: or TOC total organic content array
    '''
   

    COT = pp/(5*sr*rho_sed)

    return COT
#-------------Content--------------#
# Burial Efficiency calculations   #
#----------------------------------#


def alin(pp):
    '''
    Alin et al.(2007)  burial efficiency as a function of primary productivity.
    Input
     - pp: primary productivy array
    Output:
     - be: burial effiency array
    '''

    be = 204 * pp**(0.800)

    return be

def alin2(zm):
   '''
    Alin et al.(2007)  burial efficiency as a function of depth.
    Input
     - zm: depth array
    Output:
     - be: burial effiency array
    '''

   be = 0.0055 * zm + 0.621

   return be

def sobek(sr):
    '''
    Sobek et al.(2009) burial efficiency as a function of sedimentation rate for marine and non-marine environment.
    Input:
    - sr: sedimentation rate array
    Output:
    - be: burial efficiency array
    '''
    
    C = input('Is it a marine environment? (yes or no): ')
    
    if C == no:
        be = 31.1 + 27.9 * np.log(sr)
    if C == yes:
        be = ((sr * 1000)**(0.4))/2.1

    return be

def sobek2(oet,matterial):
    '''
    Sobek et al.(2009) defines burial efficiency as a function of oxygen time exposition and the distance of the source area. The source matterial can be allochthonous or autochthonous.
    Input:
     -oet: oxygen exposure time (years) array
     -matterial: list containing origin of organic matter. Can be autochthonous or allochthonous.
     Output:
     -be: burial efficiency array
    '''


    if matterial == []:
        print("WARNING! The list containing the organic matter origin is empty.")
    if matterial == allochthonous or aloctone:
        be = 61.2 - 16.7 * np.log(oet)
    if matterial == autochthonous or autoctone:
        be = 23.3 - 4.39 * np.log(oet)
    else:
        print("WARNING! You mispel the organic origin character.")
    
    return be

 
def alin3(cot,pp):
    '''
    Alin et al(2007) definition of burial efficiency by total organic content and primary productivity.
    Input:
     -cot: Total organic content array
     -pp: primary productivity array
    Output:
     -be: burial efficiency
    '''

    return be
#----------------------#
# Primary Productivity #
#----------------------#

def alin4(zm):
    '''
    Alin et al(2007) primary productivity calculations.
    Input:
     -zm: Depth array in meters
    Output:
     -pp: primary productivity
    '''

    pp = (2597 * zm)**-0.337

    return pp

def alin5(e):
    """
    Alin et al(2007) primary productivity calculation based on insolation parameter.
    Input:
     -e: insolation rate array in kWh m^-2 y^-1
    Output:
     -pp: primary productivity array
    """

    pp = 0.322 * e**(0.0015)

    return pp

#------------------------#
# Mass Accumulation Rate #
#------------------------#

def alin6(oc,phi,lsr,rho):
    """
    Alin et al(2007) mass accumulation rate caluculation based in oc parameter.
    Inputs:
     - oc: oc index array in %
     - phi: porosity array
     - lsr: linear sedimentation rate array
     - rho: sediment density array.
    Output:
     - mar: mass accumulation rate
    """

    marsed = (1-phi)*lsr*rho

    mar = marsed * (oc/100)

    return mar


#-------------#
# Carbon Flux #
#-------------#


def pace(pp,zm):
    '''
    Pace et al(1987) carbon flux calculation model for marine environments.
    Input:
     -pp: primary productivity
     -zm: depth array in meters
    Output:
     - cf: carbon flux array
    '''
   
    cf = 3.523 * zm**(-0.734) * pp

    return cf




    

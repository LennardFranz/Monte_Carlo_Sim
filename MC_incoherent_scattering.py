"""Programm zur Erhebung einer Stichprobe des des Richtingskosinuses des polaren
Streuwinkels bei inkohärenter Streuung von Photonen der Energien 300keV und 661.7keV
an Blei. Die entstehende Verteilung wurde mithilfe des Rejektionsmethode auf Basis
des Wirkungsquerschnitts nach KLein-Nishina multipliziert mit dem Quadrat des vorher
aus der Datei ISF.dat linear extrapolierten Funktion der inkohärenten Streufunktion 
erstellt. Die Verteilung und der entsprechend normierte theoretische Verlauf werden 
in einem Histogramm dargestellt.

"""

import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt



def Klein_Nishina_WQ(E_strich,m_e,mu,r_e):
    """Funktion zur Berechnung des differentiellen Wirkungsquerschnitts nach 
    Klein-Nishina."""
    kappa = E_strich/m_e
    alpha = 1/(1+kappa*(1-mu))
    return r_e**2/2 * (alpha**2) * (alpha+kappa*(1-mu)+mu**2)
    
def main():
    """Hauptfunktion"""
    df = pd.read_csv("ISF.dat", sep="\s+", usecols=[0,1], skiprows=2, header = None) 
    #Import der gegeben Daten 
    x_streu = df[0]
    streufkt = df[1]
    f = interp1d(x_streu, streufkt)
    #lineare Interpolation
    
    N = 100000
    E_strich1 = 300.0
    E_strich2 =661.7
    m_e = 511
    r_e = 2.817e-15
    #Festlegung der benötigten Parameter 
    
    mu_array = np.linspace(-1.0,1.0,N)
    #Erstellung eines Arrays von äuqidistanten mu Werten zur Normierung und plotten 
    
    mu_random = np.random.uniform(-1.0, 1.0, N)
    sigma_random = np.random.uniform(0.0, 1.0, N)  
    #Erstellung von von Arrays mit Zufallszahlen für Rejektionsmethode 

    norm1 = np.max(Klein_Nishina_WQ(E_strich1,m_e,mu_array,r_e)*f(E_strich1*np.sqrt((1 - mu_array)/2)))
    norm2 = np.max(Klein_Nishina_WQ(E_strich2,m_e,mu_array,r_e)*f(E_strich2*np.sqrt((1 - mu_array)/2)))
    #Parameter zur Normierung 
    
    hit1 = sigma_random < (1/norm1 * Klein_Nishina_WQ(E_strich1, m_e, mu_random, r_e) * f(E_strich1 * np.sqrt((1 - mu_random)/2)))
    hit2 = sigma_random < (1/norm2 * Klein_Nishina_WQ(E_strich2, m_e, mu_random, r_e) * f(E_strich2 * np.sqrt((1 - mu_random)/2)))
    #Erstellung von Arrays mit Inhalt True/False (Treffer/kein Treffer) nach 
    #Rejektionsmethode
   
    mu_array_hit1 = mu_random[hit1]
    mu_array_hit2 = mu_random[hit2]
    #Anwenden der hit-Arrays auf mu-Array mit Zufallszahlen um alle benötigten 
    #mus zu erhalten.

    fig= plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    #Erstellung Plotfenster 
    ax1.set_xlabel("$\mu$")
    ax2.set_xlabel("$\mu$")
    ax1.set_ylabel("Anzahl Hits")
    ax2.set_ylabel("Anzahl Hits")
    #Achsenbeschriftung
    ax1.hist(mu_array_hit1, bins="auto" )
    #Erstellung des Histoigramms
    ax1.plot(mu_array,3000*1/norm1* Klein_Nishina_WQ(E_strich1,m_e,mu_array,r_e)*f(E_strich1*np.sqrt((1 - mu_array)/2)))
    #Plot des differentiellen WQ nach K.-N. (Faktor 3000 wird zur ungefähren Skalierung verwendet)
    ax2.hist(mu_array_hit2, bins="auto" )
    ax2.plot(mu_array,3000*1/norm1* Klein_Nishina_WQ(E_strich2,m_e,mu_array,r_e)*f(E_strich2*np.sqrt((1 - mu_array)/2)))
    #analoges Vorgehen für zweite Energie
    
    fig.suptitle("Generator für inkohärente Streuung (Photonen)")
    ax1.title.set_text("300 keV")
    ax2.title.set_text("661,7 keV")
    
    plt.subplots_adjust(hspace = 0.4)
    plt.show()
    
    #Plot Überschriften
if __name__ == "__main__":
    main()

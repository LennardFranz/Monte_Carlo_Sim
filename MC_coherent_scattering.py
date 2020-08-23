"""Programm zur Erhebung einer Stichprobe des des Richtingskosinuses des polaren
Streuwinkels bei kohärenter Streuung von Photonen der Energien 300keV und 661.7keV
an Blei. Die entstehende Verteilung wurde mithilfe des Rejektionsmethode auf Basis
des Wirkungsquerschnitts nach J.J. Thompsen multipliziert mit dem Quadrat des vorher
aus der Datei AFF.dat linear extrapolierten Funktion des Atomformfaktor erstellt.
Die Verteilung un der entsprechend normierte theoretische Verlauf werden in einem 
Histogramm dargestellt.

Aufgrund der relativ hohen Energien sollte die Zoomfunktion innerhalb des
Plotfensters verwendet werden um den Bereich der X-Achse einzuschränken. 
"""


import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt



def Thompson_Streuformel(r_e,mu):
    """Funktion zur Berechnung des WQ nach JJ. Thompson"""
    return (r_e**2) * (1+mu**2)/2
    
def main():
    """Hauptfunktion"""
    df = pd.read_csv("AFF.dat", sep="\s+", usecols=[0,1], skiprows=2, header = None) 
    #Import der gegeben Daten 
    x_streu = df[0]
    streufkt = df[1]
    f = interp1d(x_streu, streufkt)
    #lineare Interpolation
    
    N = 10000
    E_strich1 = 300.0
    E_strich2 = 661.7
    m_e = 511
    r_e = 2.817e-15
    #Festlegung der benötigten Parameter 
    
    mu_array = np.linspace(-1.0,1.0,N)
    #Erstellung eines Arrays von äuqidistanten mu Werten zur Normierung und plotten 
    
    mu_random = np.random.uniform(-1.0, 1.0, N)
    sigma_random = np.random.uniform(0.0, 1.0, N)  
    #Erstellung von von Arrays mit Zufallszahlen für Rejektionsmethode 

    norm1 = np.max(Thompson_Streuformel(r_e,mu_array)*(f(E_strich1*np.sqrt((1 - mu_array)/2)))**2)
    norm2 = np.max(Thompson_Streuformel(r_e,mu_array)*(f(E_strich2*np.sqrt((1 - mu_array)/2)))**2)
    #Parameter zur Normierung 
    #print(1/norm1 * Thompson_Streuformel(r_e,mu_array) * (f(E_strich1 * np.sqrt((1 - mu_random)/2)))**2)
    hit1 = sigma_random < (1/norm1 * Thompson_Streuformel(r_e,mu_array) * (f(E_strich1 * np.sqrt((1 - mu_random)/2)))**2)
    hit2 = sigma_random < (1/norm2 * Thompson_Streuformel(r_e,mu_array) * (f(E_strich2 * np.sqrt((1 - mu_random)/2)))**2)
    #Erstellung von Arrays mit Inhalt True/False (Treffer/kein Treffer) nach 
    #Rejektionsmethode
    #print(1/norm1 * Thompson_Streuformel(r_e,mu_array) * (f(E_strich1 * np.sqrt((1 - mu_random)/2)))**2)
    mu_array_hit1 = mu_random[hit1]
    mu_array_hit2 = mu_random[hit2]
    #Anwenden der hit-Arrays auf mu-Array mit Zufallszahlen um alle benötigten 
    #mus zu erhalten.

    
    fig = plt.figure(figsize=(10,8))
    
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    #Erstellung Plotfenster 
    ax1.set_xlabel("$\mu$")
    ax2.set_xlabel("$\mu$")
    ax1.set_ylabel("Anzahl Hits")
    ax2.set_ylabel("Anzahl Hits")
    #Achsenbeschriftung
    ax1.set_xlim(-1.0,1.0)
    ax2.set_xlim(-1.0,1.0)
    
    ax1.hist(mu_array_hit1, bins=100 )
    #Erstellung des Histoigramms
    ax1.plot(mu_array,55*1/norm1* Thompson_Streuformel(r_e,mu_array)*(f(E_strich1*np.sqrt((1 - mu_array)/2)))**2,lw=2, ls="dotted")
    #Plot des differentiellen WQ nach K.-N. (Faktor 55 wird zur ungefähren Skalierung verwendet)
    ax2.hist(mu_array_hit2, bins=20 )
    ax2.plot(mu_array,12*1/norm1* Thompson_Streuformel(r_e,mu_array)*(f(E_strich2*np.sqrt((1 - mu_array)/2)))**2, lw=2, ls="dotted")
    #analoges Vorgehen für zweite Energie
    fig.suptitle("Generator für kohärente Streuung (Photonen)")
    ax1.title.set_text("300 keV")
    ax2.title.set_text("661,7 keV")
    #Plot Überschriften
    plt.subplots_adjust(hspace = 0.4)
    plt.show()
   
    
if __name__ == "__main__":
    main()

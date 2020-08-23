"""Programm zur Aproximation von Pi mittels Monte-Carlo Methoden

Es werden Approximationen der Kreiszahl Pi mithilfe der klassischen Monte_Carlo
Methode und der Neumannschen Rejektionsmethode durchgeführt.
Außerdem wird der relative Fehler für verschiedene Stichprobenumfänge berechnet
und dargestellt.
"""
from random import random 
import numpy as np 
import matplotlib.pyplot as plt

def func(x):
    """Funktion innerhalb des zu lösenden Integrals"""
    return np.sqrt(1-(x**2))

def klassisch_MC (anzahl,funktion):
    """Funktion zur Aproximation von Pi durch Lösen des gegeben Integrals.
    (Funktion im Intervall 0 bis 1) 
    Der benötigte Paramter anzahl legt die Anzahl an verwendeten Zufallszahlen fest
    Zurückgegeben wird ein Array der Länge anzahl mit Aproximationen von Pi.
    Dabei entspricht der Index innerhalb des Array der jeweiligen Anzahl an
    verwendeten Zufallszahlen."""
    pi = np.zeros(anzahl)
    for i in range(1,anzahl):
        q = np.random.uniform(0.0,1.0,i)
        a = funktion(q)
        pi[i] = 4*sum(a)/i
    return(pi)


def Neumann_rejek(anzahl):
    """Funktion zur Approximation von Pi durch die Neumannsche Rejektionsmethode.
    Der benötigte Paramter anzahl legt die Anzahl an verwendeten Zufallszahlen fest
    Zurückgegeben wird ein Array der Länge anzahl mit Aproximationen von Pi.
    DAbei entspricht der Index innerhalb des Array der jeweiligen Anzahl an 
    verwendeten Zufallszahlen.
    """
    pi = np.zeros(anzahl)
    for i in range (1,anzahl):
        x = np.random.uniform(0.0,1.0,i)
        y = np.random.uniform(0.0, 1.0,i)
        pi[i] = 4*sum(x*x + y*y < np.ones(i))/i
    return pi
          

def relativer_Fehler(num,real):
    """Funktion zur Berechnung des relativen Fehlers"""
    return abs((real-num)/real)

def main():
    N = 1000  #Stichprobenumfang
    
    fig= plt.figure(figsize=(10,8))
    ax1 = fig.add_subplot(211, xscale="log", yscale="log")
    ax2 = fig.add_subplot(212, xscale="log", yscale="log")
    #Erstellung des Plotfensters 

    ax1.plot(relativer_Fehler(Neumann_rejek(N), np.pi), linestyle="None", marker=".")
    ax2.plot(relativer_Fehler(klassisch_MC(N, func), np.pi), linestyle="None", marker=".")
    #Plotten der Arrays in doppeltlogarithmischer Darstellung 
    ax1.grid(True)
    ax2.grid(True)
    #Hinzufügen des Gitters

    ax1.set_xlabel("Stichprobenumfang")
    ax2.set_xlabel("Stichprobenumfang")
    ax1.set_ylabel("relativer Fehler")
    ax2.set_ylabel("relativer Fehler")
    #Achsenbeschriftung
    fig.suptitle("Relativer Fehler der Approximation von Pi (Monte-Carlo Methoden)")
    ax1.title.set_text("klassische Monte-Carlo Methode ")
    ax2.title.set_text("Neumannsche Rejektionsmethode")
    #Plotüberschriften

    plt.subplots_adjust(hspace = 0.4)
    #Anpassen der Subplots um Überschneidungen zu vermeiden

    plt.show()

if __name__ == "__main__":
    main()
    
    
    
"""
Formel für Mittelwert: X = 1/N * sum(f(x_i))

Fehler des Mittelwerts (Varianz): sigma^2 = (1/N * sum(f(x_i)^2)) - (1/N sum(f(x_i)))^2
Somit ist der Fehler proportional zu 1/N^(-1/2)

Die effizientere Methode stellt die klassische Monte-Carlo Methode, welche die
Lösung des gegeben Integrals verwedet, da die Varianz dieser Funktion kleiner ist.

Der gewählte Stichprobenumfang N von 1000 kann bei höherer Rechenleistung bzw. 
längerer Rechenzeit natürlich vergrößert werden.
 

"""

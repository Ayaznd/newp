import numpy as np


def matdis(don):
    n=don.shape[0]
    matrice = np.empty((n,n))
    for i in range(n):
      for j in range(n):
            premiere_ligne = don[i]
            deuxieme_ligne = don[j]
            dis = dist(premiere_ligne, deuxieme_ligne)
            matrice[i][j] = dis
    return matrice
def dist(i1, i2):
    a = len(i1)
    d = 0
    for i in range(a):
        d = d + (i1[i] - i2[i]) ** 2
    d = d ** 0.5
    return d
import math
def matdis(don):
    n=don.shape[0]
    matrice = np.empty((n,n))
    for i in range(n):
      for j in range(n):
            premiere_ligne = don[i]
            deuxieme_ligne = don[j]
            dis = dist(premiere_ligne, deuxieme_ligne)
            matrice[i][j] = dis
    return matrice

def disward(don, cluster):
    n = don.shape[0]
    matrice = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            a = math.sqrt((len(cluster[i])*len(cluster[j])) / (len(cluster[i]) + len(cluster[j])))
            matrice[i][j] = a * dist(don[i],don[j])
            #matrice[i][j] = a * np.linalg.norm(don[i] - don[j]) ** 2
    return matrice

def diswardk(help, cluster,don):
    m=help.shape[0]
    n=don.shape[0]
    matrice = np.empty((m,n))
    for i in range(m):
        for j in range(n):
            b=trouver_cluster(j,cluster)
            a = math.sqrt((len(cluster[i])*len(cluster[b]) / (len(cluster[i])+len(cluster[b]))))
            matrice[i][j] = a*dist(help[i],don[j])
    return matrice
def diskeuc(help,don):
    m=help.shape[0]
    n=don.shape[0]
    matrice = np.empty((m,n))
    for i in range(m):
        for j in range(n):
            matrice[i][j] = dist(help[i],don[j])
    return matrice

def trouver_cluster(point, clusters):
    for i, cluster in enumerate(clusters):
        if point in cluster:
            return i
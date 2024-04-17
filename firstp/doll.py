from firstp.dist import matdis,disward
import matplotlib.pyplot as plt

def func(don):
    L = []
    resultats_html = "<ul>"  # Commence une liste non ordonnée HTML
    cluster = [[i] for i in range(len(don))]
    b=don.shape[0]
    for k in range(b- 1):

        matrice = disward(don,cluster)
        valeurs = [valeur for ligne in matrice for valeur in ligne]
        valeurs_uniques = sorted(set(valeurs))
        deuxieme_valeur_minimale = valeurs_uniques[1]
        indices = [(i, j) for i, ligne in enumerate(matrice) for j, valeur in enumerate(ligne) if
                   valeur == deuxieme_valeur_minimale]
        L.append(valeurs_uniques[1])
        cluster[indices[0][0]].extend(cluster[indices[0][1]])
        del cluster[indices[0][1]]
        print(cluster)
        don = NewMatrix(indices[0][0], indices[0][1], don)
        # Ajoutez chaque résultat à la liste HTML
        resultats_html += f"<li>La distance minimale dans la matrice est : {deuxieme_valeur_minimale}</li>"
        resultats_html += f"<li>Donc les individus a combiner sont : {cluster[indices[0][0]]}</li>"
        resultats_html += f"<li>et les clusters deviennent : {cluster}</li>"
        #resultats_html += f"<li>Liste L mise à jour : {L}</li>"
    resultats_html += "</ul>"  # Ferme la liste HTML
    return resultats_html
def function(don) :
   L=[]
   cluster=[[i] for i in range(len(don))]
   for k in range(don.shape[0]-1):
     matrice =disward(don,cluster)
     valeurs = [valeur for ligne in matrice for valeur in ligne]
     valeurs_uniques = sorted(set(valeurs))
     deuxieme_valeur_minimale = valeurs_uniques[1]
     print("La valeur minimale dans la matrice est :", deuxieme_valeur_minimale)
     indices = [(i, j) for i, ligne in enumerate(matrice) for j, valeur in enumerate(ligne) if valeur == deuxieme_valeur_minimale]
     print("Indices de la valeur minimale dans la matrice :", indices)
     L.append(valeurs_uniques[1])
     cluster[indices[0][0]].extend(cluster[indices[0][1]])
     del cluster[indices[0][1]]
     print(cluster)
     don=NewMatrix(indices[0][0],indices[0][1],don)

def plot_dendrogram(data, clusters):
         """Affiche le dendrogramme."""
         plt.figure(figsize=(10, 5))
         for i, cluster in enumerate(clusters):
             x = [data[idx][0] for idx in cluster]
             y = [data[idx][1] for idx in cluster]
             plt.scatter(x, y, label=f'Cluster {i + 1}')
         plt.title('Dendrogramme')
         plt.xlabel('Index des points')
         plt.ylabel('Distance')
         plt.legend()
         plt.show()





import numpy as np
from numpy import ndarray, dtype


def distance(i1, i2):
    a = len(i1)
    d = 0
    for i in range(a):
        d = d + (i1.iloc[i] - i2.iloc[i]) ** 2
    d = d ** 0.5
    return d
def NewMatrix(i, j, mat):
    newl = (mat[i] + mat[j])/2
    mat[i] = newl
    mat = np.delete(mat, [j], axis=0)
    return mat


import base64
import numpy as np
import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from streamlit_option_menu import option_menu
from firstp import NewMatrix, disward
from firstp.dist import diswardk, matdis, diskeuc

def plot_clusters(data, labels, centroids, names):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5, label='Data points')
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='red', label='Centroids')
    ax.set_title('Clusters')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    for i, txt in enumerate(names):
        ax.text(data[i, 0], data[i, 1], txt, fontsize=8, ha='right')
    st.pyplot(fig)
def kmeans_pp_init(don, k):
    centroids = []
    centroid_indices = []  # Liste pour stocker les indices des centroïdes sélectionnés

    # Choisir le premier centroid aléatoirement parmi les données
    first_centroid_index = np.random.choice(len(don))
    centroids.append(don[first_centroid_index])
    centroid_indices.append(first_centroid_index)

    for _ in range(1, k):
        # Calculer les distances au carré de chaque point de données aux centroids existants
        distances = np.array([min(np.linalg.norm(d - c) ** 2 for c in centroids) for d in don])
        # Choisir le prochain centroid avec une probabilité proportionnelle à sa distance au carré au centroid le plus proche
        new_centroid_index = np.random.choice(len(don), p=distances / distances.sum())
        centroids.append(don[new_centroid_index])
        centroid_indices.append(new_centroid_index)

    return np.array(centroids), np.array(centroid_indices)
def algokm(don,k,names) :
    clusters = [[i] for i in range(don.shape[0])]
    help,va=kmeans_pp_init(don,k)
    mat = diswardk(help,clusters, don)
    plot_clusters(don, clusters, help,names)
    output = ""  # Chaîne de caractères pour stocker les résultats

    # Ajoutez les résultats à la chaîne de caractères
    output += f"<li>Indices choisis par kmeans_pp_init: {va}\n</li>"
    output += f"<li>Matrice de distance initiale :\n{mat}\n</li>"

    # Ajoutez les autres résultats à la chaîne de caractères

    list = []

    for j in range(don.shape[0]):
        if mat[0][j] != 0:
            min = mat[0][j]
        else:
            min = mat[2][j]
        for i in range(int(k)):
            if mat[i][j] < min and mat[i][j] != 0:
                min = mat[i][j]
        list.append(min)
    output += f"<li>La liste des distances minimales : {list}\n</li>"

    indices = []
    for j in range(len(mat[0])):
        for i in range(len(mat)):
            if mat[i][j] == list[j]:
                indices.append((i, j))
                break

    output += f'<li>Les indices des distances minimales sont : {indices}\n</li>'

    for i in range(len(indices)):
        if indices[i][1] not in va:
            clusters[indices[i][0]].append(indices[i][1])

    for i in range(k):
        clusters[i][0] = va[i]
    for l in range(len(clusters) - k):
        del clusters[len(clusters) - 1]

    output += f'<li>Clusters après la première itération : {clusters}\n</li>'
    cluster_names = [[names[i] for i in cluster] for cluster in clusters]
    output += f'<li>Alors les clusters sont : {cluster_names}\n</li>'
    help = np.empty((k, don.shape[1]))
    for l in range(k):
        cluster_indices = clusters[l]
        cluster_data = don[cluster_indices]
        cluster_mean = np.mean(cluster_data, axis=0)
        help[l] = cluster_mean

    output += f'<li>Centroides après la première itération : {help}\n</li>'

    mat = diswardk(help,clusters, don)

    list = []
    for j in range(don.shape[0]):
        if mat[0][j] != 0:
            min = mat[0][j]
        else:
            min = mat[2][j]
        for i in range(int(k)):
            if mat[i][j] < min and mat[i][j] != 0:
                min = mat[i][j]
        list.append(min)
    output += f'<li>Liste des distances minimales : {list}\n</li>'

    indices = []
    for j in range(len(mat[0])):
        for i in range(len(mat)):
            if mat[i][j] == list[j]:
                indices.append((i, j))
                break

    output += f'<li>Indices des distances minimales : {indices}\n</li>'

    for i in range(len(indices)):
        for j in range(k):
            if indices[i][1] in clusters[j] and len(clusters[j])!=1 :
                clusters[j].remove(indices[i][1])
                clusters[indices[i][0]].append(indices[i][1])
                break
    cluster_names = [[names[i] for i in cluster] for cluster in clusters]
    output += f'<li> Clusters après une itération : {clusters}\n</li>'
    output += f'<li>Alors les clusters sont : {cluster_names}\n</li>'
    newclus = None
    c = 1
    while newclus != clusters:

        newclus = [sublist[:] for sublist in clusters]

        help = np.empty((k, don.shape[1]))
        for l in range(k):
            cluster_indices = clusters[l]
            cluster_data = don[cluster_indices]
            cluster_mean = np.mean(cluster_data, axis=0)
            help[l] = cluster_mean

        mat = diswardk(help,clusters, don)

        list = []
        for j in range(don.shape[0]):
            if mat[0][j] != 0:
                min = mat[0][j]
            else:
                min = mat[2][j]
            for i in range(int(k)):
                if mat[i][j] < min and mat[i][j] != 0:
                    min = mat[i][j]
            list.append(min)

        indices = []
        for j in range(len(mat[0])):
            for i in range(len(mat)):
                if mat[i][j] == list[j]:
                    indices.append((i, j))
                    break

        for i in range(len(indices)):
            for j in range(k):
                if indices[i][1] in clusters[j] and len(clusters[j])!=1:
                    clusters[j].remove(indices[i][1])
                    clusters[indices[i][0]].append(indices[i][1])
                    break
        c = c + 1
        output += f'<li>Clusters après {c} itération: {clusters}\n</li>'
        cluster_names = [[names[i] for i in cluster] for cluster in clusters]
        output += f'<li>Alors après {c} itération les clusters sont: {cluster_names}\n</li>'

    return output

def funcsw(don,names):
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
        resultats_html += f"<li>La distance minimale dans la matrice de distance est : {deuxieme_valeur_minimale}</li>"
        resultats_html += f"<li>Donc les individus a combiner sont : {cluster[indices[0][0]]}</li>"
        cluster_names = [names[i] for i in cluster[indices[0][0]]]
        resultats_html += f"<li>on combine alors : {cluster_names}</li>"
        resultats_html += f"<li>et les clusters deviennent : {cluster}</li>"
        cluster_names = [[names[i] for i in clusters] for clusters in cluster]
        resultats_html+= f'<li>c-a-d les clusters deviennent : {cluster_names}\n</li>'
        #resultats_html += f"<li>Liste L mise à jour : {L}</li>"
    resultats_html += "</ul>"  # Ferme la liste HTML
    return resultats_html
def funcse(don,names):
    L = []
    resultats_html = "<ul>"  # Commence une liste non ordonnée HTML
    cluster = [[i] for i in range(len(don))]
    b=don.shape[0]
    for k in range(b- 1):

        matrice = matdis(don)
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
        cluster_names = [names[i] for i in cluster[indices[0][0]]]
        resultats_html += f"<li>on combine alors : {cluster_names}</li>"
        resultats_html += f"<li>et les clusters deviennent : {cluster}</li>"
        #resultats_html += f"<li>Liste L mise à jour : {L}</li>"
    resultats_html += "</ul>"  # Ferme la liste HTML
    return resultats_html

def get_base64(png_file):
    with open(png_file, 'rb') as f:
        data = f.read()
        base64_data = base64.b64encode(data).decode('utf-8')
    return base64_data

def set_background(png_file,opacity=1):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        opacity: {opacity};
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)




def main():
    set_background('C:/Users/user/PycharmProjects/pythonProject.py/firstp/data.png')
    def classificationw():
        st.title("Classification Hiérarchique")

        st.write(
            "Bonjour ! Voici une application qui effectue la classification hiérarchique sur votre jeu de données.")

        # Ajouter un composant pour télécharger un fichier Excel
        uploaded_file = st.file_uploader("Téléchargez un fichier Excel", type=["xlsx", "xls"])

        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            don = df[df.columns[1:df.shape[1]]]
            # Data frame devient une matrice numpy
            don = don.values
            names = df[df.columns[0]].tolist()
            # Affichage du contenu en fonction de l'onglet sélectionné
            st.write("Voici votre jeux de données :")
            st.write(df)
            linkage_matrix = hierarchy.linkage(don, method='ward')
        # Créer la figure et le dendrogramme
            fig, ax = plt.subplots(figsize=(7, 3))
            hierarchy.dendrogram(linkage_matrix, ax=ax,labels=names)
            plt.title('Dendrogramme')
            plt.xlabel('Index des points')
            plt.ylabel('Distance')
            st.pyplot(fig)
            with st.expander("RÉSULTAT DE CLASSIFICATION HIÉRARCHIQUE "):
                result = funcsw(don,names)
            st.markdown(f"""
                                            <p style="font-size: 20px; color: green;">
                                                Résultat de la classification hiérarchique
                                            </p>
                                            <p style="arial: 17px;">
                                                <span style="color: black;">{result}</span>
                                            </p>
                                            """, unsafe_allow_html=True)

    def classificatione():
        st.title("Classification Hiérarchique")

        st.write(
            "Bonjour ! Voici une application qui effectue la classification hiérarchique sur votre jeu de données.")

        # Ajouter un composant pour télécharger un fichier Excel
        uploaded_file = st.file_uploader("Téléchargez un fichier Excel", type=["xlsx", "xls"])

        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            don = df[df.columns[1:df.shape[1]]]
            # Data frame devient une matrice numpy
            don = don.values
            # Création des onglets avec st.sidebar
            names=df[df.columns[0]].tolist()
            #concatenated_names = " ".join(names)
            #st.write(concatenated_names)
            # Affichage du contenu en fonction de l'onglet sélectionné
            st.write("Voici votre jeux de données :")
            st.write(df)
            linkage_matrix = hierarchy.linkage(don, method='single')
        # Créer la figure et le dendrogramme
            fig, ax = plt.subplots(figsize=(7, 3))
            hierarchy.dendrogram(linkage_matrix, ax=ax,labels=names)
            plt.title('Dendrogramme')
            plt.xlabel('Index des points')
            plt.ylabel('Distance')
            st.pyplot(fig)
            with st.expander("RÉSULTAT DE CLASSIFICATION HIÉRARCHIQUE "):
                result = funcsw(don,names)
            st.markdown(f"""
                                            <p style="font-size: 20px; color: green;">
                                                Résultat de K-Means
                                            </p>
                                            <p style="arial: 17px;">
                                                <span style="color: black;">{result}</span>
                                            </p>
                                            """, unsafe_allow_html=True)
    # Liste des options d'onglets
    selected=option_menu(
        menu_title=None,
        options=["Home","Classification Hiérarchique","K-means"],
        icons=["house","math","finance"],
        orientation="horizontal",
        styles={
            "icon": {"color": "#F4AEB9", "font-size": "25px"},
            "nav-link": {"font-size": "20px", "text-align": "left", "margin": "0px", "color": "black",
                         "padding": "5px 10px","white-space": "nowrap"},
            "nav-link-selected": {"background-color": "#75AAEE"},
        }
    )
    if selected=="Home":
        st.title("K-Means et la Classification Hiérarchique sont deux algorithmes populaires de classification non supervisée. ")
        st.write(
            '<span style="font-size:20px">Ils permettent de regrouper des données en clusters sans avoir à les étiqueter manuellement.</span>',
            unsafe_allow_html=True)
        st.write(
            '<span style="font-size:20px">L\'utilisation de K-Means et de la Classification Hiérarchique permet de découvrir des structures et des patterns cachés dans les données.</span>',
            unsafe_allow_html=True)
        st.write(
            '<span style="font-size:20px">Ces algorithmes peuvent être utilisés pour la segmentation des clients, l\'analyse de marché, la recommandation de produits, etc.</span>',
            unsafe_allow_html=True)
        st.write(
            '<span style="font-size:20px">La classification des données permet de mieux comprendre les clients, d\'améliorer l\'expérience client et de prendre de meilleures décisions.</span>',
            unsafe_allow_html=True)
    if selected=="Classification Hiérarchique":
        selec= option_menu(
            menu_title=None,
            options=["Distance euclidienne", "Distance de Ward (recommandée)"],
            icons=["home", "finance"],
            #orientation="horizontal",
            styles={
                "icon": {"color": "#F4AEB9", "font-size": "25px"},
                "nav-link": {"font-size": "18px", "text-align": "left", "margin": "0px", "color": "black",
                             "padding": "5px 10px", "white-space": "nowrap"},
                "nav-link-selected": {"background-color": "#75AAEE"},

            }
        )
        if selec=="Distance euclidienne":
          classificatione()
        if selec=="Distance de Ward (recommandée)":
          classificationw()
    if selected=="K-means":
        st.title("K-means ")

        st.write(
            "Bonjour ! Voici une application qui technique K-means sur votre jeu de données.")
        number = st.number_input("Entrez le nombre de clusters souhaité :", min_value=0, max_value=20, step=1,
                                 value=3)
        # Ajouter un composant pour télécharger un fichier Excel
        uploaded_file = st.file_uploader("Téléchargez un fichier Excel", type=["xlsx", "xls"])

        # Afficher le nombre saisi par l'utilisateur
        st.write("Vous avez saisi :", number)
        if uploaded_file is not None and number is not None:
            df = pd.read_excel(uploaded_file)
            don = df[df.columns[1:df.shape[1]]]
            # Data frame devient une matrice numpy
            don = don.values
            names = df[df.columns[0]].tolist()

            # Affichage du contenu en fonction de l'onglet sélectionné
            st.write("Voici votre jeux de données :")
            st.write(df)
            with st.expander("DIAGRAMME DE DISPERSION (K-Means)"):
                result =algokm(don,number,names)
            st.markdown(f"""
                                            <p style="font-size: 20px; color: green;">
                                                Résultat de K-Means
                                            </p>
                                            <p style="arial: 17px;">
                                                <span style="color: black;">{result}</span>
                                            </p>
                                            """, unsafe_allow_html=True)




        #st.write(funcs(don))


if __name__ == "__main__":
    main()
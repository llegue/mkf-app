# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:31:15 2022                                                  

\_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/
/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \
\_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/
/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \
\_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/
/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \
\_/ \_/ \_/ \_████████████████████████████████████/ \_/ \_/ \_/ \_/ \_/ \_/ \_/
/ \_/ \_/ \_/ ██      ██████▓▓                  ██\_/ \_/ \_/ \_/ \_/ \_/ \_/ \
\_/ \_/ \_/ \_██      ████████████████████████████/ \_/ \_/ \_/ \_/ \_/ \_/ \_/   
/ \_/ \_/ \_/ ████████████████████████████████████\_/ \_/ \_/ \_/ \_/ \_/ \_/ \
\_/ \_/ \_/ \_██████████████        ██████████████/ \_/ \_/ \_/ \_/ \_/ \_/ \_/  
/ \_/ \_/ \_/ ████████████            ████████████\_/ \_/ \_/ \_/ \_/ \_/ \_/ \
\_/ \_/ \_/ \_██████████    ████████    ██████████/ \_/ \_/ \_/ \_/ \_/ \_/ \_/
/ \_/ \_/ \_/ ██████████  ██  ████████  ██████████\_/ \_/ \_/ \_/ \_/ \_/ \_/ \
\_/ \_/ \_/ \_██████████  ████████████  ██████████/ \_/ \_/ \_/ \_/ \_/ \_/ \_/  
/ \_/ \_/ \_/ ██████████    ████████    ██████████\_/ \_/ \_/ \_/ \_/ \_/ \_/ \
\_/ \_/ \_/ \_████████████  ██████    ████████████/ \_/ \_/ \_/ \_/ \_/ \_/ \_/  
/ \_/ \_/ \_/ ██████████████        ██████████████\_/ \_/ \_/ \_/ \_/ \_/ \_/ \
\_/ \_/ \_/ \_████████████████████████████████████/ \_/ \_/ \_/ \_/ \_/ \_/ \_/  
/ \_/ \_/ \_/ ██                                ██\_/ \_/ \_/ \_/ \_/ \_/ \_/ \
\_/ \_/ \_/ \_██                                ██/ \_/ \_/ \_/ \_/ \_/ \_/ \_/   
/ \_/ \_/ \_/ ████████████████████████████████████\_/ \_/ \_/ \_/ \_/ \_/ \_/ \
\_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/
/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \
\_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/
/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \
\_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/



Script contenant des class et des fonctions pour la correlation d'images.

La class TargetTracking permet le suivi de cible, tandis que la classe TemplateMatching
permet la detection et le suivi de motifs.

Ces scripts ont ete realises afin de pouvoir suivre le deplacement des noeuds
d'un filet lors d'essais de traction filmes.

@author: llegue
"""

# import des modules utiles
import os
import time
from datetime import datetime
import numpy as np
from skimage.io import imread, imshow, imsave
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage.feature import match_template
from skimage.feature import peak_local_max
from skimage.transform import rotate
from matplotlib.widgets import RectangleSelector
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib as mpl
import tkinter
mpl.use("TkAgg")

def make_gif(chemin_dossier_images, chemin_dossier_sauvegarde, length = 200):
    """
    Fonction qui permet de creer un gif a partir d'un dossier contenant des images

    Parameters
    ----------
    chemin_dossier_images : str
        Chemin vers le dossier contenant les frames du GIF à créer.
    chemin_dossier_sauvegarde : str
        Chemin vers le fichier de sauvegarde du gif. Il doit se terminer par .gif
    length : int, optional
        Longueur du gif a creer en secondes. TLa valeur par defaut est de 200.

    Returns
    -------
    None.

    """
    
    # recuperation de la liste d'images
    liste_images = [x for x in os.listdir(chemin_dossier_images) if x!='.DS_Store']
    
    # tri des images dans le bon ordre 
    # d'abord l'int correspondant a chaque image est recupere du nom de l'image, 
    # puis la fonction zip est utilisee pour trier le nom des images en fonction 
    # des ints correspondants 
    # (la variable _ est utilise car la fonction zip renvoit deux iterateurs et seul x nous interesse)
    Y = [int(x.split('.')[0]) for x in liste_images]
    liste_images_sorted = [x for _,x in sorted(zip(Y,liste_images))]
    
    # stockage des images dans un iterateur
    imgs = (Image.open(os.path.join(chemin_dossier_images,f)) for f in liste_images_sorted)
    img = next(imgs)
    
    # sauvegarde du gif vers le chemin demande
    img.save(fp=chemin_dossier_sauvegarde, 
             format='GIF', append_images=imgs, save_all=True, duration = length, loop=0)

def read_data_ni(chemin_datas_temps_ni):
    """
    Fonction qui pprmet de lire le fichier de sortie du logiciel d'acquisition
    "AcqBasler"
    Parameters
    ----------
    chemin_datas_temps_ni : str
        chemin vers le fichier de sortie du logiciel "AcqBasler".

    Returns
    -------
    temps : list 
        liste de floats des instants enregistres.
    efforts : list
        liste de floats des efforts enregistres.

    """
    
    # ouverture du fichier et recuperation des lignes
    with open(chemin_datas_temps_ni,'r') as f:
        all_lignes = f.readlines()[1:]
    
    # stockage de la date de debut de l'essai
    date_debut_essai = datetime.strptime(all_lignes[0].strip().split('\t')[1], "%d/%m/%Y %H:%M:%S")
    
    # initialisation des liste pour les sorties
    efforts = []
    temps = []
    
    # parcours des lignes et recuperation des donnees
    for i,a in enumerate(all_lignes):  
        efforts.append(float(a.strip().split('\t')[-1]))
        
        temps_i = datetime.strptime(all_lignes[i].strip().split('\t')[1], "%d/%m/%Y %H:%M:%S")
        dt = temps_i - date_debut_essai
        
        temps.append(dt.total_seconds())
    
    return temps, efforts


class TargetTracking:
    """
    Class pour le suivi de cibles dans une liste d'images.
    
    Etapes pour utiliser cette class : 
        - definir un chemin vers un dossier avec les images de l'essai,
        - definir le nombre de cible à suivre dans les images,
        - initialiser un objet appartenant a la class avec le chemin et le nombre
          de cibles,
        - utiliser la methode target_tracking enrenseignant les differentes options.
    
    Le scripts doit obligatoirement etre lance dans une fenetre de terminal pour
    que la selection des cibles fonctionne.
    
    """
    def __init__(self, dossier_images, n_target, rotation = False):
        """
        Initialisation de la class a partir du dossier contenant les images et
        le nombre de cibles.
        
        L'initialisation d'un objet de cette class lance la selection des cibles.
        
        Parameters
        ----------
        chemin_input : str
            dossier contenant les images de l'essai. Les images doivent etre numerotees.
        n_target : int
            le nombre de cibles a suivre a travers la liste d'images.

        Returns
        -------
        None.

        """
        
        # recuperation du chemin vers les images
        self.chemin_input = dossier_images
    
        # recuperation de la liste des images
        # de ce dossier pour obtenir l'etat a t = 0
        liste_images = [x for x in os.listdir(dossier_images) if x!='.DS_Store']
        
        # tri des images dans l'ordre de numerotation croissant
        Y = [ int(x.split('.')[0]) for x in liste_images]
        liste_images_sorted = [x for _,x in sorted(zip(Y,liste_images))]
        
        self.chemin_image_t0 = os.path.join(dossier_images,liste_images_sorted[0])
        image_t0 = imread(self.chemin_image_t0, as_gray=True)
        
        self.rotation = rotation
        
        if self.rotation:
            self.image_t0 = rotate(image_t0,270,resize=True)
        else:
            self.image_t0 = image_t0
        
        # stockage du nombre de cibles
        self.n_target = n_target

        # initialisation d'un dict pour stocker les coordonees des selections
        self.dict_targets = {}
        
        # initialisation d'un dict pour la position a t0
        self.position_t0 = {}
        
        # dictionnaire pour stocker les dimensions des cibles selectionnees
        self.dict_dimensions = {}
                
        # boucle sur le nombre de cibles
        for n in range(n_target):
            
            print("\nSelection de la cible {}\n".format(n))
            
            # definition de la figure de selection
            self.fig, self.ax = plt.subplots(figsize = (8,6))
            
            # affichage de l'image 
            self.ax.imshow(self.image_t0, cmap = 'Greys_r')
            self.ax.axis('off')
            
            # lancement de la fonction de selection
            self.RS = RectangleSelector(self.ax, self.line_select_callback,
                                                   useblit=True,
                                                   button=[1,3],  # don't use middle button
                                                   minspanx=5, minspany=5,
                                                   spancoords='pixels',
                                                   interactive=True)
            
            
            # ajout des titres et des legendes
            plt.title("\nSelection de la cible {}\n".format(n), {'fontweight' : 'bold'})
            plt.xlabel("Presser enter lorsque la selection est faite")            
            
            # activation du mode interactif
            plt.connect('key_press_event', self.toggle_selector)
            
            # affichage d'un carree autour des cibles deja selectionnees
            if n != 0:
                for i in self.position_t0:
                    
                    x_i = self.position_t0[i][0]
                    y_i = self.position_t0[i][1]
                    
                    h_sel = self.dict_dimensions[i][0]
                    l_sel = self.dict_dimensions[i][1]
                    
                    rect = plt.Rectangle((y_i-0.5*h_sel, x_i-0.5*l_sel), h_sel, l_sel, color='tab:red', fc='none')
                    
                    self.ax.add_patch(rect)

                # affichage de la fenetre
                plt.show()                    

                    
            elif n == self.n_target - 1:
                plt.close()
            
            else:
                plt.show()              

    
    #=========================================================================#
    #                  fonctions pour la selection des cibles                 #
    #=========================================================================#
    
    
    def line_select_callback(self,eclick, erelease):
        """
        Fonction qui affiche les coordonnees selectionnees 

        """
        
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        print("\tselection : (%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))

    def toggle_selector(self,event):
        """
        Fonction qui permet de stocker la selection et de fermer la fenetre
        lorsque la selection est terminee
        """
        if event.key == 'enter' and self.RS.active:
            
            print("\nSauvegarde de la selection.")
            
            n = len(self.dict_targets.keys())
            
            x1 = int(self.RS.extents[0])
            x2 = int(self.RS.extents[1])

            y1 = int(self.RS.extents[2])
            y2 = int(self.RS.extents[3])

            self.x1 = x1
            self.y1 = y1
            
            width = abs(x2-x1)
            height = abs(y2-y1)


            self.dict_targets[n] = self.image_t0[y1:y2,x1:x2]
            self.position_t0[n] = (int(y1+height/2),int(x1+width/2))
            self.dict_dimensions[n] = (width, height)
            
            plt.close()

    def target_tracking(self, binarize = False, output_txt = True, actualise = False, save_track = False, **kwargs):
        """
        
        fonction qui permet de suivre la position des cibles au sein des images
        
        Parameters
        ----------
        binarize : bool, optional
            Si binarize = True, toutes les images sont binarisees.
            La valeur par defaut est False.
            
        output_txt : bool ou str, optionnel
            option pour sauvegarder ou non le dictionnaire de suivi dans un 
            fichier txt. Possibilite de mettre True pour sauvegarder dans le 
            dossier courant, ou d'indiquer directement le dossier de sauvegarde.
            La valeur par defaut est True
            
        actualise : bool, optional
            quand actualise = True, l'algorithme actualise les motifs de recherche
            avec les motifs trouves a chaque step. 
            La valeur par defaut est False.
        
        save_track : bool ou str, optional
            permet de choisir de sauvegarder ou des images de suivi du tracking.
            Le but de cette option est de pouvoir verifier le bon fonctionnement
            de l'algorithme. 
            
            save_track = True : le dossier contenant les images est sauvegarde 
                                dans le dossier de lancement du script.
            
            save_track = chemin : le dossier est sauvegarde vers le chemin specifie.
            
        **kwargs : 
        
        h_fenetre : int
            hauteur de la fenetre de recherche. Par defaut elle est fixee a deux 
            fois la hauteur de la cible.

        l_fenetre : int
            largeur de la fenetre de recherche. Par defaut elle est fixee a deux 
            fois la largeur de la cible.

        Returns
        -------
        dict_suivi : dict
            dictionnaire contenant la position de chaque cible au sein de chaque
            image.

        """
        
        # stockage du temps au demarrage
        start = time.time()
                
        # si l'input donnee est un fichier et qu'aucun dossier n'a ete renseigne
        # -> affichage d'une erreur et arret de la fonction, sinon le script se
        #    lance
        if os.path.isfile(self.chemin_input):
            print("\nIl faut specifier un dossier et non un fichier.")
        else:
            # print("Debut tracking...")
            
            # recuperation de la liste des images
            liste_images = [x for x in os.listdir(self.chemin_input) if x!='.DS_Store']
            
            # tri des images par precaution 
            Y = [ int(x.split('.')[0]) for x in liste_images]

            liste_images_sorted = [x for _,x in sorted(zip(Y,liste_images))]
            
            # initialisation d'un dictionnaire pour stocker les infos a 
            # l'instance i
            self.dict_suivi = {}
            
            empty = False
            
            i_empty = len(liste_images_sorted)-1
            
            # parcours du nom des fichiers images 
            for i,nm in enumerate(liste_images_sorted):
                
                print("\tLecture de l'image '{}' | ({}/{})".format(nm,i+1,(len(liste_images))))
                
                # creation d'une cle a la racine du dictionnaire de suivi
                self.dict_suivi[i] = {}
                
                # recuperation du chemin vers le fichier 
                chemin_image_i = os.path.join(self.chemin_input,nm)
                
                self.binarize = binarize
                
                # chargement de l'image 
                if self.binarize:
                    
                    if not self.rotation:
                        img = imread(chemin_image_i, as_gray=True)
                    else:
                        img = rotate(imread(chemin_image_i, as_gray=True),270,resize=True)
                    
                    thresh = threshold_otsu(img)
                    binary = img > thresh
                    

                    image_i = binary.astype(float)
                else:
                    if not self.rotation:
                        image_i = imread(chemin_image_i, as_gray=True)
                    else:
                        image_i = rotate(imread(chemin_image_i, as_gray=True),270,resize=True)                
                
                # creation de la figure si sauvegarde voulue
                if save_track:
                    plt.figure(num = nm)
                    imshow(image_i)
                
                # parcours des templates 
                for n in range(self.n_target):
                
                    # dimensions_motif = self.dict_templates[n].shape
                    
                    self.dict_suivi[i][n]  = {}
                    # recuperation des dimensions du template n
                    l_motif, h_motif = self.dict_targets[n].shape
                    
                    
                    h_fenetre = kwargs.get('h_fenetre', h_motif*2.0)
                    l_fenetre = kwargs.get('l_fenetre', l_motif*2.0)
                        
                    # print('Recherche motif {}/{}\n'.format(j,len(self.position_t0[n].keys())-1))
                    
                    
                    if i == 0:
                        
                        # print('\tRecuperation de la position initiale...\n')
                        # recuperation de la position du noeud a t0
                        x_i_prev = self.position_t0[n][0]
                        y_i_prev = self.position_t0[n][1]
                        
                        if save_track:
                            plt.figure(num = nm)
                            # rectangle motif de base
                            plt.plot(y_i_prev, x_i_prev,'rx')
                        
                        self.dict_suivi[i][n] = (x_i_prev, y_i_prev)
                        
                    else:
                        
                        # print('\tRecuperation de la position precedente...\n')
                        # sinon recuperation de la position du motif au 
                        # step precedent
                        x_i_prev = self.dict_suivi[i-1][n][0]
                        y_i_prev = self.dict_suivi[i-1][n][1]
                        
                        # rognage de l'image pour garder seulement la zone de 
                        # la fenetre avec la taille souhaitee
                        
                        # print('\tDefinition de la fenetre de recherche : ')
                        
                    
                        x_start = int(x_i_prev-(l_fenetre*0.5))
                        x_end = int(x_i_prev+(l_fenetre*0.5))
                        
                        y_start = int(y_i_prev-(h_fenetre*0.5))
                        y_end = int(y_i_prev+(h_fenetre*0.5))
                        
                        # print('\t\tx_start = {}, x_end = {}, y_start = {}, y_end = {}\n'.format(x_start, x_end, y_start, y_end))
                        
                        # rognage de l'image pour ne garder que la zone dans 
                        # laquelle la recherche est a effectuer
                        zone_correlation = image_i[x_start:x_end,y_start:y_end]
                        
                        if len(zone_correlation) == 0:
                            
                            empty = True
                            
                            break
                        
                        # utilisation de la fonction de template matching dans 
                        # la zone de recherche
                        matrice_correlation_n_i = match_template(zone_correlation, self.dict_targets[n])
                        
                        correlation_max = np.unravel_index(matrice_correlation_n_i.argmax(), matrice_correlation_n_i.shape)
                                                    
                        x_i_loc = correlation_max[0] + 0.5*l_motif
                        y_i_loc = correlation_max[1] + 0.5*h_motif

                        x_i_glob = x_i_loc + x_start
                        y_i_glob = y_i_loc + y_start
                        
                        nouv_motif_ti = zone_correlation[correlation_max[0]:correlation_max[0]+l_motif,correlation_max[1]:correlation_max[1]+h_motif]
                        
                        if actualise:
                            self.dict_targets[n] = nouv_motif_ti
                        
                        if save_track:
                            plt.figure(num = nm)
                            # rectangle motif de base
                            plt.plot(y_i_prev, x_i_prev,'rx')
                            # rect_motif = plt.Rectangle((y_i_prev-0.5*h_motif, x_i_prev-0.5*l_motif), h_motif, l_motif, color='r', fc='none')
                            # plt.gca().add_patch(rect_motif)                        
                            
                            # rectangle recherche
                            rect_zone_recherche = plt.Rectangle((y_start, x_start), h_fenetre, l_fenetre,
                                                                color='b', fc='none')
                            plt.gca().add_patch(rect_zone_recherche)
                            
                            # rectangle motif trouve
                            plt.plot(y_i_glob, x_i_glob,'gx')
                            rect_trouve = plt.Rectangle((y_i_glob-0.5*h_motif, x_i_glob-0.5*l_motif), h_motif, l_motif, color='g', fc='none')
                            plt.gca().add_patch(rect_trouve)
                            
                        
                        # stockage de la position trouvee dans le dictionnaire
                        # de sortie
                        self.dict_suivi[i][n] = (x_i_glob, y_i_glob)
                
                if empty:
                    
                    i_empty = i
                    
                    break
                
                # sauvegarde des figures de suivi si demande
                if save_track or type(save_track)==str:
                    
                    # si premier instant -> creation d'un dossier de sauvegarde
                    if i == 0:
                        
                        if save_track == True:
                            
                            dossier_out = os.getcwd()
                            nom_dossier_save_track = os.path.join(dossier_out,"tracking")
                        
                        elif type(save_track)==str:
                            nom_dossier_save_track = save_track
                            
                        # si il existe deja, il est nettoye
                        
                        if os.path.isdir(nom_dossier_save_track):
                            for f in os.listdir(nom_dossier_save_track):
                                os.remove(os.path.join(nom_dossier_save_track,f))
                        else:
                            os.mkdir(nom_dossier_save_track)

                            
                            

    
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                                hspace = 0, wspace = 0)
                    plt.margins(0,0)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
                    # sauvegarde de la figure
                    plt.savefig(os.path.join(nom_dossier_save_track,nm), dpi=200, bbox_inches = 'tight',pad_inches = 0)
                    
                    plt.close()
                    
                    
        # stockage du temps en fin de script    
        end = time.time()
        
        # affichage du temps d'execution 
        print("\nTemps d'execution : {:.2f} secondes\n".format(end-start))
        
        # sauvegarde du fichier output
        if output_txt == True:
            # si l'entree est un bool True, le fichier est enregistre dans le 
            # dossier de lancement du script
            self.write_tracking(os.getcwd(), i_empty)
        elif type(output_txt) == str:
            # sinon le dossier est ecrit dans le dossier specifie
            self.write_tracking(output_txt, i_empty)
        
        return self.dict_suivi
        
        
    def write_tracking(self, dossier, i_empty):
        """
        Fonction qui ecrit le contenu du dictionnaire de tracking dans des 
        fichiers propres a chaque motif

        """
        nom_fichier = os.path.join(dossier,'output_suivi.txt')
        
        if 'output_suivi.txt' in os.listdir(dossier):
            os.remove(nom_fichier)
        
        # boucle d'ecriture des fichiers
        for i in list(self.dict_suivi.keys())[:i_empty]:
            for n in self.dict_suivi[i]:
                with open(nom_fichier, 'a') as f:
                        if i == 0 and n ==0:
                            for k in self.dict_suivi[i]:
                                if k == 0:
                                    f.write("step\tX{0}\tY{0}".format(k))
                                else:
                                    f.write("\tX{0}\tY{0}".format(k))
                        
                        if n == 0:
                            f.write("\n{}\t{}\t{}".format(i,self.dict_suivi[i][n][0],self.dict_suivi[i][n][1]))
                        else:
                            f.write("\t{}\t{}".format(self.dict_suivi[i][n][0],self.dict_suivi[i][n][1]))



class TemplateMatching:
    """
    Class pour le suivi de motifs dans une liste d'images.
    
    Etapes pour utiliser cette class : 
        - definir un chemin vers un dossier avec les images de l'essai,
        - definir le nombre de motifs à suivre dans les images,
        - initialiser un objet appartenant a la class avec le chemin et le nombre
          de motifs,
        - utiliser la methode template_matching_t0 pour la detection des motifs
        avant le suivi,
        - utiliser la methode template_tracking en renseignant les differentes options.
    
    Le scripts doit obligatoirement etre lance dans une fenetre de terminal pour
    que la selection des motifs fonctionne.
    
    """
    
    
    def __init__(self, chemin_input, n_motifs, save_template = False, template = False, binarize = False):
        """
        Initialisation de la class a partir du dossier contenant les images et
        le nombre de motifs.
        
        L'initialisation d'un objet de cette class lance la selection des motifs.

        Parameters
        ----------
        chemin_input : str
            chemin vers le dossier contenant les images.
            
        n_motifs : int
            le nombre de motifs a suivre.
            
        save_template : bool, optional
            sauvegarde des motifs selectionnes si True ou non si False.
            La valeur par défaut est False.
            
        template : str, optional
            chemin vers un motif deja selectionne auparavant. 
            La valeur par defaut est False.
            Ne marche pas avec plusieurs motifs.
            
        binarize : bool, optional
            Si binarize = True, toutes les images sont binarisees.
            La valeur par defaut est False.
            La binarisation n'ameliore pas la detection des motifs.

        """
        
        self.chemin_input = chemin_input
        
        # test pour determiner le type d'entree renseigne par l'utilisateur
        if os.path.isdir(chemin_input):
            # l'input est un dossier, il faut donc recuperer la premiere image
            self.type_input = 'dir'
            
            # de ce dossier pour obtenir l'etat a t = 0
            # recuperation de la liste des images
            liste_images = [x for x in os.listdir(chemin_input) if x!='.DS_Store']
            
            # tri des images par precaution 
            Y = [ int(x.split('.')[0]) for x in liste_images]

            liste_images_sorted = [x for _,x in sorted(zip(Y,liste_images))]
            
            self.chemin_image_t0 = os.path.join(chemin_input,liste_images_sorted[0])
        
        else:
            # sinon l'input est un chemin vers un fichier
            self.type_input = 'file'
            
            # et l'image peut etre chargee directement via ce chemin
            self.chemin_image_t0 = chemin_input
        
        
        # lecture de l'image a t0
        # si l'option binarize est demande, l'image est binarisee, sinon elle 
        # est lue en nuance de gris.
        if binarize:
            img = imread(self.chemin_image_t0, as_gray=True)
            thresh = threshold_otsu(img)
            binary = img > thresh

            self.image_t0 = binary.astype(np.uint8)
                        
            self.binarize = True
        else: 
            self.image_t0 = imread(self.chemin_image_t0, as_gray=True)
            self.binarize = False
            
        #------------ selection des templates a l'initialisation -------------#
        
        # stockage du nombre de motifs indiques par l'utilisateur
        self.n_motifs = n_motifs
        
        # initialisation d'un dict pour stocker les coordonees des selections
        self.dict_templates = {}
        
        if not template:
            # boucle sur le nombre de templates
            for n in range(n_motifs):
                
                print("\nSelection of template {}\n".format(n))
                
                # definition de la figure de selection
                self.fig, self.ax = plt.subplots(num=n, figsize = (8,6))
    
                # affichage de l'image 
                self.ax.imshow(self.image_t0, cmap = 'Greys_r')
                                
                # lancement de la fonction de selection
                self.RS = RectangleSelector(self.ax, self.line_select_callback,
                                                       drawtype='box', useblit=True,
                                                       button=[1,3],  # don't use middle button
                                                       minspanx=5, minspany=5,
                                                       spancoords='pixels',
                                                       interactive=True)
                
                # ajout des titres et des legendes
                plt.title("Template {} selection".format(n), {'fontweight' : 'bold'})
                plt.xlabel("Press enter when finished")            
                
                # activation du mode interactif
                plt.connect('key_press_event', self.toggle_selector)
                

                
                # affichage de la fenetre
                plt.show()
                
        else:
            self.dict_templates[0] = imread(template)
        
        if save_template:
            for n in self.dict_templates:
                imsave('template_{}.png'.format(n),self.dict_templates[n] )
        
        
        #---------------- initialisation des variables utiles ----------------#
        
        self.is_t0 = False
        
        
    #=========================================================================#
    #                    fonctions pour la selection des templates            #
    #=========================================================================#
    
    
    def line_select_callback(self,eclick, erelease):
        """
        Fonction qui affiche les coordonnees selectionnees 

        """
        
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        print("\tselection : (%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))

    def toggle_selector(self,event):
        """
        Fonction qui permet de stocker la selection et de fermer la fenetre
        lorsque la selection est terminee
        """
        # print(' Key pressed.')
        if event.key == 'enter' and self.RS.active:
            
            print("\nSauvegarde de la selection.")
            
            n = len(self.dict_templates.keys())
            
            x1 = int(self.RS.extents[0])
            x2 = int(self.RS.extents[1])

            y1 = int(self.RS.extents[2])
            y2 = int(self.RS.extents[3])

            
            self.dict_templates[n] = self.image_t0[y1:y2,x1:x2]
            
            self.RS.set_active(False)
            
            plt.close()
 

    #=========================================================================#
    #                    fonctions pour le template matching                  #
    #=========================================================================#

    def template_matching_t0(self, show = True, save = False, thr= 0.6):
        """
        Fonction qui recherche et affiche les occurences trouvees pour l'image a t0

        Parameters
        ----------
        show : bool, optional
            si show = True un graphique avec les occurences est affiche pour 
            chaque motif. 
            La valeur par defaut est True.
        save : bool, optional
            si save = True le graphique avec les occurences est sauvegarde. 
            La valeur par defaut est False.
        thr : float, optional
            thr est compris entre 0.0 et 1.0 et controle le threshold de detection 
            de motif dans la matrice de correlation. Si un point obtient une valeur 
            superieure a thr, alors un motif sera detecte a ce point. Plus cette 
            valeur est haute, plus l'algorithme sera exigent sur la ressemblance 
            avec le motif d'entree, et inversement.
            La valeur par defaut est de 0.6

        Returns
        -------
        None.

        """

        # definition du dictionnaire de stockage des positions t0
        self.position_t0 = {}
        
        self.templates_t0 = {}
        
        
        # parcours des templates
        for n in range(self.n_motifs):
            
            self.position_t0[n] = {}
            
            self.templates_t0[n] = {}
            
            # utilisation de la fonction match template pour calculer la matrice de correlation
            matrice_correlation = match_template(self.image_t0, self.dict_templates[n])
                        
            # recuperation des dimensions du motif pour l'affichage des carrees
            dimensions_motif = self.dict_templates[n].shape

            # utilisation de la fonction peak_local_max pour recuperer les 
            # maximums locaux dans la matrice de correlation
            peak_local = peak_local_max(matrice_correlation, min_distance=dimensions_motif[0], threshold_abs=thr, exclude_border = True)
                       
            # initialisation d'une variable pour la numerotation des noeuds
            j = 0
            
            # parcours des coordonnees des maximums locaux
            for x, y in peak_local:

                # stockage de la position du noeud dans un dictionnaire
                self.position_t0[n][j] = (x+0.5*dimensions_motif[0],y+0.5*dimensions_motif[1])
                
                self.templates_t0[n][j] = self.image_t0[x:x+dimensions_motif[0],y:y+dimensions_motif[1]]
                
                
                if show or save:
                    # tracage d'un rectangle
                    rect = plt.Rectangle((y, x), dimensions_motif[1], dimensions_motif[0], 
                                         color='r', fc='none')
                    plt.gca().add_patch(rect)
                    
                    # affichage du numero de l'objet detecte
                    plt.text(y+0.5*dimensions_motif[1], x+0.5*dimensions_motif[0], j
                             , color = 'white', weight = 'bold', ha = 'center', va = 'center')
                
                # incrementation de j 
                j += 1
        
        if show or save:
            # affichage de l'image
            imshow(self.image_t0, cmap = 'Greys_r')
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            
            
        
        # sauvegarde du graph si demande
        if save != False:
            if type(save) == str:
                plt.savefig(os.path.join(save,'t0.png'),dpi=250, bbox_inches = 'tight',pad_inches = 0)
            else:
                plt.savefig('t0.png',dpi=250, bbox_inches = 'tight',pad_inches = 0)
        
        # affichage du graphique
        if show:
            plt.show()
        
        # modification de la variable qui indique si le template matching a t0
        # a ete fait
        self.is_t0 = True
        

    def template_tracking(self, save = True, actualise = False, save_track = False, **kwargs):
        """
        
        Fonction qui permet le suivi des points detectes avec le template matching
        
        Parameters
        ----------
        output_txt : bool ou str, optionnel
            option pour sauvegarder ou non le dictionnaire de suivi dans un 
            fichier txt. Possibilite de mettre True pour sauvegarder dans le 
            dossier courant, ou d'indiquer directement le dossier de sauvegarde.
            La valeur par defaut est True
            
        actualise : bool, optional
            quand actualise = True, l'algorithme actualise les motifs de recherche
            avec les motifs trouves a chaque step. 
            La valeur par defaut est False.
        
        save_track : bool ou str, optional
            permet de choisir de sauvegarder ou des images de suivi du tracking.
            Le but de cette option est de pouvoir verifier le bon fonctionnement
            de l'algorithme. 
            
            save_track = True : le dossier contenant les images est sauvegarde 
                                dans le dossier de lancement du script.
            
            save_track = chemin : le dossier est sauvegarde vers le chemin specifie.
            
        **kwargs : 
        
        save_zone : list
            liste des zones de recherche a exporter durant le suivi. Les zones 
            de recherche correspondent au numero indique dans la figure donnant
            les motifs a t0.
        
        h_fenetre : int
            hauteur de la fenetre de recherche. Par defaut elle est fixee a deux 
            fois la hauteur de la cible.

        l_fenetre : int
            largeur de la fenetre de recherche. Par defaut elle est fixee a deux 
            fois la largeur de la cible.

        Returns
        -------
        dict_suivi : dict
            dictionnaire contenant la position de chaque cible au sein de chaque
            image.

        """
        
        # stockage du temps au demarrage
        start = time.time()
        
        # analyse de l'image a t0 si cela n'a pas ete fait 
        if not self.is_t0:
            self.template_matching_t0(show = False, save = False)
        
        save_zone = kwargs.get('save_zone', [])
        
        if len(save_zone) != 0:
            for i in save_zone:
                try:
                    os.mkdir(os.path.join(save,"save_zone_{}".format(i)))
                except:
                    pass
        
        # si l'input donnee est un fichier et qu'aucun dossier n'a ete renseigne
        # -> affichage d'une erreur et arret de la fonction, sinon le script se
        #    lance
        if os.path.isfile(self.chemin_input):
            print("\nIl faut specifier un dossier et non un fichier 'dossier='.")
        else:
            # print("Debut tracking...")
            
            # recuperation de la liste des images
            liste_images = [x for x in os.listdir(self.chemin_input) if x!='.DS_Store']
            
            # tri des images par precaution 
            Y = [ int(x.split('.')[0]) for x in liste_images]

            liste_images_sorted = [x for _,x in sorted(zip(Y,liste_images))]
            
            # initialisation d'un dictionnaire pour stocker les infos a 
            # l'instance i
            self.dict_suivi = {}
            
            # parcours du nom des fichiers images 
            for i,nm in enumerate(liste_images_sorted):
                
                print("\tLecture image '{}' | ({}/{})".format(nm,i+1,(len(liste_images))))
                
                # creation d'une cle a la racine du dictionnaire de suivi
                self.dict_suivi[i] = {}
                
                # recuperation du chemin vers le fichier 
                chemin_image_i = os.path.join(self.chemin_input,nm)
                
                # chargement de l'image 
                if self.binarize:
                    img = imread(chemin_image_i, as_gray=True)
                    thresh = threshold_otsu(img)
                    binary = img > thresh
    
                    image_i = binary.astype(float)
                
                else:
                    image_i = imread(chemin_image_i, as_gray=True)
                
                # creation de la figure si sauvegarde voulue
                if save_track:
                    plt.figure(num = nm)
                    imshow(image_i)
                
                # parcours des templates 
                for n in range(self.n_motifs):
                
                    # dimensions_motif = self.dict_templates[n].shape
                    
                    self.dict_suivi[i][n]  = {}
                    # recuperation des dimensions du template n
                    l_motif, h_motif = self.dict_templates[n].shape
                    
                    
                    h_fenetre = kwargs.get('h_fenetre', h_motif*2.0)
                    l_fenetre = kwargs.get('l_fenetre', l_motif*2.0)
                    
                    
                    # parcours des points detectes lors de l'initialisation
                    for j in self.position_t0[n].keys():
                        
                        # print('Recherche motif {}/{}\n'.format(j,len(self.position_t0[n].keys())-1))
                        
                        
                        if i == 0:
                            
                            # print('\tRecuperation de la position initiale...\n')
                            # recuperation de la position du noeud a t0
                            x_i_prev = self.position_t0[n][j][0]
                            y_i_prev = self.position_t0[n][j][1]
                            if save_track:
                                plt.figure(num = nm)
                                # rectangle motif de base
                                plt.plot(y_i_prev, x_i_prev,'rx')
                            
                            self.dict_suivi[i][n][j] = (x_i_prev, y_i_prev)
                            
                        else:
                            
                            # print('\tRecuperation de la position precedente...\n')
                            # sinon recuperation de la position du motif au 
                            # step precedent
                            x_i_prev = self.dict_suivi[i-1][n][j][0]
                            y_i_prev = self.dict_suivi[i-1][n][j][1]
                            
                            # rognage de l'image pour garder seulement la zone de 
                            # la fenetre avec la taille souhaitee
                            
                            # print('\tDefinition de la fenetre de recherche : ')
                            
                        
                            x_start = int(x_i_prev-(l_fenetre*0.5))
                            x_end = int(x_i_prev+(l_fenetre*0.5))
                            
                            y_start = int(y_i_prev-(h_fenetre*0.5))
                            y_end = int(y_i_prev+(h_fenetre*0.5))
                            
                            # print('\t\tx_start = {}, x_end = {}, y_start = {}, y_end = {}\n'.format(x_start, x_end, y_start, y_end))
                            
                            # rognage de l'image pour ne garder que la zone dans 
                            # laquelle la recherche est a effectuer
                            zone_correlation = image_i[x_start:x_end,y_start:y_end]
                            
                            if len(save_zone) != 0:
                                if j in save_zone:
                                    plt.figure('zone')
                                    imshow(zone_correlation)
                                    plt.gca().set_axis_off()
                                    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                                                hspace = 0, wspace = 0)
                                    plt.margins(0,0)
                                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                                    plt.gca().yaxis.set_major_locator(plt.NullLocator())

                                    plt.savefig(os.path.join(os.path.join(save,"save_zone_{}".format(j)),nm), dpi=200, bbox_inches = 'tight',pad_inches = 0)
                                    
                                    plt.close("zone")
                                    
                            # utilisation de la fonction de template matching dans 
                            # la zone de recherche
                            matrice_correlation_n_i = match_template(zone_correlation, self.templates_t0[n][j])
                            
                            correlation_max = np.unravel_index(matrice_correlation_n_i.argmax(), matrice_correlation_n_i.shape)
                                                        
                            x_i_loc = correlation_max[0] + 0.5*l_motif
                            y_i_loc = correlation_max[1] + 0.5*h_motif

                            x_i_glob = x_i_loc + x_start
                            y_i_glob = y_i_loc + y_start
                            
                            nouv_motif_ti = zone_correlation[correlation_max[0]:correlation_max[0]+l_motif,correlation_max[1]:correlation_max[1]+h_motif]
                            
                            if actualise:
                                self.templates_t0[n][j] = nouv_motif_ti
                            
                            if save_track:
                                plt.figure(num = nm)
                                # rectangle motif de base
                                plt.plot(y_i_prev, x_i_prev,'rx')
                                # rect_motif = plt.Rectangle((y_i_prev-0.5*h_motif, x_i_prev-0.5*l_motif), h_motif, l_motif, color='r', fc='none')
                                # plt.gca().add_patch(rect_motif)                        
                                
                                # rectangle recherche
                                rect_zone_recherche = plt.Rectangle((y_start, x_start), h_fenetre, l_fenetre,
                                                                    color='b', fc='none')
                                plt.gca().add_patch(rect_zone_recherche)
                                
                                # rectangle motif trouve
                                plt.plot(y_i_glob, x_i_glob,'gx')
                                rect_trouve = plt.Rectangle((y_i_glob-0.5*h_motif, x_i_glob-0.5*l_motif), h_motif, l_motif, color='g', fc='none')
                                plt.gca().add_patch(rect_trouve)  
                                
                            
                            # stockage de la position trouvee dans le dictionnaire
                            # de sortie
                            self.dict_suivi[i][n][j] = (x_i_glob, y_i_glob)
                
                # sauvegarde des figures de suivi si demande
                if save_track:
                    
                    # si premier instant -> creation d'un dossier de sauvegarde
                    if i == 0:
                        nom_dossier_save_track = "tracking"
                        # si il existe deja, il est nettoye
                        if nom_dossier_save_track in os.listdir(os.getcwd()):
                            for f in os.listdir(nom_dossier_save_track):
                                os.remove(os.path.join(nom_dossier_save_track,f))
                        else:
                            os.mkdir(nom_dossier_save_track)

                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                                hspace = 0, wspace = 0)
                    plt.margins(0,0)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())

                    # sauvegarde de la figure
                    plt.savefig(os.path.join(nom_dossier_save_track,nm), dpi=200, bbox_inches = 'tight',pad_inches = 0)
                    
                    if i == len(liste_images)-1:
                        make_gif(nom_dossier_save_track)
                    
                    
                    
        # stockage du temps en fin de script    
        end = time.time()
        
        # affichage du temps d'execution 
        print("\nTemps d'execution : {:.2f} secondes\n".format(end-start))
        
        # recuperation de l'option de sauvegarde
        
        
        if not save:
            # si rien n'est rentre, aucun  fichier n'est exporte
            print("Tracking output not asked\n")
        else:
            # sinon le dossier est ecrit dans le dossier specifie
            self.write_tracking(save)
            
        return self.dict_suivi
        
        
    def write_tracking(self, dossier):
        """
        Fonction qui ecrit le contenu du dictionnaire de tracking dans des 
        fichiers propres a chaque motif

        """
        # suppression des fichier deja present
        # eventuellement a changer (inclure la date dans le nom de fichier)
        for n in range(self.n_motifs):
            # si le fichier est present dans le dossier de lancement -> del
            if 'output_template_{}.txt'.format(n) in os.listdir(dossier):
                os.remove(os.path.join(dossier,'output_template_{}.txt').format(n))
        
        # boucle d'ecriture des fichiers
        for s in self.dict_suivi:
            for n in self.dict_suivi[s]:
                
                nom_fichier = os.path.join(dossier,'output_template_{}.txt'.format(n))
                
                with open(nom_fichier, 'a') as f:
                    for j in self.dict_suivi[s][n]:
                        if s == 0 and j ==0:
                            for k in self.dict_suivi[s][n]:
                                if k == 0:
                                    f.write("step\tX{0}\tY{0}".format(k))
                                else:
                                    f.write("\tX{0}\tY{0}".format(k))
                        
                        if j == 0:
                            f.write("\n{}\t{}\t{}".format(s,self.dict_suivi[s][n][j][0],self.dict_suivi[s][n][j][1]))
                        else:
                            f.write("\t{}\t{}".format(self.dict_suivi[s][n][j][0],self.dict_suivi[s][n][j][1]))

    def min_distances(self):
        """
        Fonction qui permet de trouver la distance minimale entre les motifs trouves

        Returns
        -------
        dict_min_distance : dict
            dictionnaire avec les distances minimales pour chaque motif.

        """
        self.dict_min_distance = {}
        
        for n in self.position_t0:
            
            liste_distances = []
            
            for i in self.position_t0[n]:
                
                position_noeud_i = self.position_t0[0][i]
                
                x_i = position_noeud_i[0]
                y_i = position_noeud_i[1]
                
                for j in self.dict_suivi[n]:
                    if j != i:
                        
                        position_noeud_j = self.position_t0[0][j]
                        x_j = position_noeud_j[0]
                        y_j = position_noeud_j[1]
                        
                        distance_i_j = ((x_j-x_i)**2 + (y_j-y_i)**2)**0.5
                        
                        liste_distances.append(distance_i_j)
            
            self.dict_min_distance[n] = min(liste_distances)            
            
            return self.dict_min_distance
        
        
        
        
        




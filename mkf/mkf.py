# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:58:35 2023

@author: llegue
"""

import os
import subprocess
from datetime import datetime
import streamlit as st
from streamlit_extras.let_it_rain import rain
from skimage.io import imread, imshow
from skimage.transform import rotate
from moviepy.editor import VideoFileClip, clips_array
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import plotly.express as px
import matplotlib as mpl
import tkinter
mpl.use("TkAgg")



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
        Longueur du gif a creer en secondes. La valeur par defaut est de 200.

    Returns
    -------
    None.

    """
    
    # recuperation de la liste d'images
    liste_images = [x for x in os.listdir(chemin_dossier_images) if x != '.DS_Store']
    
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


def gif_to_mp4(gif_file, mp4_file, fps=30):
    """
    Convertit un fichier GIF en un fichier MP4 en utilisant la bibliothèque moviepy.

    :param gif_file: Le chemin vers le fichier GIF d'entrée.
    :param mp4_file: Le nom du fichier de sortie pour la vidéo MP4.
    :param fps: Taux de trame de la vidéo de sortie (images par seconde).
    """
    video_clip = VideoFileClip(gif_file)
    
    # Tentative d'extraction du taux de trame à partir du fichier GIF
    if video_clip.fps is not None:
        fps = video_clip.fps
    
    video_clip.write_videofile(mp4_file, fps=fps)
    print(f"La vidéo MP4 '{mp4_file}' a été créée avec succès à partir du GIF '{gif_file}'.")


st.image("https://upload.wikimedia.org/wikipedia/fr/thumb/c/c8/Institut_fran%C3%A7ais_de_recherche_pour_l%27exploitation_de_la_mer_%28logo%29.svg/1280px-Institut_fran%C3%A7ais_de_recherche_pour_l%27exploitation_de_la_mer_%28logo%29.svg.png",
         width = 350)
st.title('Digital image correlation and tracking for fishing gear mechanical testing')
st.markdown("Description")


#=============================================================================#
#                                                                             #
#                             Test definition                                 #
#                                                                             #
#=============================================================================#

st.divider()

st.subheader(":memo: Test definition")

test_type = st.selectbox('Type of sample',
                         ('yarn', 'knot', 'net'))

if test_type == "yarn":
    n_targets = 2
elif test_type == "knot":
    n_targets = 4
elif test_type == "net":
    n_targets = st.number_input("Targets number:", min_value=1, max_value=None, step=1)


dossier_input = st.text_input('Test folder')

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("gif export:")

with col2:
    is_tracking = st.toggle("tracking")
    
with col3:
    is_result = st.toggle("result")
    
with col4:
    is_rotate = st.toggle("rotate")

dossier_output = os.path.join(dossier_input,"tracking_outputs")
dossier_tracking = os.path.join(dossier_output,"tracking")
dossier_camera = os.path.join(dossier_input,"images")
dossier_frames = os.path.join(dossier_output,"frames")

fichier_ni = os.path.join(dossier_input,"load_Ni.txt")    

#=============================================================================#
#                                                                             #
#                                 Treatement                                  #
#                                                                             #
#=============================================================================#

st.divider()

st.subheader(":dart: Treatement")

bouton_run = st.button("Run")

if bouton_run:
    subprocess_file = "subprocess_input.inp"
    with open(subprocess_file,'w') as f:
        
        f.write("dossier_basler\t{}\n".format(dossier_input))
        f.write("n_targets\t{}\n".format(n_targets))
        f.write("tracking\t{}\n".format(is_tracking))
        f.write("result\t{}\n".format(is_result))
        f.write("rotate\t{}\n".format(is_rotate))
    
    subprocess.run("python correlation.py", shell = True)
    
    
    if is_tracking:
        
        make_gif(dossier_tracking, os.path.join(dossier_output,"tracking.gif"), length = 100)
        subprocess.run(os.path.join(dossier_output,"tracking.gif"), shell = True)

        
                
#=============================================================================#
#                                                                             #
#                                   Results                                   #
#                                                                             #
#=============================================================================#

st.divider()

st.subheader(":chart_with_upwards_trend: Results")

if test_type == 'net':
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        bouton_plot = st.button("Plot")
    
    with col6:
        export_track = st.toggle("export tracking")
        
    with col7:
        export_file = st.toggle("export file")
    
    with col8:
        is_repartition = st.toggle("repartition")

else:
    col5, col6, col7 = st.columns(3)
    
    with col5:
        bouton_plot = st.button("Plot")
    
    with col6:
        export_track = st.toggle("export tracking")
        
    with col7:
        export_file = st.toggle("export file")
    

if export_track:
    
    downsample = st.slider("Downsample",1,20)

if bouton_plot:
    
    liste_images = [x for x in os.listdir(dossier_camera) if x != '.DS_Store']
    liste_images_sorted = [x for _,x in sorted(zip([int(x.split('.')[0]) for x in liste_images],liste_images))]

    chemin_donnees = os.path.join(dossier_output,"output_suivi.txt")
    positions = pd.read_csv(chemin_donnees,sep='\t')
        
    temps, efforts_kn = read_data_ni(fichier_ni)
    
    efforts = [x*1e3 for x in efforts_kn]
    
    df_ni = pd.DataFrame()
    df_ni["temps"] = temps
    df_ni["efforts"] = efforts
    
    idxmax_effort = df_ni["efforts"].idxmax()
    
    if test_type == "yarn":
        
        # definition des listes pour stocvker les donnees
        liste_x0 = []
        liste_y0 = []

        liste_x1 = []
        liste_y1 = []

        liste_distance = []
        
        for i in range(len(positions["X0"])):
            
            x0_i = positions["X0"][i]
            y0_i = positions["Y0"][i]
        
            x1_i = positions["X1"][i]
            y1_i = positions["Y1"][i]
            
            liste_x0.append(x0_i)
            liste_y0.append(y0_i)

            liste_x1.append(x1_i)
            liste_y1.append(y1_i)
            
            distance_i = ((x1_i - x0_i)**2 + (y1_i - y0_i)**2)**0.5
            
            liste_distance.append(distance_i)

                    
        L0 = liste_distance[0]
        liste_deformation = [(x-L0)/L0 for x in liste_distance]

        fig = px.scatter(x = liste_deformation[:idxmax_effort], y = efforts[:idxmax_effort])
  
        fig.update_layout(xaxis_title="Strain [%]", yaxis_title="Load [N]")
        
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
        if export_file:
            df_output = pd.DataFrame()
            
            df_output["load"] = efforts[:idxmax_effort]
            df_output["strain"] = liste_deformation[:idxmax_effort]
        
            # df_output["load"] = efforts
            # df_output["strain"] = liste_deformation
            
            nom_fichier_txt_out = 'mono_' + dossier_input.split(os.path.sep)[-1] + '.txt'
    
            df_output.to_csv(os.path.join(dossier_output,nom_fichier_txt_out), index = False, sep = '\t')


        if export_track:
            
            progress_text = "Exporting tracking gif..."
            my_bar = st.progress(0, text=progress_text)
            
            if not os.path.isdir(dossier_frames):
                os.mkdir(dossier_frames)
            #else:
                #for f in os.listdir(dossier_frames):
                    #os.remove(os.path.join(dossier_frames,f))
            
            
            for s in range(len(positions["X0"]))[1::downsample]:
                
                x0_i = positions["X0"][s]
                y0_i = positions["Y0"][s]
                
                x1_i = positions["X1"][s]
                y1_i = positions["Y1"][s]   
            
                my_bar.progress((s/len(positions["X0"])), text=progress_text)    
            
                # recuperation du chemon vers le fichier contenant l'image
                chemin_image_i = os.path.join(dossier_camera,liste_images_sorted[s])
                
                # lecture de l'image
                if is_rotate:
                    image_i = rotate(imread(chemin_image_i, as_gray=True),270,resize=True)
                else:
                    image_i = imread(chemin_image_i, as_gray=True)                
                # creation d'une figure correspondant a l'image
                plt.figure(num = s)
                
                # affichage de l'image dans la figure
                imshow(image_i)
                
                # tracage des points
                plt.plot(y0_i, x0_i,'bx')
                plt.plot(y1_i, x1_i,'bx')

                
                # plt.text(0,0, "{}.png".format(s))
                
                plt.gca().set_axis_off()
                
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                            hspace = 0, wspace = 0)
                
                plt.margins(0,0)
                
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                
                nom_image = liste_images_sorted[s].split('.')[0] + ".png"
                
                # sauvegarde de la figure
                plt.savefig(os.path.join(dossier_frames,nom_image), bbox_inches = 'tight',pad_inches = 0,transparent=True)
                
                plt.close()
                
            chemin_mp4 = os.path.join(dossier_output,"knot_strain.mp4")
            
            fps = 30
            os.system("ffmpeg -y -framerate {} -pattern_type glob -i '{}' -c:v libx264 -vf '{}' -pix_fmt yuv420p {}".format(fps,dossier_frames+os.path.sep+'*.png', "pad=ceil(iw/2)*2:ceil(ih/2)*2",chemin_mp4))
            
            
            video_file = open(chemin_mp4, 'rb')
            video_bytes = video_file.read()
            
            my_bar.empty()
            st.markdown("Gif exported.")
            
            st.video(video_bytes)



    elif test_type == "knot":
    ###############################################################################
    #                                      knot                                   #
    ###############################################################################

        # definition des listes pour stocker les donnees
        liste_x01 = []
        liste_y01 = []

        liste_x23 = []
        liste_y23 = []

        liste_distance = []
        
        for i in range(len(positions["X0"])):
            
            x0_i = positions["X0"][i]
            y0_i = positions["Y0"][i]
            
            x1_i = positions["X1"][i]
            y1_i = positions["Y1"][i]   
            
            x01_i = (x0_i + x1_i) * 0.5
            y01_i = (y0_i + y1_i) * 0.5
            
            liste_x01.append(x01_i)
            liste_y01.append(y01_i)
            
            x2_i = positions["X2"][i]
            y2_i = positions["Y2"][i]
            
            x3_i = positions["X3"][i]
            y3_i = positions["Y3"][i]   
            
            x23_i = (x2_i + x3_i) * 0.5
            y23_i = (y2_i + y3_i) * 0.5
            
            liste_x23.append(x23_i)
            liste_y23.append(y23_i)
            
            distance_i = ((x23_i - x01_i)**2 + (y23_i - y01_i)**2)**0.5
            
            liste_distance.append(distance_i)
            
            
        L0 = liste_distance[0]
        liste_deformation = [(x-L0)/L0 for x in liste_distance]
        
        fig = px.scatter(x = liste_deformation[:idxmax_effort], y = efforts[:idxmax_effort])
  
        fig.update_layout(xaxis_title="Strain [%]", yaxis_title="Load [N]")
        
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
        if export_file:
            df_output = pd.DataFrame()
            df_output["load"] = efforts[:idxmax_effort]
            df_output["strain"] = liste_deformation[:idxmax_effort]
            

    
            nom_fichier_txt_out = 'knot_' + dossier_input.split(os.path.sep)[-1] + '.txt'
    
            df_output.to_csv(os.path.join(dossier_output,nom_fichier_txt_out), index = False, sep = '\t')
        
        
        if export_track:
            
            progress_text = "Exporting tracking gif..."
            my_bar = st.progress(0, text=progress_text)
            
            if not os.path.isdir(dossier_frames):
                os.mkdir(dossier_frames)
            #else:
                #for f in os.listdir(dossier_frames):
                    #os.remove(os.path.join(dossier_frames,f))
            
            
            for s in range(len(positions["X0"]))[1::downsample]:
                
                x0_i = positions["X0"][s]
                y0_i = positions["Y0"][s]
                
                x1_i = positions["X1"][s]
                y1_i = positions["Y1"][s]   
                
                x01_i = (x0_i + x1_i) * 0.5
                y01_i = (y0_i + y1_i) * 0.5
                
                liste_x01.append(x01_i)
                liste_y01.append(y01_i)
                
                x2_i = positions["X2"][s]
                y2_i = positions["Y2"][s]
                
                x3_i = positions["X3"][s]
                y3_i = positions["Y3"][s]   
                
                x23_i = (x2_i + x3_i) * 0.5
                y23_i = (y2_i + y3_i) * 0.5
                
                my_bar.progress((s/len(positions["X0"])), text=progress_text)    
            
                # recuperation du chemon vers le fichier contenant l'image
                chemin_image_i = os.path.join(dossier_camera,liste_images_sorted[s])
                
                # lecture de l'image
                image_i = imread(chemin_image_i, as_gray=True)
                
                # creation d'une figure correspondant a l'image
                plt.figure(num = s)
                
                # affichage de l'image dans la figure
                imshow(image_i)
                
                # tracage des points
                plt.plot(y0_i, x0_i,'bx')
                plt.plot(y1_i, x1_i,'bx')
                plt.plot(y2_i, x2_i,'bx')
                
                plt.plot(y3_i, x3_i,'bx')
                
                plt.plot(y01_i, x01_i, 'rx')
                plt.plot(y23_i, x23_i, 'rx')
                
                # plt.text(0,0, "{}.png".format(s))
                
                plt.gca().set_axis_off()
                
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                            hspace = 0, wspace = 0)
                
                plt.margins(0,0)
                
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                
                nom_image = liste_images_sorted[s].split('.')[0] + ".png"
                
                # sauvegarde de la figure
                plt.savefig(os.path.join(dossier_frames,nom_image), bbox_inches = 'tight',pad_inches = 0,transparent=True)
                
                plt.close()
                
            chemin_mp4 = os.path.join(dossier_output,"knot_strain.mp4")
            
            fps = 30
            os.system("ffmpeg -y -framerate {} -pattern_type glob -i '{}' -c:v libx264 -vf '{}' -pix_fmt yuv420p {}".format(fps,dossier_frames+os.path.sep+'*.png', "pad=ceil(iw/2)*2:ceil(ih/2)*2",chemin_mp4))
            
            video_file = open(chemin_mp4, 'rb')
            video_bytes = video_file.read()
            
            my_bar.empty()
            st.markdown("Gif exported.")
            
            st.video(video_bytes)
        

    elif test_type == "net":

        ###############################################################################
        #                                      net                                    #
        ###############################################################################

        #----------- variables relatives a la detection des connexions ---------------#

        thr_distance = 2.3
        thr_angle = 1.5

        #------------- initialisation des dictionnaires de stockage ------------------#

        dict_loc = {}
        dict_def = {}
        dict_l0 = {}

        dict_position_t0 = {}

        dict_connexions = {}
        
        
        #=============================================================================#
        #                         calcul des distances                                #
        #=============================================================================#

        nb_noeuds = int((len(positions.columns)-1)/2)

        # recuperation de la position des noeuds a t = 0
        for i in range(nb_noeuds):
            
            x0_i = positions['X{}'.format(i)][0]
            y0_i = positions['Y{}'.format(i)][0]
            
            dict_position_t0[i] = (y0_i, x0_i)


        # calcul de la plus petite distance entre deux noeuds
        liste_distances = []
        for i in dict_position_t0:
            
            position_noeud_i = dict_position_t0[i]
            
            x_i = position_noeud_i[0]
            y_i = position_noeud_i[1]
            
            for j in dict_position_t0:
                if j != i:
                    
                    position_noeud_j = dict_position_t0[j]
                    x_j = position_noeud_j[0]
                    y_j = position_noeud_j[1]
                    
                    distance_i_j = ((x_j-x_i)**2 + (y_j-y_i)**2)**0.5
                    
                    liste_distances.append(distance_i_j)

        #=============================================================================#
        #                        detection des liaisons                               #
        #=============================================================================#

        min_distance = min(liste_distances)

        # parcours des noeuds i
        for i in dict_position_t0:
            # print("=========== Noeud {} ===========\n".format(i))
            dict_connexions[i] = []
            
            position_noeud_i = dict_position_t0[i]
            
            x_i = position_noeud_i[0]
            y_i = position_noeud_i[1]
            
            # deuxieme parcours de noeuds j pour comparer les distances entre le noeud i
            # et les noeuds j
            for j in dict_position_t0:
                
                # a condition que le noeud j ne soit pas le meme que le noeud i
                if i != j:
                    position_noeud_j = dict_position_t0[j]
                    x_j = position_noeud_j[0]
                    y_j = position_noeud_j[1]
                    
                    distance_i_j = ((x_j-x_i)**2 + (y_j-y_i)**2)**0.5
                    
                    
                    if x_j != x_i:
                        angle_i_j = (y_j-y_i)/(x_j-x_i)
                    else:
                        angle_i_j = float("inf")
                        
                        
                    if distance_i_j < thr_distance*min_distance and abs(angle_i_j)>thr_angle:
                        dict_connexions[i].append(j)

                        if '{}_{}'.format(i,j) in dict_l0.keys() or '{}_{}'.format(j,i) in dict_l0.keys():
                            pass 
                        else:
                            dict_l0['{}_{}'.format(i,j)] = distance_i_j
        

    
        #=============================================================================#
        #                        calcul des deformations                              #
        #=============================================================================#

        dict_suivi_def = {}

        all_def = []

        for s in range(len(positions)):
            
            dict_suivi_def[s] = dict.fromkeys(dict_l0.keys())
            
            for i in dict_position_t0:
                
                x_i = positions['X{}'.format(i)][s]
                y_i = positions['Y{}'.format(i)][s]
                
                for j in dict_connexions[i]:
                    
                    x_j = positions['X{}'.format(j)][s]
                    y_j = positions['Y{}'.format(j)][s]
                    
                    distance_i_j = ((x_j-x_i)**2 + (y_j-y_i)**2)**0.5
                    
                    if '{}_{}'.format(i,j) in dict_suivi_def[s].keys():
                        
                        deformation = (distance_i_j - dict_l0['{}_{}'.format(i,j)])/dict_l0['{}_{}'.format(i,j)]
                        
                        dict_suivi_def[s]['{}_{}'.format(i,j)] = deformation
                        
                        all_def.append(deformation)
            
            all_def.sort()

            def_min = []
            def_moy = []
            def_max = []
            
            for s in dict_suivi_def:
                liste_def_s = []
                for i in dict_suivi_def[s]:
                    liste_def_s.append(dict_suivi_def[s][i])
                
                def_min.append(min(liste_def_s))
                def_moy.append(sum(liste_def_s)/len(liste_def_s))
                def_max.append(max(liste_def_s))
    
        data = pd.DataFrame({
            'X': def_moy[:idxmax_effort],
            'Y': efforts[:idxmax_effort]})
        
        fig = px.scatter(data, x='X', y='Y')
                
        fig.update_layout(xaxis_title="Strain [%]", yaxis_title="Load [N]")

        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
        
        if export_file:
            df_output = pd.DataFrame()
            df_output["load"] = efforts[:idxmax_effort]
            df_output["def_moy"] = def_moy[:idxmax_effort]
            df_output["def_min"] = def_min[:idxmax_effort]
            df_output["def_max"] = def_max[:idxmax_effort]
            
        
            nom_fichier_txt_out = 'net_' + dossier_input.split(os.path.sep)[-1] + '.txt'
            
            df_output.to_csv(os.path.join(dossier_output,nom_fichier_txt_out), index = False, sep = '\t')
        
        if export_track:
            
            if not os.path.isdir(dossier_frames):
                os.mkdir(dossier_frames)
            
            cmap = plt.get_cmap('rainbow',len(all_def))
            norm = mpl.colors.Normalize(vmin=min(all_def), vmax=max(all_def))
                
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
        
            dict_couleurs = {}
        
            colors = plt.cm.rainbow(np.linspace(0, 1, len(all_def)))
            for n,i in enumerate(all_def):
                dict_couleurs[i] = colors[n]
            
            progress_text = "Exporting tracking gif..."
            my_bar = st.progress(0, text=progress_text)
                            
            
            for s in range(len(positions["X0"]))[1::downsample]:
                
                
                my_bar.progress((s/len(positions["X0"])), text=progress_text)
                
                
                # recuperation du chemon vers le fichier contenant l'image
                chemin_image_i = os.path.join(dossier_camera,liste_images_sorted[s])
                
                if is_rotate:
                    image_i = rotate(imread(chemin_image_i, as_gray=True),270,resize=True)
                else:
                    image_i = imread(chemin_image_i, as_gray=True)
                
                # creation d'une figure correspondant a l'image
                fig = plt.figure(num = i)
                ax_net = fig.add_subplot(111)
                
                
                # affichage de l'image dans la figure
                imshow(image_i)
                
                for i in dict_position_t0:
                                    
                    x_i = positions['X{}'.format(i)][s]
                    y_i = positions['Y{}'.format(i)][s]
                    
                    # ax2.plot(y_i, x_i,'o', mec = 'gray', mfc='white')
                    
                    for j in dict_connexions[i]:
                        
                        x_j = positions['X{}'.format(j)][s]
                        y_j = positions['Y{}'.format(j)][s]
                        
                        
                        if '{}_{}'.format(i,j) in dict_suivi_def[s].keys():
                            
                            couleur = colors[all_def.index(dict_suivi_def[s]['{}_{}'.format(i,j)])]
                            
                            ax_net.plot([y_i, y_j],[x_i,x_j],'o-',mec = 'gray', mfc='white', color = dict_couleurs[dict_suivi_def[s]['{}_{}'.format(i,j)]])
                            
                ax_net.set_axis_off()
                
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                            hspace = 0, wspace = 0)
                
                plt.margins(0,0)
                
                ax_net.xaxis.set_major_locator(plt.NullLocator())
                ax_net.yaxis.set_major_locator(plt.NullLocator())
                    
                fig.colorbar(sm, ax = ax_net,label = "Strain (-)")
                
                fig.savefig(os.path.join(dossier_frames,liste_images_sorted[s].split('.')[0]+'.png'), bbox_inches = 'tight',pad_inches = 0,transparent=True)
                
                fig.clf()
            
            # chemin_gif = os.path.join(dossier_output,"net_strain.gif")
            # make_gif(dossier_frames, chemin_gif)
            
            my_bar.empty()
            st.markdown("Gif exported.")

            chemin_mp4 = os.path.join(dossier_output,"net_strain.mp4")
            
            fps = 30
            os.system("ffmpeg -y -framerate {} -pattern_type glob -i '{}' -c:v libx264 -vf '{}' -pix_fmt yuv420p {}".format(fps,dossier_frames+os.path.sep+'*.png', "pad=ceil(iw/2)*2:ceil(ih/2)*2", chemin_mp4))
            
            
            
            video_file = open(chemin_mp4, 'rb')
            video_bytes = video_file.read()
            
            st.video(video_bytes)
            
        # rain(emoji = "🐟", font_size = 64, falling_speed=4, animation_length=2)



















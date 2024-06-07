# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 09:44:29 2022

@author: llegue
"""

import os
import template_matching
from skimage.io import imread
from skimage.transform import rotate
import matplotlib.pyplot as plt

import matplotlib as mpl
import tkinter
mpl.use("TkAgg")

#=============================================================================#
#                                                                             #
#                    Definition des parametres de suivi                       #
#                                                                             #
#=============================================================================#

# recuperation du dossier courant
dossier_courant = os.getcwd()

#-------------------- recuperation des infos streamlit -----------------------#

# lecture du fichier inp
with open("subprocess_input.inp") as f:
    lignes = f.readlines()

# recuperation des infos
infos = {}
for i in lignes:
    infos[i.split('\t')[0]] = i.split('\t')[1]

dossier_bertrand = infos['dossier_basler'].strip()

if infos['result'].strip() == 'False':
    plot = False
else:
    plot = True

if infos['tracking'].strip() == 'False':
    tracking = False
else:
    tracking = True

if infos['rotate'].strip() == 'False':
    rotation = False
else:
    rotation = True

n_targets = int(infos["n_targets"])

#------------------------ definition des chemins------------------------------#

dossier_images = "Camera_1"

chemin_dossier_image = os.path.join(dossier_bertrand,dossier_images)

dossier_output = "tracking_outputs"

if dossier_output not in dossier_bertrand:
    try:
        os.mkdir(os.path.join(dossier_bertrand,dossier_output))
    except:
        pass

chemin_dossier_output = os.path.join(dossier_bertrand,dossier_output)

dossier_export_frames = "frames"
if dossier_export_frames not in os.listdir(chemin_dossier_output):
    try:
        os.mkdir(os.path.join(chemin_dossier_output,dossier_export_frames))
    except:
        pass


chemin_dossier_export_frames = os.path.join(chemin_dossier_output,dossier_export_frames)


if tracking:
    dossier_tracking = "tracking"

    if dossier_tracking not in os.listdir(chemin_dossier_output):
        try:
            chemin_dossier_tracking = os.path.join(chemin_dossier_output,dossier_tracking)
            os.mkdir(chemin_dossier_tracking)
        except:
            chemin_dossier_tracking = os.path.join(chemin_dossier_output,dossier_tracking)
            pass
    else:
        chemin_dossier_tracking = os.path.join(chemin_dossier_output,dossier_tracking)
else:
    chemin_dossier_tracking = False


gif_tracking = os.path.join(chemin_dossier_output,"tracking.gif")

gif_plot = os.path.join(chemin_dossier_output,"result.gif")


liste_images = [x for x in os.listdir(chemin_dossier_image) if x != '.DS_Store']

Y = [ int(x.split('.')[0]) for x in liste_images]

liste_images_sorted = [x for _,x in sorted(zip(Y,liste_images))]

#=============================================================================#
#                                                                             #
#                            Traitement des données                           #
#                                                                             #
#=============================================================================#

noeud = template_matching.TargetTracking(chemin_dossier_image, n_targets, rotation = rotation)

noeud.target_tracking(save_track = chemin_dossier_tracking, output_txt = chemin_dossier_output)


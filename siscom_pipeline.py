# -*- coding: utf-8 -*-
"""SISCOM pipeline: Subtraction Ictal SPECT CO-registered to MRI.

Usage:
    python siscom_pipeline.py --data-dir data/ --out-dir figures/

Expected inputs in --data-dir (co-registered NIfTI volumes): crICTAL.nii, cINTERICTAL.nii, RM.nii

# **Pràctica 7 - Tècniques d'imatge multimodalitat en epilèpsia, Dra Aida Niñerola**
#### Aplicacions Mèdiques de l'Enginyeria I, Enginyeria Biomèdica
#### Sergi Marsol Torrent
#### Desembre 2023

____
## **Taula de continguts**

1. Instal·lacions i importacions
2. Visualització imatges
3. Màscara del cervell per SPECT i RM
4. Normalització en intensitat
5. Imatge diferència
6. Selecció de la zona epiletògena
7. Fusió de la zona epiletògena
8. Localització estructura anatòmica

____
## **1. Instal·lacions i importacions**

En aquest apartat s'instal·len i s'importen les llibreries necessàries. També es connecta el Google Drive on es guarden els fitxers i es fa una primera instància del directori de Drive on es troben els fitxers.
"""





#import used libraries
import nibabel as nib
import nilearn
from nilearn import plotting
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
import dipy
from dipy.segment.mask import median_otsu
from dipy.core.histeq import histeq
import os
from mni_to_atlas import AtlasBrowser

import argparse

#input/output folders
parser = argparse.ArgumentParser(description='SISCOM epileptogenic-zone localization')
parser.add_argument('--data-dir', default='data', help='folder with crICTAL.nii, cINTERICTAL.nii and RM.nii')
parser.add_argument('--out-dir', default='figures', help='folder where figures are written')
args = parser.parse_args()
path = args.data_dir
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

"""____
## **2. Visualització imatges**

En aquest apartat es mostren les imatges després del realiniament i el coregistre duts a terme durant la sessió pràctica. Les imatges també es guarden en format jpg en aquest apartat i els següents. Cal comentar que en aquest apartat i en alguns altres casos concrets les imatges es mostren també amb la funció bàsica de matplotlib.pyplot per a una millor observació.

**Avís:** totes les imatges que es mostren amb nilearn.plotting s'han centrat a les coordenades (x,y,z) = (33, 1, -75), que corresponen a la Zona Epileptògena trobada. S'ha fet directament així per poder comparar els resultats inicials amb els finals i per veure el procediment sencer d'una forma coherent. També s'ha fet així per a poder utilitzar aquestes imatges per al pòster.

### SPECT ictal
La imatge ja ha estat realineada amb la interictal i coregistrada sobre la RM.
"""

#create path to file
file_name = 'crICTAL.nii'
file_path = os.path.join(path, file_name)

#load ictal image
ictal = nib.nifti1.load(file_path)
ictal_data = ictal.get_fdata() #get image data
ictal_header = ictal.header #get image header

#plot and save image
ictal_plot = plotting.plot_img(ictal, cut_coords=(33, 1, -75), colorbar=True, title='SPECT ictal')
ictal_plot.savefig(os.path.join(out_dir, 'ictal.png'))
plotting.show()

"""També es fa un plot de la imatge en color i tallant de forma perpendicular a la z en z=118 (aquest valor s'explica més endavant):"""

#plot of ictal data at z=118 section
section = 118
plt.imshow(histeq(ictal_data[:,:,section].astype('float')).T, cmap='jet', origin='lower')
plt.colorbar()
plt.title('SPECT ictal')
plt.savefig(os.path.join(out_dir, 'ictal2.png')) #save the image
plt.show()

"""### SPECT interictal
Aquesta imatge només ha estat coregistrada sobre la RM.
"""

#create path to file
file_name = 'cINTERICTAL.nii'
file_path = os.path.join(path, file_name)

#load interictal image
interictal = nib.nifti1.load(file_path)
interictal_data = interictal.get_fdata() #get image data
interictal_header = interictal.header #get image header

#plot and save image
interictal_plot = plotting.plot_img(interictal, cut_coords=(33, 1, -75), colorbar=True, title='SPECT interictal')
interictal_plot.savefig(os.path.join(out_dir, 'interictal.png'))
plotting.show()

"""També es fa un plot de la imatge en color i tallant de forma perpendicular a la z en z=118 (aquest valor s'explica més endavant):"""

#plot of interictal data at z=118 section
section = 118
plt.imshow(histeq(interictal_data[:,:,section].astype('float')).T, cmap='jet', origin='lower')
plt.colorbar()
plt.title('SPECT interictal')
plt.savefig(os.path.join(out_dir, 'interictal2.png')) #save image
plt.show()

"""### RM
Aquesta imatge no ha passat per cap mena de processament previ.
"""

#create path to file
file_name = 'RM.nii'
file_path = os.path.join(path, file_name)

#load RM image
rm = nib.nifti1.load(file_path)
rm_data = rm.get_fdata() #get image data
rm_header = rm.header #get image header

#plot and save image
rm_plot = plotting.plot_img(rm, cut_coords=(33, 1, -75), colorbar=True, title='MRI')
rm_plot.savefig(os.path.join(out_dir, 'rm.png'))
plotting.show()

"""En el cas de la RM s'ha fet el plot també en blanc i negre des de cadascun dels eixos, en lloc de només perpendicularment a l'eix z. Es pot veure que s'han usat els talls als índexs x=72, y=104 i z=118, i que això dona el mateix resultat que a la imatge anterior. Això és perquè aquests índexs de la matriu corresponen a les posicions anatòmiques x=33, y=1 i z=-75, que són les que es mostren a totes les imatges (i que contenen la Zona Epileptògena). En les imatges anteriors i posteriors també s'utilitzarà z=118 quan fem referència als índexs de la matriu per a poder veure aquesta mateixa zona."""

#plots of RM data at  x=72, y=104 i z=118 sections
plt.figure("RM")
plt.subplot(1,3,1).set_axis_off() #subplot all images in a same row
plt.imshow(histeq(rm_data[:,104,:].astype('float')).T, cmap='gray', origin='lower')
plt.subplot(1,3,2).set_axis_off()
plt.imshow(histeq(rm_data[72,:,:].astype('float')).T, cmap='gray', origin='lower')
plt.title('MRI')
plt.subplot(1,3,3).set_axis_off()
plt.imshow(histeq(rm_data[:,:,118].astype('float')).T, cmap='gray', origin='lower')
plt.savefig(os.path.join(out_dir, 'rm2.png')) #save image
plt.show()

"""____
## **3. Màscara del cervell per SPECT i RM**

La màscara es calcula a partir de la RM utilitzant el mètode d'Otsu (de la llibreria dipy). S'ha escollit aquest mètode perquè sembla ser el més adequat per a triar les zones d'interés a partir de la imatge estructural de RM. També es podria haver fet a partir dels SPECTs o amb una funció de la llibreria nilearn).
"""

rm_data_masked, mask = median_otsu(rm_data) #calculate mask and masked RM data

#plot of ictal data at z=118 section
section = 118
plt.figure("Brain mask")
plt.subplot(1,3,1).set_axis_off() #subplot in a same row
plt.imshow(histeq(rm_data[:,:,section].astype('float')).T, cmap='gray', origin='lower')
plt.title('MRI')
plt.subplot(1,3,2).set_axis_off()
plt.imshow(histeq(mask[:,:,section].astype('float')).T, cmap='gray', origin='lower')
plt.title('Mask')
plt.subplot(1,3,3).set_axis_off()
plt.imshow(histeq(rm_data_masked[:,:,section].astype('float')).T, cmap='gray', origin='lower')
plt.title('MRI with mask')
plt.savefig(os.path.join(out_dir, 'rm_mask.png')) #save image
plt.show()

"""En les imatges mostrades a dalt, es pot observar la RM a la meitat del cervell sense la màscara, la màscara, i el cervell amb la màscara aplicada, en aquest ordre. Es pot observar com la màscara elimina el fons i el soroll de bona part de la imatge.

A continuació, s'aplica la màscara trobada a les 3 imatges (ictal, interictal i RM) multiplicant els arrays que conformen la imatge i la màscara directament. Això simplement dona valor 0 a les zones fora de la màscara i deixa els valors que tenien a les zones incloses a la màscara.

### Màscara SPECT Ictal
"""

#apply mask to ictal image with product
ictal_data_masked = ictal_data*mask

#transform to nifti image
ictal_masked = nib.Nifti1Image(ictal_data_masked, ictal.affine)

#plot and save image
ictal_mask_plot = plotting.plot_img(ictal_masked, cut_coords=(33, 1, -75), colorbar=True, title='SPECT ictal masked')
ictal_mask_plot.savefig(os.path.join(out_dir, 'ictal_masked.png'))
plotting.show()

"""### Màscara SPECT Interictal"""

#apply mask to interictal image with product
interictal_data_masked = interictal_data*mask

#transform to nifti image
interictal_masked = nib.Nifti1Image(interictal_data_masked, interictal.affine)

#plot and save image
interictal_masked_plot = plotting.plot_img(interictal_masked, cut_coords=(33, 1, -75), colorbar=True, title='SPECT interictal masked')
interictal_masked_plot.savefig(os.path.join(out_dir, 'interictal_masked.png'))
plotting.show()

#apply mask to interictal image with product
rm_data_masked = rm_data*mask

#transform to nifti image
rm_masked = nib.Nifti1Image(rm_data_masked, rm.affine)

#plot and save image
rm_masked_plot = plotting.plot_img(rm_masked, cut_coords=(33, 1, -75), colorbar=True, title='MRI masked')
rm_masked_plot.savefig(os.path.join(out_dir, 'rm_masked.png'))
plotting.show()

"""____
## **4. Normalització en intensitat**

Primer de tot, es mostra l'histograma de la imatge sense normalitzar:
"""

#1D array of the image data
data_ictal_1D = ictal_data_masked.flatten()
data_interictal_1D = interictal_data_masked.flatten()

#plot and save histogram
plt.hist(data_ictal_1D, bins = 500) #500 bins for correct representation
plt.hist(data_interictal_1D, bins = 500)
plt.xlabel('Intensity')
plt.ylabel('Counts')
plt.title('Histogram of not normalized SPECT images')
plt.legend(["Ictal","Interictal"])
plt.ylim(0,125000)
plt.xlim(-10,200)
plt.savefig(os.path.join(out_dir, 'histogram_NOT_normalized.png')) #save image
plt.show()

"""Algunes opcions per normalitzar les dades són:
1. **Mean normalization** - s'inclou a Zscore normalization.
2. **Std normalization** - s'inclou a Zscore normalization.
3. **Zscore normalization** - l'opció utilitzada, consisteix en restar la mitjana i dividir el resultat entre la desviació estàndard. S'aconsegueix tenir una nova mitjana de 0 i una desviació estàndard de 1.
4. **CDF (Cumulative Distribution Factor) normalization** - consisteix a dividir la suma acumulativa entre la suma en cada valor.
5. **Min-max normalization** - es calcula un factor de normalització que és la resta del valor màxim menys el mínim i es divideixen els vòxels entre ell, de manera que el resultat es troba entre 0 i 1.
6. **Log normalization** - consisteix a prendre el logaritme natural dels valors.

Algunes d'aquestes opcions es mostren a continuació (algunes amb correcte funcionament i algunes no):

**Normalització amb Z-score**
"""

#Zscore normalization: subtract mean and divide by standard deviation
normalized_z_ictal_data = (ictal_data_masked - np.mean(ictal_data_masked))/ np.std(ictal_data_masked)
normalized_z_interictal_data = (interictal_data_masked - np.mean(interictal_data_masked))/ np.std(interictal_data_masked)

#1D array of the image data
data_ictal_1D = normalized_z_ictal_data.flatten()
data_interictal_1D = normalized_z_interictal_data.flatten()

#plot and save histogram
plt.hist(data_ictal_1D, bins = 500)
plt.hist(data_interictal_1D, bins = 500)
plt.xlabel('Intensity')
plt.ylabel('Counts')
plt.title('Histogram of normalized SPECT images')
plt.legend(["Ictal","Interictal"])
plt.ylim(0,125000)
plt.xlim(-1,5)
plt.savefig(os.path.join(out_dir, 'histogram_z_normalized.png'))
plt.show()

"""L'histograma mostra diferències clares entre les dades ictals i interictals. Aquest tipus de normalització és el que acabarem utilitzant (amb alguna millora posterior).

**Normalització amb CDF**
"""

#CDF normalization: divide cumulative sum between sum
normalized_cdf_ictal_data = np.cumsum(ictal_data_masked) / np.sum(ictal_data_masked)
normalized_cdf_interictal_data = np.cumsum(interictal_data_masked) / np.sum(interictal_data_masked)

#1D array of the image data
data_ictal_1D = normalized_cdf_ictal_data.flatten()
data_interictal_1D = normalized_cdf_interictal_data.flatten()

#plo and save histogram
plt.hist(data_ictal_1D, bins = 500)
plt.hist(data_interictal_1D, bins = 500)
plt.xlabel('Intensity')
plt.ylabel('Counts')
plt.title('Histogram of normalized SPECT images')
plt.legend(["Ictal","Interictal"])
plt.ylim(0,60000)
plt.xlim(0,1)
plt.savefig(os.path.join(out_dir, 'histogram_cdf_normalized.png'))
plt.show()

"""En aquest cas, l'histograma obtingut no dona els resultats esperats i es pot descartar aquest mètode. Sembla que alguna part del càlcul no s'ha realitzat correctament, així que es deixa com a exemple erroni però no s'utilitza més enllà.

**Normalització min-max**
"""

#Min-max normalization
#compute factor: maximum value - minimum value
d_ictal = np.max(ictal_data_masked) - np.min(ictal_data_masked)
d_interictal = np.max(interictal_data_masked) - np.min(interictal_data_masked)
#divide by the factor
normalized_d_ictal_data = ictal_data_masked/d_ictal
normalized_d_interictal_data = interictal_data_masked/d_interictal

#1D array of the image data
data_ictal_1D = normalized_d_ictal_data.flatten()
data_interictal_1D = normalized_d_interictal_data.flatten()

#plot and save histogram
plt.hist(data_ictal_1D, bins = 500)
plt.hist(data_interictal_1D, bins = 500)
plt.xlabel('Intensity')
plt.ylabel('Counts')
plt.title('Histogram of normalized SPECT images')
plt.legend(["Ictal","Interictal"])
plt.ylim(0,120000)
plt.xlim(-0.1,0.8)
plt.savefig(os.path.join(out_dir, 'histogram_d_normalized.png'))
plt.show()

"""D'aquestes opcions s'ha escollit la normalització Z-score pel bon resultat que s'observa a les dades, encara que la normalització min-max també dona bon resultat. A més de la normalització, altres processaments com filtratge de la senyal poden ser afegits. En aquest cas s'ha optat per una millora de la normalització Z-score i per un filtrat.

**Normalització Z-score eliminant els valors nuls**

Primer, s'ha decidit treure els valors 0 de la imatge per a calcular la mitja i la desviació estàndard amb que es fa la normalització. Això s'ha fet perquè aquests valors 0 afecten molt en el resultat (ja que n'hi ha molts) i provenen del background. Per tant, els eliminem per a aquest càlcul, encara que segueixen estant a la imatge.
"""

#find the ictal and interictal data that is different from 0
#flatten the filtered data
data_ictal_1D = ictal_data_masked.flatten()
data_interictal_1D = interictal_data_masked.flatten()
nonzero_ictal = data_ictal_1D[data_ictal_1D!=0] #rule out the 0 data
nonzero_interictal = data_interictal_1D[data_interictal_1D!=0] #rule out the 0 data

#Zscore normalization after filtering
#the mean and std are computed leaving out the 0s to avoid a deviation due to them
normalized_z_ictal_data = (ictal_data_masked - np.mean(nonzero_ictal)) / np.std(nonzero_ictal)
normalized_z_interictal_data = (interictal_data_masked - np.mean(nonzero_interictal)) / np.std(nonzero_interictal)

#1D array of the image data
data_z_ictal_1D = normalized_z_ictal_data.flatten()
data_z_interictal_1D = normalized_z_interictal_data.flatten()

#plot and save histogram
plt.hist(data_z_ictal_1D, bins = 500)
plt.hist(data_z_interictal_1D, bins = 500)
plt.xlabel('Intensity')
plt.ylabel('Counts')
plt.title('Histogram of normalized SPECT images')
plt.legend(["Ictal","Interictal"])
plt.ylim(0,125000)
plt.xlim(-2,3)
plt.savefig(os.path.join(out_dir, 'histogram_z_normalized2.png'))
plt.show()

"""**Normalització Z-score amb filtrat de Wiener**

En aquest cas, s'ha aplicat un filtre de Wiener per eliminar el soroll no desitjat de la imatge, ja que s'ha vist que millorava el resultat final. A més, s'ha mantingut el fet de treure els valors 0 a l'hora de calcular la mitjana i la desviació estàndard.
"""

#find the ictal and interictal data that is different from 0
#flatten the filtered data
data_ictal_1D = ictal_data_masked.flatten()
data_interictal_1D = interictal_data_masked.flatten()
nonzero_ictal = data_ictal_1D[data_ictal_1D!=0] #rule out the 0 data
nonzero_interictal = data_interictal_1D[data_interictal_1D!=0] #rule out the 0 data

#Filtering with Wiener filter
filtered_ictal_data = wiener(ictal_data_masked)
filtered_interictal_data = wiener(interictal_data_masked)

#Zscore normalization after filtering
#the mean and std are computed leaving out the 0s to avoid a deviation due to them
normalized_f_ictal_data = (filtered_ictal_data - np.mean(nonzero_ictal)) / np.std(nonzero_ictal)
normalized_f_interictal_data = (filtered_interictal_data - np.mean(nonzero_interictal)) / np.std(nonzero_interictal)

#1D array of the image data
data_ictal_1D = normalized_f_ictal_data.flatten()
data_interictal_1D = normalized_f_interictal_data.flatten()

#plot and save histogram
plt.hist(data_ictal_1D, bins = 500)
plt.hist(data_interictal_1D, bins = 500)
plt.xlabel('Intensity')
plt.ylabel('Counts')
plt.title('Histogram of normalized SPECT images')
plt.legend(["Ictal","Interictal"])
plt.ylim(0,125000)
plt.xlim(-2,3)
plt.savefig(os.path.join(out_dir, 'histogram_f_normalized.png'))
plt.show()

"""Aquest histograma és el que representa les dades que finalment s'han utilitzat per als següents apartats. Cal comentar que amb unes dades normalitzades sense eliminar els 0s i sense filtrar els resultats són semblants, però pel que he trobat els resultats obtinguts amb aquestes dades semblen lleugerament més precisos.

____

## **5. Imatge diferència**

Simplement, consisteix en la substracció de la imatge ictal (quan hi ha una crisi) menys la imatge interictal, ambdues després del processament anterior. Així, s'observen les zones més actives en la crisi i que no ho estaven en estat interictal. També es repeteix la normalització per Z-score per mantenir les dades en el rang d'interés, encara que no sigui necessària.
"""

#subtract ictal - interictal
diff_data = normalized_f_ictal_data - normalized_f_interictal_data
diff_z_data = (diff_data - np.mean(diff_data))/ np.std(diff_data) #Zscore normalization for the data again (to keep the values in the correct range)
diff = nib.Nifti1Image(diff_z_data, ictal.affine, ictal_header) #transform to nifti image format

#plot and save the image
diff_plot = plotting.plot_img(diff, cut_coords=(33, 1, -75), colorbar=True, title='Difference between ictal and interictal')
diff_plot.savefig(os.path.join(out_dir, 'diff.png'))
plotting.show()

"""____
## **6. Selecció de la zona epiletògena**

Cal comentar que per trobar exactament aquesta zona epileptògena s'ha dut a terme el següent càlcul, en el qual es busca el valor màxim dins la imatge diferència:
"""

max_idx = np.argmax(diff_z_data) #index with the max value (diff image after thresholding)
x, y, z = np.unravel_index(max_idx, diff_z_data.shape) #x,y,z indexs of the matrix
print("Els índexs de la matriu del punt màxim són: ", x,y,z)

"""Com s'observa, el punt màxim de la ZE és el que correspon als índexs (72, 104, 118) de la matriu de la imatge. Ja s'ha explicat al principi que aquests índexs corresponen als valors x=33, y=1 i z=-75 a nivell anatòmic.

Es pot comprovar que tots els índexs amb valors elevats (per sobre de 4 en aquest cas) es troben al voltant d'aquest punt. El valor de 4 s'ha escollit arbitràriament per fer la comprovació.
"""

indices_over_4 = np.argwhere(diff_z_data > 4) #indices > 4

#list of (x, y, z) indices where values are over 7
indices_list = [(x, y, z) for x, y, z in indices_over_4]

print("Índexs amb valor > 4:")
print(indices_list)

"""Com es pot observar en aquest resultat, tots els punts de valor màxim es troben al voltant dels índexs (72, 104, 118), el que confirma que es troba al centre de la Zona Epileptògena. Això també confirma que no hi ha cap altre focus epileptògen de la mateixa importància en tot el cervell.

Es procedeix a la selecció d'aquesta zona, que es selecciona amb una threshold definida per eliminar els valors inferiors a ella. Aquesta s'ha trobat provant diferents valors en el loop que es mostra a continuació. Cal tenir en compte que aquest loop s'ha fet mirant directament la secció de z=118, que, com s'ha comentat, correspon a la Zona Epileptògena i permet observar els resultats perfectament. Tot i això, també s'ha realitzat en altres valors de z per assegurar que no s'estava ignorant cap altra zona il·luminada.
"""

diff_z_data_flat = diff_z_data.flatten()

#define different thr values
thresholds = np.arange(1.5, 4.5, 0.5).tolist()

section = 118 #plots at z=118 section
fig, axs = plt.subplots(2, 3) #initialize the subplots (2 rows and 3 columns)

#Iterate over different threshold values to find the best ZE
for i, thr in enumerate(thresholds):
  #iterate to identify the ZE (where values > thr) and set other values to 0
  diff_z_data_flat_thr = np.array([0 if diff_z_data_flat[j] < thr else val for j, val in enumerate(diff_z_data_flat)])
  diff_data_i = diff_z_data_flat_thr.reshape(diff_z_data.shape) #reshape the image data for the given threshold

  row = i // 3  #calculate row index
  col = i % 3   #calculate column index
  axs[row, col].imshow(histeq(diff_data_i[:,:,section].astype('float')).T, cmap='jet', origin='lower') #plot the ZE fir each sublot
  axs[row, col].set_title(f'thr={thr}')  #titles for each subplot
  axs[row, col].set_axis_off()  #turn off axis for each subplot

#plot and save image
plt.tight_layout() #use a tight layout
plt.savefig(os.path.join(out_dir, 'ZE_thresholds.png'))
plt.show()

"""Tenint en compte els resultats trobats, s'ha optat per una threhsold de valor thr=3, que permet veure una Zona Epileptògena suficientment gran per identificar-la i alhora veure'n només un únic focus."""

#Define the threshold for selecting the ZE
thr = 3 #obtained in the last code section

#iterate to identify the ZE (where values > thr) and set other values to 0
diff_z_data_flat_3 = np.array([0 if diff_z_data_flat[j] < thr else val for j, val in enumerate(diff_z_data_flat)])
ZE_data = diff_z_data_flat_3.reshape(diff_z_data.shape)
ZE = nib.Nifti1Image(ZE_data, interictal.affine, interictal_header) #transform to Nifti image

#plot and save image
ZE_plot = plotting.plot_img(ZE, cut_coords=(33, 1, -75), colorbar=True, title='Epileptogenic Zone')
ZE_plot.savefig(os.path.join(out_dir, 'ZE.png'))
plotting.show()

"""Amb la thr=3 s'ha identificat la zona del gràfic de dalt com la Zona Epileptògena, ja que és a la que la imatge diferència té la major intensitat. Com s'observa, el punt anatòmic on es centra la ZE és el x=33, y=1 i z=-75.

____
## **7. Fusió de la zona epiletògena**

Per últim, la ZE s'ha mostrat en imatges a sobre de les imatges estructurals proporcionades per RM, de manera que es puguin localitzar aquestes zones al cervell.
"""

#plot the ROI of both the ZE and the masked RM (fusion of both)
fusion_masked = plotting.plot_roi(ZE, rm_masked, cut_coords=(33, 1, -75), title='Epileptogenic Zone over MRI')
fusion_masked.savefig(os.path.join(out_dir, 'fusion.png')) #save the image
plotting.show()

"""Podem veure que amb thr=3 la ZE està molt ben localitzada en un punt concret, però si volem veure'n la influència a una zona una mica més extensa podem reduir la threshold a thr=2, per exemple. El resultat és el següent:"""

#Define the threshold for selecting the ZE
thr = 2 #for a more extense ZE

#iterate to identify the ZE (where values > thr) and set other values to 0
diff_z_data_flat_2 = np.array([0 if diff_z_data_flat[j] < thr else val for j, val in enumerate(diff_z_data_flat)])
ZE_data2 = diff_z_data_flat_2.reshape(diff_z_data.shape)
ZE2 = nib.Nifti1Image(ZE_data2, interictal.affine, interictal_header) #transform to Nifti image

#plot the ROI of both the ZE and the masked RM (fusion of both)
fusion_masked = plotting.plot_roi(ZE2, rm_masked, cut_coords=(33, 1, -75), title='Extense Epileptogenic Zone over MRI')
fusion_masked.savefig(os.path.join(out_dir, 'fusion2.png')) #save the image
plotting.show()

"""Com s'observa, el focus epilptògen segueix estant a les coordenades indicades, però es mostren àrees d'alta activitat a zones properes també.

____
## **8. Localització estructura anatòmica**

Aquest últim pas no s'ha aconseguit reproduir de forma correcta. En primer lloc s'ha intentat prenent (33, 1,  -75) com a coordenades MNI:
"""

# Instantiate the AtlasBrowser class and specify the atlas to use
atlas = AtlasBrowser("AAL")

# Provide MNI coordinates as an (n x 3) array
coordinates = np.array([[33, 1,  -75]])

# Find the brain regions at the MNI coordinates (plotting is optional)
regions = atlas.find_regions(coordinates, plot=True)

"""Com s'observa, la ZE es situa fora del cervell, així que no està funcionant correctament.

També s'ha intentat usar aquestes coordenades com a coordenades del vòxels i transformar-les a coordenades MNI:
"""

mni_coords = nilearn.image.coord_transform(33, 1, -75, diff.affine)
print("MNI coordinates:", mni_coords)

# Instantiate the AtlasBrowser class and specify the atlas to use
atlas = AtlasBrowser("AAL")

# Provide MNI coordinates as an (n x 3) array
coordinates = np.array([[65.9021224975586, 95.13411140441895,  -240.14212036132812]])

# Find the brain regions at the MNI coordinates (plotting is optional)
regions = atlas.find_regions(coordinates, plot=True)

"""Com s'observa, dona error de dimensions.

Per últim, s'ha intentat fer el càlcul a partir dels índexs de la matriu i transformant-los a coordenades MNI:
"""

mni_coords = nilearn.image.coord_transform(72, 104, 118, diff.affine)
print("MNI coordinates:", mni_coords)

# Instantiate the AtlasBrowser class and specify the atlas to use
atlas = AtlasBrowser("AAL")

# Provide MNI coordinates as an (n x 3) array
coordinates = np.array([[32.386497497558594, 2.4349365234375, -74.28274536132812]])

# Find the brain regions at the MNI coordinates (plotting is optional)
regions = atlas.find_regions(coordinates, plot=True)

"""Aquí també es situa la ZE fora del cervell, de manera que és erroni.

Així, s'ha decidit mostrar un exemple en el qual el codi funciona correctament i dona lloc a una zona prou propera a la ZE (fet a vista). Com s'observa, sembla que es tracta d'un punt proper al pol temporal dret superior i la ZE podria estar per aquesta zona. Aquesta és la figura que s'usa al pòster.
"""

# Instantiate the AtlasBrowser class and specify the atlas to use
atlas = AtlasBrowser("AAL")

# Provide MNI coordinates as an (n x 3) array
coordinates = np.array([[45, 15,  -25]])

# Find the brain regions at the MNI coordinates (plotting is optional)
regions = atlas.find_regions(coordinates, plot=True)

"""Com es comenta al pòster, aquests errors segurament es deuen a no haver normalitzat les imatges correctament a l'espai estàndard per trobar les MNI correctes."""
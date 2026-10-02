# -*- coding: utf-8 -*-
"""SISCOM pipeline: Subtraction Ictal SPECT CO-registered to MRI.

Usage:
    python siscom_pipeline.py --data-dir data --out-dir figures

Expected inputs in --data-dir (co-registered NIfTI volumes): crICTAL.nii, cINTERICTAL.nii, RM.nii

Multimodal imaging for epilepsy: SPECT/MRI epileptogenic-zone localization
Author: Sergi Marsol Torrent (December 2023), supervised by Dr. Aida Niñerola

Originally written as a Google Colab notebook; the notebook's markdown cells are
kept below as (English-translated) string blocks between the code sections.

____
## Table of contents

1. Installs and imports
2. Image visualization
3. Brain mask for SPECT and MRI
4. Intensity normalization
5. Difference image
6. Epileptogenic-zone selection
7. Epileptogenic-zone fusion
8. Anatomical-structure localization

____
## 1. Installs and imports

This section imports the required libraries. (In the original Colab notebook it also
mounted Google Drive, where the input files were stored; here the input folder is
given with --data-dir.)
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
## 2. Image visualization

This section shows the images after the realignment and co-registration performed (in SPM12)
during the lab session. Images are saved to disk in this section and the following ones. In
this section, and in a few other specific cases, images are also shown with plain
matplotlib.pyplot for easier inspection.

**Note:** every image shown with nilearn.plotting is centred on (x, y, z) = (33, 1, -75), the
coordinates of the Epileptogenic Zone (EZ) that was eventually found. This makes it possible to
compare the initial and final results and to follow the whole procedure consistently; the same
views were used for the poster.

### Ictal SPECT
This image has already been realigned to the interictal SPECT and co-registered to the MRI.
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

"""The image is also plotted in colour as an axial slice perpendicular to z at index z=118 (this value is explained later):"""

#plot of ictal data at z=118 section
section = 118
plt.imshow(histeq(ictal_data[:,:,section].astype('float')).T, cmap='jet', origin='lower')
plt.colorbar()
plt.title('SPECT ictal')
plt.savefig(os.path.join(out_dir, 'ictal2.png')) #save the image
plt.show()

"""### Interictal SPECT
This image has only been co-registered to the MRI.
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

"""The image is also plotted in colour as an axial slice perpendicular to z at index z=118 (this value is explained later):"""

#plot of interictal data at z=118 section
section = 118
plt.imshow(histeq(interictal_data[:,:,section].astype('float')).T, cmap='jet', origin='lower')
plt.colorbar()
plt.title('SPECT interictal')
plt.savefig(os.path.join(out_dir, 'interictal2.png')) #save image
plt.show()

"""### MRI
This image has not undergone any previous processing.
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

"""For the MRI, a greyscale plot is also made along each of the three axes, instead of only perpendicular to z. The slices at array indices x=72, y=104 and z=118 give the same view as the previous image, because these array indices correspond to the anatomical positions x=33, y=1, z=-75 shown in every figure (which contain the Epileptogenic Zone). In the previous and following figures, z=118 is also used whenever array indices are referred to, in order to show this same region."""

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
## 3. Brain mask for SPECT and MRI

The mask is computed from the MRI with Otsu's method (dipy's median_otsu). This method was chosen
because it seemed the most suitable for selecting the regions of interest from the structural MRI.
It could also have been computed from the SPECTs, or with a nilearn function.
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

"""The figure above shows, in order, the MRI at mid-brain without the mask, the mask, and the brain with the mask applied. The mask removes the background and noise from a large part of the image.

Next, the mask is applied to the 3 images (ictal, interictal and MRI) by element-wise multiplication of each image array with the mask. This sets voxels outside the mask to 0 and keeps the original values inside it.

### Ictal SPECT mask
"""

#apply mask to ictal image with product
ictal_data_masked = ictal_data*mask

#transform to nifti image
ictal_masked = nib.Nifti1Image(ictal_data_masked, ictal.affine)

#plot and save image
ictal_mask_plot = plotting.plot_img(ictal_masked, cut_coords=(33, 1, -75), colorbar=True, title='SPECT ictal masked')
ictal_mask_plot.savefig(os.path.join(out_dir, 'ictal_masked.png'))
plotting.show()

"""### Interictal SPECT mask"""

#apply mask to interictal image with product
interictal_data_masked = interictal_data*mask

#transform to nifti image
interictal_masked = nib.Nifti1Image(interictal_data_masked, interictal.affine)

#plot and save image
interictal_masked_plot = plotting.plot_img(interictal_masked, cut_coords=(33, 1, -75), colorbar=True, title='SPECT interictal masked')
interictal_masked_plot.savefig(os.path.join(out_dir, 'interictal_masked.png'))
plotting.show()

#apply mask to MRI image with product
rm_data_masked = rm_data*mask

#transform to nifti image
rm_masked = nib.Nifti1Image(rm_data_masked, rm.affine)

#plot and save image
rm_masked_plot = plotting.plot_img(rm_masked, cut_coords=(33, 1, -75), colorbar=True, title='MRI masked')
rm_masked_plot.savefig(os.path.join(out_dir, 'rm_masked.png'))
plotting.show()

"""____
## 4. Intensity normalization

First, the histogram of the non-normalized images is shown:
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

"""Some options for normalizing the data are:
1. **Mean normalization** - included in Z-score normalization.
2. **Std normalization** - included in Z-score normalization.
3. **Z-score normalization** - the option used: subtract the mean and divide by the standard deviation, giving a new mean of 0 and a standard deviation of 1.
4. **CDF (Cumulative Distribution Factor) normalization** - divide the cumulative sum by the total sum at each value.
5. **Min-max normalization** - compute a normalization factor equal to the maximum minus the minimum value and divide the voxels by it, so that the result lies between 0 and 1.
6. **Log normalization** - take the natural logarithm of the values.

Some of these options are shown below (some work correctly and some do not):

**Z-score normalization**
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

"""The histogram shows clear differences between the ictal and interictal data. This is the normalization that is eventually used (with some later improvements).

**CDF normalization**
"""

#CDF normalization: divide cumulative sum between sum
normalized_cdf_ictal_data = np.cumsum(ictal_data_masked) / np.sum(ictal_data_masked)
normalized_cdf_interictal_data = np.cumsum(interictal_data_masked) / np.sum(interictal_data_masked)

#1D array of the image data
data_ictal_1D = normalized_cdf_ictal_data.flatten()
data_interictal_1D = normalized_cdf_interictal_data.flatten()

#plot and save histogram
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

"""Here the resulting histogram does not give the expected result, so this method is discarded. Part of the computation appears not to be correct; it is kept as a (failed) example but not used further.

**Min-max normalization**
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

"""Of these options, Z-score normalization was chosen because of the good result observed on the data, although min-max normalization also works well. Besides normalization, other processing such as signal filtering can be added. Here, an improved Z-score normalization plus a filtering step was chosen.

**Z-score normalization excluding zero values**

First, the zero-valued voxels are excluded when computing the mean and standard deviation used for normalization. These zeros strongly bias the result (there are many of them) and come from the background, so they are left out of this computation, although they remain in the image.
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

"""**Z-score normalization with Wiener filtering**

Here a Wiener filter is applied to remove unwanted noise from the image, since it was found to improve the final result. Zero values are still excluded when computing the mean and standard deviation.
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

"""This histogram represents the data finally used in the following sections. With data normalized without removing the zeros and without filtering, the results are similar, but in my tests the results obtained with this version looked slightly more precise.

____

## 5. Difference image

This is simply the ictal image (acquired during a seizure) minus the interictal image, both after the processing above. It highlights the regions that are more active during the seizure than in the interictal state. Z-score normalization is applied again to keep the values in the range of interest, although it is not strictly necessary.
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
## 6. Epileptogenic-zone selection

To locate the epileptogenic zone exactly, the following computation searches for the maximum value in the difference image:
"""

max_idx = np.argmax(diff_z_data) #index with the max value of the (z-scored) difference image
x, y, z = np.unravel_index(max_idx, diff_z_data.shape) #x,y,z indexs of the matrix
print("Array indices of the maximum voxel: ", x,y,z)

"""The maximum of the EZ corresponds to array indices (72, 104, 118). As explained at the beginning, these indices correspond to anatomical coordinates x=33, y=1, z=-75.

We can check that all indices with high values (above 4 here) lie around this point. The value 4 was chosen arbitrarily for this check.
"""

indices_over_4 = np.argwhere(diff_z_data > 4) #indices > 4

#list of (x, y, z) indices where values are over 4
indices_list = [(x, y, z) for x, y, z in indices_over_4]

print("Indices with value > 4:")
print(indices_list)

"""As this output shows, all the highest-valued voxels lie around indices (72, 104, 118), confirming that this is the centre of the Epileptogenic Zone. It also indicates that there is no other epileptogenic focus of similar magnitude elsewhere in the brain.

The zone is then selected with a threshold that removes all values below it. The threshold was found by trying different values in the loop below. The loop looks directly at slice z=118, which, as discussed, contains the Epileptogenic Zone and shows the results clearly; it was also run at other z values to make sure no other bright region was being missed.
"""

diff_z_data_flat = diff_z_data.flatten()

#define different thr values
thresholds = np.arange(1.5, 4.5, 0.5).tolist()

section = 118 #plots at z=118 section
fig, axs = plt.subplots(2, 3) #initialize the subplots (2 rows and 3 columns)

#Iterate over different threshold values to find the best EZ (ZE = "zona epileptògena")
for i, thr in enumerate(thresholds):
  #iterate to identify the ZE (where values > thr) and set other values to 0
  diff_z_data_flat_thr = np.array([0 if diff_z_data_flat[j] < thr else val for j, val in enumerate(diff_z_data_flat)])
  diff_data_i = diff_z_data_flat_thr.reshape(diff_z_data.shape) #reshape the image data for the given threshold

  row = i // 3  #calculate row index
  col = i % 3   #calculate column index
  axs[row, col].imshow(histeq(diff_data_i[:,:,section].astype('float')).T, cmap='jet', origin='lower') #plot the EZ for each subplot
  axs[row, col].set_title(f'thr={thr}')  #titles for each subplot
  axs[row, col].set_axis_off()  #turn off axis for each subplot

#plot and save image
plt.tight_layout() #use a tight layout
plt.savefig(os.path.join(out_dir, 'ZE_thresholds.png'))
plt.show()

"""Given these results, a threshold of thr=3 was chosen: it leaves an Epileptogenic Zone large enough to identify while showing a single focus."""

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

"""With thr=3, the region in the figure above is identified as the Epileptogenic Zone, since it is where the difference image has the highest intensity. The anatomical point on which the EZ is centred is x=33, y=1, z=-75.

____
## 7. Epileptogenic-zone fusion

Finally, the EZ is overlaid on the structural MRI so that it can be located in the brain.
"""

#plot the ROI of both the ZE and the masked RM (fusion of both)
fusion_masked = plotting.plot_roi(ZE, rm_masked, cut_coords=(33, 1, -75), title='Epileptogenic Zone over MRI')
fusion_masked.savefig(os.path.join(out_dir, 'fusion.png')) #save the image
plotting.show()

"""With thr=3 the EZ is tightly localized to a single spot; to see its influence over a somewhat larger area, the threshold can be lowered to, e.g., thr=2. The result is:"""

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

"""The epileptogenic focus remains at the same coordinates, but high-activity areas in neighbouring regions are now also shown.

____
## 8. Anatomical-structure localization

This last step could not be completed correctly. First, (33, 1, -75) was used directly as MNI coordinates:
"""

# Instantiate the AtlasBrowser class and specify the atlas to use
atlas = AtlasBrowser("AAL")

# Provide MNI coordinates as an (n x 3) array
coordinates = np.array([[33, 1,  -75]])

# Find the brain regions at the MNI coordinates (plotting is optional)
regions = atlas.find_regions(coordinates, plot=True)

"""The EZ falls outside the brain (the AAL lookup returns 'Undefined'), so this does not work correctly.

Next, these coordinates were treated as voxel coordinates and transformed to MNI coordinates:
"""

mni_coords = nilearn.image.coord_transform(33, 1, -75, diff.affine)
print("MNI coordinates:", mni_coords)

# Instantiate the AtlasBrowser class and specify the atlas to use
atlas = AtlasBrowser("AAL")

# Provide MNI coordinates as an (n x 3) array
coordinates = np.array([[65.9021224975586, 95.13411140441895,  -240.14212036132812]])

# Find the brain regions at the MNI coordinates (plotting is optional)
# In the original notebook this lookup failed with a dimension (IndexError) error because the
# point lies outside the atlas volume; the error is caught and reported here so that the script
# can run end-to-end.
try:
    regions = atlas.find_regions(coordinates, plot=True)
except IndexError as err:
    print("Atlas lookup failed (coordinates outside the atlas volume):", err)

"""This raises a dimension (out-of-bounds index) error.

Finally, the array indices were transformed to MNI coordinates:
"""

mni_coords = nilearn.image.coord_transform(72, 104, 118, diff.affine)
print("MNI coordinates:", mni_coords)

# Instantiate the AtlasBrowser class and specify the atlas to use
atlas = AtlasBrowser("AAL")

# Provide MNI coordinates as an (n x 3) array
coordinates = np.array([[32.386497497558594, 2.4349365234375, -74.28274536132812]])

# Find the brain regions at the MNI coordinates (plotting is optional)
regions = atlas.find_regions(coordinates, plot=True)

"""Here the EZ is again placed outside the brain, so this result is also wrong.

Therefore, an example is shown where the code works and gives a point reasonably close to the EZ (chosen by eye). It appears to be a point near the right superior temporal pole, so the EZ could lie around this region. This is the (illustrative) figure used in the poster.
"""

# Instantiate the AtlasBrowser class and specify the atlas to use
atlas = AtlasBrowser("AAL")

# Provide MNI coordinates as an (n x 3) array
coordinates = np.array([[45, 15,  -25]])

# Find the brain regions at the MNI coordinates (plotting is optional)
regions = atlas.find_regions(coordinates, plot=True)

"""As discussed in the poster, these errors are most likely because the images were not spatially normalized to standard (MNI) space, which is required to obtain correct MNI coordinates."""
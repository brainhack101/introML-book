---
redirect_from:
  - "/01/main-tutorial-intro-to-nilearn"
interact_link: content/01/MAIN_tutorial_intro_to_nilearn.ipynb
title: 'Intro to Nilearn'
prev_page:
  url: /intro
  title: 'Home'
next_page:
  url: /01/MAIN_tutorial_machine_learning_with_nilearn
  title: 'Machine Learning with Nilearn'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---



{:.input_area}
```python
%matplotlib inline
import nilearn
```


## [Understanding neuroimaging data](http://nilearn.github.io/manipulating_images/input_output.html)

### Text files: phenotype or behavior

Phenotypic or behavioral data are often provided as a text or CSV (Comma Separated Values) file. They can be loaded with the [pandas package](https://pandas.pydata.org/) but you may have to specify some options. Here, we'll specify the `sep` field, since our data is tab-delimited rather than comma-delimited.

For our dataset, let's load participant level information:



{:.input_area}
```python
import os
import pandas as pd

data_dir = '/home/emdupre/Desktop/MAIN_tutorial/'  # Update this path to match your local machine !
participants = 'participants.tsv'
phenotypic_data = pd.read_csv(os.path.join(data_dir, participants), sep='\t')
phenotypic_data.head()
```


### Nifti data

For volumetric data, nilearn works with data stored in the [Nifti structure](http://nipy.org/nibabel/nifti_images.html) (via the [nibabel package](http://nipy.org/nibabel/)).

The NifTi data structure (also used in Analyze files) is the standard way of sharing data in neuroimaging research. Three main components are:

  * data:	raw scans in form of a numpy array:  
    `data = img.get_data()`
  * affine:	returns the transformation matrix that maps from voxel indices of the `numpy` array to actual real-world     locations of the brain:  
    `affine = img.affine`
  * header:	low-level informations about the data (slice duration, etc.):  
  `header = img.header`

It is important to appreciate that the representation of MRI data we'll be using is a big 4D matrix representing (3D MRI + 1D for time), stored in a single Nifti file.

### Niimg-like objects

Nilearn functions take as input argument what we call "Niimg-like objects":

Niimg: A Niimg-like object can be one of the following:

  * A string with a file path to a Nifti image
  * A SpatialImage from `nibabel`, i.e., an object exposing the get_data() method and affine attribute, typically a Nifti1Image from `nibabel`.

Niimg-4D: Similarly, some functions require 4D Nifti-like data, which we call Niimgs or Niimg-4D. Accepted input arguments are:

  * A path to a 4D Nifti image
  * List of paths to 3D Nifti images
  * 4D Nifti-like object
  * List of 3D Nifti-like objects

**Note:** If you provide a sequence of Nifti images, all of them must have the same affine !

## [Manipulating and looking at data](http://nilearn.github.io/auto_examples/plot_nilearn_101.html#sphx-glr-auto-examples-plot-nilearn-101-py)

There is a whole section of the [Nilearn documentation](http://nilearn.github.io/plotting/index.html#plotting) on making pretty plots for neuroimaging data ! But let's start with a simple one.



{:.input_area}
```python
# Let's use a Nifti file that is shipped with nilearn
from nilearn import datasets

# Note that the variable MNI152_FILE_PATH is just a path to a Nifti file
print('Path to MNI152 template: {}'.format(datasets.MNI152_FILE_PATH))
```


In the above, MNI152_FILE_PATH is nothing more than a string with a path pointing to a nifti image. You can replace it with a string pointing to a file on your disk. Note that it should be a 3D volume, and not a 4D volume.



{:.input_area}
```python
from nilearn import plotting
plotting.plot_img(datasets.MNI152_FILE_PATH)
```


We can also directly manipulate these images using Nilearn ! As an example, let's try smoothing this image.



{:.input_area}
```python
from nilearn import image
smooth_anat_img = image.smooth_img(datasets.MNI152_FILE_PATH, fwhm=6)

# While we are giving a file name as input, the function returns
# an in-memory object:
print(smooth_anat_img)
```




{:.input_area}
```python
plotting.plot_img(smooth_anat_img)
```


We can then save this manipulated image from in-memory to disk as follows:



{:.input_area}
```python
smooth_anat_img.to_filename('smooth_anat_img.nii.gz')
os.getcwd()  # We'll' check our "current working directory" (cwd) to see where the file was saved
```


## [Visualizing neuroimaging volumes](https://nilearn.github.io/auto_examples/01_plotting/plot_visualization.html#visualization)

What if we want to view not a structural MRI image, but a functional one ?
No problem ! Let's try loading one:



{:.input_area}
```python
fmri_filename = 'downsampled_derivatives:fmriprep:sub-pixar109:sub-pixar109_task-pixar_run-001_swrf_bold.nii.gz'
plotting.plot_epi(os.path.join(data_dir, fmri_filename))
```


Uh-oh, what happened ?! Let's look back at the error message:

> DimensionError: Input data has incompatible dimensionality: Expected dimension is 3D and you provided a 4D image. See http://nilearn.github.io/manipulating_images/input_output.html.

We can fix that ! Let's take an average of the EPI image and plot that instead:



{:.input_area}
```python
from nilearn.image import mean_img

plotting.view_img(mean_img(os.path.join(data_dir, fmri_filename)))
```


## [Convert the fMRI volumes to a data matrix](http://nilearn.github.io/auto_examples/plot_decoding_tutorial.html#convert-the-fmri-volume-s-to-a-data-matrix)

These are some really lovely images, but for machine learning we want matrices so that we can use all of the techniques we learned this morning !

To transform our Nifti images into matrices, we'll use the `nilearn.input_data.NiftiMasker` to extract the fMRI data from a mask and convert it to data series.

First, let's do the simplest possible mask&mdash;a mask of the whole brain. We'll use a mask that ships with Nilearn and matches the MNI152 template we plotted earlier.



{:.input_area}
```python
brain_mask = datasets.load_mni152_brain_mask()
plotting.plot_roi(brain_mask, cmap='Paired')
```




{:.input_area}
```python
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=brain_mask, standardize=True)
masker
```




{:.input_area}
```python
# We give the masker a filename and retrieve a 2D array ready
# for machine learning with scikit-learn !
fmri_masked = masker.fit_transform(os.path.join(data_dir, fmri_filename))
print(fmri_masked)
```




{:.input_area}
```python
print(fmri_masked.shape)
```


One way to think about what just happened is to look at it visually:

![](http://nilearn.github.io/_images/masking.jpg)

Essentially, we can think about overlaying a 3D grid on an image. Then, our mask tells us which cubes or "voxels" (like 3D pixels) to sample from. Since our Nifti images are 4D files, we can't overlay a single grid -- instead, we use a series of 3D grids (one for each volume in the 4D file), so we can get a measurement for each voxel at each timepoint. These are reflected in the shape of the matrix ! You can check this by checking the number of positive voxels in our brain mask.

There are many other strategies in Nilearn [for masking data and for generating masks](http://nilearn.github.io/manipulating_images/manipulating_images.html#computing-and-applying-spatial-masks). I'd encourage you to spend some time exploring the documentation for these !

We can also [display this time series](http://nilearn.github.io/auto_examples/03_connectivity/plot_adhd_spheres.html#display-time-series) to get an intuition of how the whole brain signal is changing over time.

We'll display the first three voxels by sub-selecting values from the matrix. You can also find more information on [how to slice arrays here](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#basic-slicing-and-indexing).



{:.input_area}
```python
import matplotlib.pyplot as plt
plt.plot(fmri_masked[5:150, :3])

plt.title('Voxel Time Series')
plt.xlabel('Scan number')
plt.ylabel('Normalized signal')
plt.tight_layout()
```


## [Extracting signals from a brain parcellation](http://nilearn.github.io/auto_examples/03_connectivity/plot_signal_extraction.html#extracting-signals-from-a-brain-parcellation)

Now that we've seen how to create a data series from a single region-of-interest (ROI), we can start to scale up ! What if, instead of wanting to extract signal from one ROI, we want to define several ROIs and extract signal from all of them  ? Nilearn can help us with that, too ! 🎉

For this, we'll use `nilearn.input_data.NiftiLabelsMasker`. `NiftiLabelsMasker` which works like `NiftiMasker` except that it's for labelled data rather than binary. That is, since we have more than one ROI, we need more than one value ! Now that each ROI gets its own value, these values are treated as labels.



{:.input_area}
```python
# First, let's load a parcellation that we'd like to use
multiscale = datasets.fetch_atlas_basc_multiscale_2015()
print('Atlas ROIs are located at: %s' % multiscale.scale064)
```




{:.input_area}
```python
plotting.plot_roi(multiscale.scale064)
```




{:.input_area}
```python
from nilearn.input_data import NiftiLabelsMasker
label_masker = NiftiLabelsMasker(labels_img=multiscale.scale064, standardize=True)
label_masker
```




{:.input_area}
```python
fmri_matrix = label_masker.fit_transform(os.path.join(data_dir, fmri_filename))
print(fmri_matrix)
```




{:.input_area}
```python
print(fmri_matrix.shape)
```


### [Compute and display a correlation matrix](http://nilearn.github.io/auto_examples/03_connectivity/plot_signal_extraction.html#compute-and-display-a-correlation-matrix)

Now that we have a matrix, we'd like to create a _connectome_. A connectome is a map of the connections in the brain. Since we're working with functional data, however, we don't have access to actual connections. Instead, we'll use a measure of statistical dependency to infer the (possible) presence of a connection.

Here, we'll use Pearson's correlation as our measure of statistical dependency and compare how all of our ROIs from our chosen parcellation relate to one another.



{:.input_area}
```python
from nilearn import connectome
correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
correlation_measure
```




{:.input_area}
```python
correlation_matrix = correlation_measure.fit_transform([fmri_matrix])
correlation_matrix
```




{:.input_area}
```python
import numpy as np

correlation_matrix = correlation_matrix[0]
# Mask the main diagonal for visualization:
# np.fill_diagonal(correlation_matrix, 0)
plotting.plot_matrix(correlation_matrix)
```


### [The importance of specifying confounds](http://nilearn.github.io/auto_examples/03_connectivity/plot_signal_extraction.html#same-thing-without-confounds-to-stress-the-importance-of-confounds)

In fMRI, we're collecting a noisy signal. We have artifacts like physiological noise (from heartbeats, respiration) and head motion which can impact our estimates. Therefore, it's strongly recommended that you control for these and related measures when deriving your connectome measures. Here, we'll repeat the correlation matrix example, but this time we'll control for confounds. 



{:.input_area}
```python
conf_filename = 'derivatives:fmriprep:sub-pixar109:sub-pixar109_task-pixar_run-001_ART_and_CompCor_nuisance_regressors.tsv'
clean_fmri_matrix = label_masker.fit_transform(os.path.join(data_dir, fmri_filename),
                                               confounds=os.path.join(data_dir, conf_filename))
clean_correlation_matrix = correlation_measure.fit_transform([clean_fmri_matrix])[0]
np.fill_diagonal(clean_correlation_matrix, 0)
plotting.plot_matrix(clean_correlation_matrix)
```


That looks a little different !

Looking more closely, we can see that our correlation matrix is symmetrical; that is, that both sides of the diagonal contain the same information. We don't want to feed duplicate information into our machine learning classifier, and Nilearn has a really easy way to remove this redundancy ! 



{:.input_area}
```python
vectorized_correlation = connectome.ConnectivityMeasure(kind='correlation',
                                                        vectorize=True, discard_diagonal=True)
clean_vectorized_correlation = vectorized_correlation.fit_transform([clean_fmri_matrix])[0]
clean_vectorized_correlation.shape  # Why is this value not 64 * 64 ?
```


## [Interactive connectome plotting](http://nilearn.github.io/plotting/index.html#d-plots-of-connectomes)

It can also be helpful to project these connection weightings back on to the brain, to visualize these connectomes ! Here, we'll use the interactive connectome plotting in Nilearn.



{:.input_area}
```python
coords = plotting.find_parcellation_cut_coords(multiscale.scale064)
plotting.view_connectome(clean_correlation_matrix, coords=coords)
```




{:.input_area}
```python
?plotting.view_connectome
```


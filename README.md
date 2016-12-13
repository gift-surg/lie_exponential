Python 2.7 (.11)
Libraries: numpy, matplotlib, copy, os, tabulate, sympy, pickle, nibabel, scipy, timeit, math, warnings.


# Log-Euclidean Framework

Main aim of the code is the exploration of numerical methods to compute Lie exponential: differential operator that 
maps a SVF into the corresponding diffeomorphisms through the numerical integration of the the related ordinary 
differential equation.

From a differential geometry point of view, the set of stationary velocity field (SVF) constitutes the Lie algebra g 
of the Lie group G of diffeomorphisms that are embedded in a one parameter subgroup of the group of diffeomorphisms.
Diffeomorphisms considered are the one defined over a compact subset of R^2 or R^3.


Please do consider the references:

1. Schmid, Rudolf. "Infinite dimensional Lie groups with applications to mathematical physics." J. Geom. Symmetry Phys 1 (2004): 54-120.
2. Milnor "Remarks on infinite dimensional Lie Group" Relativity, groups and topology. 2. 1984.
3. Milnor "On infinite dimensional Lie groups" IAP preprint. 1982.
4. Holm, Schmah, Stoica "Geometric Mechanics and Symmetry: from finite to infinite dimensions". Oxford
texts in applied and engineering mathematics. 2009.
5. Arsigny, O. Commowick, X. Pennec, and N. Ayache. "A log-euclidean framework for statistics on diffeomorphisms." In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2006, pages 924–931. Springer, 2006.


## Field, Image and SVF classes

The OO structure is based on the following scheme:

    \Field
        \Image (nibabel wrapping)
            \s_vf
            \s_disp


Field (in utils folder) is an "enriched" numpy ndarray.
    Dimensions of the array and attributes are conformal to images and vector fields
    that are used in image processing.

Image (in utils folder) inherits from Field and wraps the library nibabel.
    Other that attributes and methods of Fields it includes the header and affine transformation
    information. In module resampler the resampling methods for this class.


s_disp (stationary displacement fields) are 5d images that models the spatial transformations.
    Can be used as discrete representations of diffeomorphisms.
    It can be provided in deformation coordinates (Eulerian coordinate system) or displacement coordinates (Lagrangian coordinate system) subtracting the identity. Deformation coordinates are considered only for the resampling, for any other numerical manipulation, vector fields are always considered in displacement coordinates.
    At each voxel x,y,z (or pixel x,z) is associated a vector of values u,v,w (or u,v) as

        s_disp(x,y,z,0,0) = u
        s_disp(x,y,z,0,1) = v
        s_disp(x,y,z,0,2) = w


s_vf (stationary velocity fields) are 5d images that models tangent vector fields of spatial transformations.
    Can be utilized to parametrize diffeomorphisms.
    If phi is a diffeomorphisms an svf v is defined as exp(v) = phi toward the stationary differential equation

        \frac{d}{dt}\phi  = v(\phi)

An s_vf can be mapped in the corresponding s_disp using the Lie exponential, whose numerical computaitons evaluations and comparisons are the main aim of this code.

An s_disp can be mapped in an s_vf using the Lie logarithm, using for example the inverse scaling and squaring. This algorithm is not included in this code.

## Lie group and Lie algebra of the rigid body transformations SE(2)

The Lie group of matrices SE(2) and its Lie algebra se(2) are defined in the folder transformation.
They are utilized to generate vector fields based on the rigid body transformation.
The exponential and logarithm of these objects have a known closed form that can be utilized as a ground
truth to validate the numerical methods implemented for their computation of the respective vector fields.

## WBIR 2016

The code was used to produce the results proposed in the proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops 2016:

http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w15/html/Ferraris_Accurate_Small_Deformation_CVPR_2016_paper.html


Please do consider the proposed references for additional informations.

The folder 

    main_error_analysis

contains the modules that provide the results of the proposed methods to compute the Lie exponential.
 
The folder 

    main_wbir

contains the methods that load the data saved by the methods in main_error_analysis and use them to produce graphs proposed in the WBIR proceedings.
 
 
# Where to start


Please create the folder "results_folder", and link it to the code inserting the path in the moduel path_manager under the folder utils. Substitute all the paths you find in this module. The module path_manager is used as well to link the data-sets used in the real case experiments to the code.

Once data are linked correctly, and the python libraries are downloaded, you can run all the methods in the folder 
main_error_analysis, selecting the methods parameters in the aaa_general_controller.py. 
If no stationary velocity field from longitudinal studies are available, all the methods that not ends in *_real.py
are likely to work anyway and to produce the results with sinthetic data generated inside the code.

Note (to reproduce the main results proposed in the WBIR paper): results proposed in the paper obtained 
from the manually segmented Neuromorphometric data-set have been obtained from the UCL cluster and are not publicly available. 
Related computations have been performed on the cluster as well, and results with this data set are not reproducible with this code. 
You can find nontheless the approximated exponential integrator implemented in nifty-Reg

    https://sourceforge.net/projects/niftyreg/

in the last dev version of the software, and you are invited to apply the methods using other data-sets.

Core method is SVF.exponential in the folder transformations/s_vf

## Nosetests

Run nosetests --exe to see if everything works. Code is tested on Mac OS X with the provided libraries. Please do contact me if you encounter any problem.

## Programming philosophy: warning the users

It is worth remember that programming is always the choice of the single through the multiple, and the variety of options 
and possibilities it is not possible to include everything, nor to reach any ideal 
"best possible code".
Albeit, there are some principles that are a consequence of the need and the final aim.
 
This code is to be intended as a work in progress. It is not meant to be a final product, nor to be as elegant or as 
exhaustive as possible.
It was written having in mind to create the shortest path toward the aim, mining any possible future change of 
the aim itself and the route toward him, that new needs and new information acquired during the development of a 
research usually generate.
 
Any question, contribution, comment and improvement from you is very welcome.


## Funders:

Sebastiano Ferraris is supported by the EPSRC-funded UCL Centre for Doctoral Training in Medical Imaging (EP/L016478/1) 
and Doctoral Training Grant (EP/M506448/1). Pankaj Daga was funded through an Innovative Engineering for Health award 
by Wellcome Trust [WT101957]; Engineering and Physical Sciences Research Council (EPSRC) [NS/A000027/1]. Marc Modat is 
supported by the UCL Leonard Wolfson Experimental Neurology Centre. Tom Vecauteren is supported by an Innovative 
Engineering for Health award by the Wellcome Trust [WT101957]; Engineering and Physical Sciences Research Council 
(EPSRC) [NS/A000027/1].

## Website

http://www.gift-surg.ac.uk/

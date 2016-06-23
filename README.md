Python 2.7 (.11)
Libraries: 

=======================
Log-Euclidean Framework
=======================

Code initially written for my Master of Research, grounded on the Pankaj's code NiftyBIT.

From a differential geometry point of view, the set of SVF forms the Lie algebra g of the Lie group G of diffeomorphisms 
that are embedded in a one parameter subgroup of the group of diffeomorphisms.
Diffeomorphisms considered are the one defined over a compact subset of R^2 or R^3.

Main aim of the code is the exploration of numerical methods to compute Lie exponential: differential operator that 
maps a SVF into the corresponding diffeomorphisms through the solution of the the related ordinary differential equation.

Please do consider the references:
Rudolf Schmidt, Infinite-Dimensional Lie Groups and Algebras in Mathematical physics
Milnor "Remarks on infinite dimensional Lie Group" 1984
Milnor "On infinite dimensional Lie groups" IAP preprint 1982
Holm, Schmah, Stoica "Geometric Mechanics and Symmetry: from finite to infinite dimensions"


=====================
Field, images and SVF
=====================

The OO structure is based on the following structure:

Objects:

    \Field
        \Image (nibabel wrapping)
            \s_vf
            \s_disp


Field is an enriched numpy ndarray (in utils folder).
    Dimensions of the array and attributes are conformal to images and vector fields
    that are used in image processing.

Image is an enriched Field that wraps the library nibabel (in utils folder)..
    Other that attributes and methods of Fields it also has the header and affine transformation
    information. In module resampler the resampling methods for this class


s_disp (stationary displacement fields) are 5d images that models the spatial transformations.
    Can be used as discrete representations of diffeomorphisms.
    If phi is a diffeomorphism defined over Omega, a displacement is defined as (phi - Id)(x)
    where x is an element of Omega, and an s_disp here considered are discretisation of these objects.
    At each voxel x,y,z (or pixel x,z) is associated a vector of values u,v,w (or u,v) as

        s_disp(x,y,z,0,0) = u
        s_disp(x,y,z,0,1) = v
        s_disp(x,y,z,0,2) = w

An s_disp can be mapped in an s_vf using the map logarithm whose numerical computations are 
performed with various techniques.

s_vf (stationary velocity fields) are 5d images that models tangent vector fields of spatial transformations.
    Can be utilized to parametrize diffeomorphisms.
    If phi is a diffeomorphisms an svf v is defined as exp(v) = phi toward the stationary differential equation

        \frac{d}{dt}\phi  = v(\phi)

An s_vf can be mapped to an s_disp using the map exponential whose numerical computations are
    performed with various techniques.


Other transformations are the in the finite dimensional Lie group SE(2).

=================================================================
Lie group and Lie algebra of the rigid body transformations SE(2)
=================================================================

Within the vector fields, the Lie group of matrices SE(2) and its Lie algebra se(2) are defined as well.
They can utilized to generate vector fields.
The exponential and logarithm of these objects possess a closed form that can be utilized as a ground
truth to validate the numerical methods implemented for their computation of the respective vector fields.

=================================================================
WBIR 2016
=================================================================
The code has been used to propose some results obtained from the application of Exponential Integrators embedded in the
Euler and in the Scaling and Squaring integration framework.

Please do consider the references:
Arsigny et al. "A fast an log-euclidean polyaffine framework for locally linear registration"
Arsigny et al. "A log-euclidean framework for statistics on diffeomorphisms"
Yng, Candes "The phase flow method" 

Folder 

    main_error_analysis

contain the module with the results of the prototyped methods in utils and transformation.
 
Folder 

    main_wbir

loads the data saved by the methods contained in the folder main_error_analysis and used to produce graphs proposed in 
the paper.
 
 
==============================================================
Where to start
==============================================================

Create a folder called results_folder, and link it in the path_manager under utils, erasing the path you find in the 
code. Path manager is used as well to link the data-sets used in the real case experiments to the code.

Once data are linked correctly, and the libraries are downloaded please run all the methods in the folder 
main_error_analysis, selecting the methods parameters in the aaa_general_controller.py. 
If no real svf from longitudinal studies are available, all the methods that not ends in *_real.py
are likely to work without them.

Note (if your aim is to reproduce all the results proposed in the WBIR paper): results proposed in the paper obtained 
from the manually segmented Neuromorphometric data-set have been obtained from the UCL cluster and are not publicly available. 
Related computations have been performed on the cluster as well, and results are not reproducible with this code. 
You can find nontheless the approximated exponential integrator implemented in nifty-Reg in the last dev version of the
software, and you are invited to reproduce the results using other data-sets.

Core method is SVF.exponential in transformations/s_vf

==============================================================
Programming philosophy: warning the users
==============================================================

It is worth remember that programming is always the choice of the single through the multiple, and the jungle of options 
and possibilities is so varied and vast that it is not possible to include everything, nor to reach any ideal 
"best possible code".
Albeit there are some principles, forged by the need and purpose, that I tried to follow.
 
This code is to be intended as a work in progress. It is not meant to be a final product, nor to be as elegant or as 
exhaustive as possible.
It was written having in mind to create the shortest path toward the aim, mining any possible future change of 
the aim itself and the route toward him, that new needs and new information acquired during the development of a 
research usually generate.

In some parts, as the structural one, where transformations and utils are stored, I tried to keep the minimum amount 
of code and to have it reasonably tested. In other parts as in the playground and in the main, the code is repetitive 
and scarcely if not at all tested. 
Few attempt has been made to make these parts less redundant in the perspective of future 
modifications, for example in the structure of the input data as well as in the structure of the module itself.
Although testing and the use of shortcuts to reduce the amount of code are invaluable to increase code maintainability 
and the certainty of results while developing a product, in the circumstance of doing research these invaluable helps 
hinders any quick modification and are a great limitation of freedom to introduce new unexpected concepts from other 
sources.

Lastly this code is not written by an experienced programmer, but it still has the ambition of being "scientific" 
i.e. it is open and it can be peer reviewed. Any contribution, comment and improvement from you is therefore welcome.

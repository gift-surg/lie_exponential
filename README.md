# Lie exponential computation for the Log-Euclidean Framework

Main aim of the code is to benchmark a range of numerical methods to compute Lie exponential: differential operator that
maps a SVF into the corresponding diffeomorphisms through the numerical integration of the the related ordinary 
differential equation. Code is written in python 3.6, back compatible with python 2.7.

In the current version, this repository is now based on the external library (calie)[https://github.com/SebastianoF/calie].
See also related documentation.
Please checkout the branch `old-state` for the version of the code proposed to produce the results of the WBIR.

## Main bibliography

1. Schmid, Rudolf. "Infinite dimensional Lie groups with applications to mathematical physics." J. Geom. Symmetry Phys 1 (2004): 54-120.
2. Milnor "Remarks on infinite dimensional Lie Group" Relativity, groups and topology. 2. 1984.
3. Milnor "On infinite dimensional Lie groups" IAP preprint. 1982.
4. Holm, Schmah, Stoica "Geometric Mechanics and Symmetry: from finite to infinite dimensions". Oxford
texts in applied and engineering mathematics. 2009.
5. Arsigny, O. Commowick, X. Pennec, and N. Ayache. "A log-euclidean framework for statistics on diffeomorphisms." In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2006, pages 924–931. Springer, 2006.


## Acknowledgements

Sebastiano Ferraris is supported by the EPSRC-funded UCL Centre for Doctoral Training in Medical Imaging (EP/L016478/1) 
and Doctoral Training Grant (EP/M506448/1). Pankaj Daga was funded through an Innovative Engineering for Health award 
by Wellcome Trust [WT101957]; Engineering and Physical Sciences Research Council (EPSRC) [NS/A000027/1]. Marc Modat is 
supported by the UCL Leonard Wolfson Experimental Neurology Centre. Tom Vecauteren is supported by an Innovative 
Engineering for Health award by the Wellcome Trust [WT101957]; Engineering and Physical Sciences Research Council 
(EPSRC) [NS/A000027/1].

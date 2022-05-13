===============================
The CSLB Concept Property Norms
===============================

This document describes the CSLB Norms data files. For more information about the 
norms, see:

    Devereux, B.J., Tyler, L.K., Geertzen, J., Randall, B. (in press). The Centre for Speech, Language
        and the Brain (CSLB) Concept Property Norms. Behavior Research Methods. 
        DOI: 10.3758/s13428-013-0420-4
      
For any additional assistance/support in using the norms, please contact Barry Devereux 
(barry@csl.psychol.cam.ac.uk)


Version History
---------------
This is version 1.1, released 2013-12-24 (first public release). 


Files
-----
Several files are included, which are described in more detail in the following sections:

- feature_matrix.dat: conceptXfeature production frequency matrix
- norms.dat: full list of features and feature variations
- cosine_noTax_5pf_long.dat: cosine similarities, excluding taxonomic features
- cosine_wTax_5pf_long.dat: cosine similarities, including taxonomic features   


feature_matrix.dat
------------------
A tab-delimited file, where each row is a production frequency vector for a concept. The first row
lists the names of the feature dimensions (normalized feature labels). The first column lists the 
names of the concepts.


norms.dat
---------
Each row lists a concept/feature pair. The following information is given:

Domain
~~~~~~
Whether the concept is living or non-living.

Feature Type
~~~~~~~~~~~~
A label indicative of the type of feature. Used labels are:

- visual perceptual
- other perceptual
- functional
- taxonomic
- encyclopaedic

Concept
~~~~~~~
The concept name. The relationship to the McRae et al. (2005) concept names is provided in the 
article Appendix. 

Feature
~~~~~~~
The final normalized feature label.

PF
~~ 
Production Frequency. The total number of participants who gave a response that was mapped to the 
given feature label. All concept/feature pairs with PF > 1 are given. 

Feature alternatives
~~~~~~~~~~~~~~~~~~~~
A semi-colon-delimited list of the linguistic variation that was mapped to the given 
normalized feature label. Note that automatically re-written variations are not given. In 
particular, the following syntactic patterns are collapsed into a single feature:
 
    does, can, may, might, third person singular of verb

Example:
    does eat, can eat, may eat, might eat, eats -> does eat

Singular and plurals are collapsed.

Example:
    is found in gardens; is found in a garden -> is found in gardens

Variation in the use of articles is also not shown.

Example:
    is found in a kitchen; is found in the kitchen; is found in kitchens -> is found in kitchens

Only linguistic variation in the raw responses relevant to each normalized feature is given.

Participant list
~~~~~~~~~~~~~~~~
A list of numerical identifiers for participants, specifying which participants gave a response that 
was mapped to the given feature label (the numerical identifiers are concept-specific; they do not 
correspond across concepts). Groups of participants are delimited by "/"; each group corresponds to 
one of the variations listed in the "Feature alternatives" column. It is thus possible to determine 
which participant gave which variation of each feature within each concept.


cosine_noTax_5pf_long.dat
-------------------------
Similarity score (cosine between feature vectors) for each pair of concepts. Taxonomic features 
are excluded from the cosine calculation. Features with production frequencies less than 5 are 
excluded. Data presented in "long" format (each row gives the similarity for a pair of concepts).


cosine_wTax_5pf_long.dat
------------------------
Same as cosine_noTax_5pf_long.dat, except WITH taxonomic features included.

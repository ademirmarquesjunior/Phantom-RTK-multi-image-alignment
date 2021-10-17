# Phantom-RTK-multi-camera-alignment
Program developed to facilitate the camera processing alignement and RTK processing of multispectral images acquired with the DJI P4 multi RTK.

## Install

To install this software/script download and unpack the software package in any folder.

 To install the required libraries navigate to script folder and run:
 
     pip install -r Requirements.txt
     
## Usage
 
 To run this program use:
 
    python main.py
  
Choose the the input folder with original images acquired with the DJI P4 multispectral UAV, and choose and or create the output folder where the aligned spectral images will be saved. Hit "Align images" to start the process.

<img src="https://github.com/ademirmarquesjunior/Phantom-RTK-multi-image-alignment/blob/main/idle.png" width="500" alt="Segmented image">
<img src="https://github.com/ademirmarquesjunior/Phantom-RTK-multi-image-alignment/blob/main/done.png" width="500" alt="Segmented image">


* Agisoft Metashape processing requires the insertion of camera calibration information for aditional cameras besides the first. To be fixed.


## TODO

Next iterations expect to improve and incorporate:

 - Update tag/exif information on all saved images
 - Update tags with post processed GNSS-RTK information


## Credits	
This work is credited to the [Vizlab | X-Reality and GeoInformatics Lab](http://vizlab.unisinos.br/) and the following developers:	[Ademir Marques Junior](https://www.researchgate.net/profile/Ademir_Junior).

## License

    MIT Licence (https://mit-license.org/)
    
## How to cite

Yet to be published.

# tp_analysis
The analysis code for Thomson parabola spectrometer (TPS).
The approval file is an image data obtained with IP, MCP, e.t.c.

## Lisence
Contact to the author (T. Minami, takumi.minami.9360@gmail.com).

## For CLI
### Installation
Execute ```$ git clone https://github.com/takumiminami/tp_analysis.git```.
Make a symbolic linc in a directory with path to tp_analysis/tpana, e.g. ```ln -s ${Installed_directory}/tp_analysis/tpana ${Directory_with_PATH}```, then you can execute the script anywhere.

### Execution
Put the data and input.py in a directory.
Edit the input.py, then execute ```$ tpana``` in the directory.
You can set an arbitrary file name of the input, e.g. laser_shot5.py, but you should follow the nomencalture rule of the python variables, for example, the use of a hyphen is prohibitted. 
You can execute with ```$ tpana laser_shot5.py```.

### Update
```$ tpana update``` will automatically update the codes from github if there are available updates.

### Uninstallation
Execute ```$ tpana uninstall```, then you will be demanded to input "y" to verify the uninstallation.
The directory "tp_analysis" and all the files in the directory will be removed automatically.

## For Visual studio code (VSC), Anaconda, and e.t.c.
Download the code with .zip format in the browser and copy the contents to the working directory.


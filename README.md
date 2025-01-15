# tp_analysis
The analysis code for Thomson parabola spectrometer (TPS).
The approval file is an image data obtained with IP, MCP, e.t.c.

## Lisence
Contact to the author (T. Minami, takumi.minami@eie.eng.osaka-u.ac.jp).
## For CLI
### Installation
Make an empty file in a directory with path (e.g. ```$ touch tpana``` in $HOME/bin). Then, copy and paste the following script in the file,
```
#!/bin/bash
set -Ce

argv=${1:-}
script_dir=$HOME/bin

if [ "$argv" = "update" ]; then
    cd $script_dir/tp_analysis && git pull  
    echo "update done"
elif [ "$argv" = "install" ]; then
    cd $script_dir && git clone https://github.com/takumiminami/tp_analysis.git
elif [ "$argv" = "" ]; then
    python3 $script_dir/tp_analysis/main.py
else
    python3 $script_dir/tp_analysis/main.py $argv 
fi
```
and set it executable (```$ chmod +x tpana```).
You can install the code with ```$ tpana install``` if you already have the lisence.
You can set an arbitrary installation directory by editing the variable ```script_dir```.

### Execution
Put the data and input.py in a directory.
Edit the input.py, then execute ```$ tpana``` in the directory.
You can set an arbitrary file name of the input, e.g. laser_shot5.py, but you should follow the nomencalture rule of the python variables, for example, the use of a hyphen is prohibitted. 
You can execute with ```$ tpana laser_shot5.py```.

### Update
```$ tpana update``` will automatically update the codes from github if there are available updates.

## For Visual studio code (VSC), Anaconda, and e.t.c.
Download the code with .zip format in the browser and copy the contents to the working directory.


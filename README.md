# tp_analysis
The analysis code for Thomson parabola spectrometer (TPS).
The approval file is an image data obtained with IP, MCP, or some detector in TPS.

## Lisence
Contact to the author (T. Minami, takumi.minami@eie.eng.osaka-u.ac.jp).
## For CLI
### Installation
Make an empty file in a directory with path (e.g. ```$ touch tpana``` in $HOME/bin). Then, copy and paste the following script in the file,
```
#!/bin/bash
set -Ceu

update_flag=$1
script_dir=$PYTHONPATH

if [ "$update_flag" = "update" ]; then
    cd $script_dir/tp_analysis && git pull  
    echo "update done"
elif [ "$update_flag" = "install" ]; then
    cd $script_dir && git clone https://github.com/takumiminami/tp_analysis.git
else
    echo "Not an approved command '"$update_flag"'"
fi

if [ $# -lt 1 ]; then
    python3 $script_dir/main.py
fi
```
and set it executable (```$ chmod +x tpana```).
You can install the code with ```$ tpana install``` if you already have the lisence.

### Execution
Put the data and input.py in a directory.
Edit the input.py, then execute ```$ tpana``` in the directory.

### Update
```$ tpana update``` will automatically update the codes from github if there are available updates.

## For Visual studio code (VSC), Anaconda, and e.t.c.
Download the code with .zip format in the browser and copy the contents to the working directory.


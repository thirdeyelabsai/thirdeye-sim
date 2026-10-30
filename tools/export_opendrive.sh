#!/bin/sh
# Write the road network as OpenDRIVE 1.4 (exports/gurgaon.xodr, UTM zone 43N georeference with SUMO's offset in
# <header><offset>), and, with ESMINI pointing at esmini's bin folder, check that esmini loads it and drives a car
# along Sunset Boulevard for 20 s (exports/esmini_test/drive.xosc). Tested with netconvert 1.28.0, esmini 3.9.0.
#     sh tools/export_opendrive.sh                     ESMINI=~/esmini/bin sh tools/export_opendrive.sh
set -e
cd "$(dirname "$0")/.."
NETCONVERT=${NETCONVERT:-.venv-sumo/bin/netconvert}
mkdir -p exports
$NETCONVERT --sumo-net-file gurgaon/gurgaon.net.xml --opendrive-output exports/gurgaon.xodr
if [ -n "$ESMINI" ]; then
    cd exports/esmini_test
    "$ESMINI/esmini" --headless --osc drive.xosc --fixed_timestep 0.05 --record drive.dat --disable_log
    "$ESMINI/dat2csv" drive.dat
    tail -1 drive.csv
fi

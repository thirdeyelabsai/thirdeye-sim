#!/bin/sh
# Build gurgaon.net.xml from gurgaon.osm (bbox 28.4565,77.0540,28.4725,77.1090).
# 1. Import the OSM roads: left-hand traffic, joined junctions, signals from OSM tags plus guessed ones (18).
#    No U-turns at junctions (only at dead ends), and turning traffic waits at the stop line rather than in the
#    middle of the junction: both caused vehicles to overlap inside junctions (28 Oct).
# 2. Set the speed limits by road class (set_speeds.py) on a plain-XML copy and rebuild from it, so the
#    junctions and signals stay exactly as imported (guessing signals again at the new speeds would move them).
# 3. Apply hand fixes checked against satellite imagery (junction_fixes.json, fix_junctions.py).
# Lanes without a width in OSM are 3.5 m, the Indian standard (netconvert's default is 3.2 m). Secondary and
# tertiary roads without a lane count in OSM get two lanes each way (osmGurgaon.typ.xml), not netconvert's one.
# 0. Cut the download at its box first (clip_osm.py): roads that cross the edge end exactly on it, where the
#    game's boundary wall stands, instead of running on as fragments without the streets around them (29 Oct).
#    The offset is pinned to the one the network had before the cut, so map coordinates stay the same.
set -e
cd "$(dirname "$0")"
OUT=${OUT:-gurgaon.net.xml}   # OUT and EXTRA (more netconvert options) are for trying variants
NC=../.venv-sumo/bin/netconvert
TMP=$(mktemp -d)
if [ "${CLIP:-1}" = 1 ]; then ../.venv-sumo/bin/python clip_osm.py gurgaon.osm $TMP/play.osm; else cp gurgaon.osm $TMP/play.osm; fi   # CLIP=0: the whole download
$NC --osm-files $TMP/play.osm -o $TMP/osm.net.xml --offset.disable-normalization true --offset.x -699124.01 --offset.y -3146552.76 --type-files osmGurgaon.typ.xml --lefthand true --geometry.remove --roundabouts.guess --ramps.guess \
    --junctions.join --junctions.corner-detail 5 --tls.guess-signals --tls.guess true --tls.guess.threshold 69 \
    --tls.discard-simple --tls.join --tls.default-type actuated --sidewalks.guess --crossings.guess \
    --remove-edges.isolated --keep-edges.by-vclass passenger --output.street-names --output.original-names \
    --osm.elevation false --no-turnarounds.except-deadend --default.connection.cont-pos 0 --no-warnings $EXTRA
$NC -s $TMP/osm.net.xml --plain-output-prefix $TMP/p --no-warnings
../.venv-sumo/bin/python set_speeds.py $TMP/p.edg.xml
../.venv-sumo/bin/python fix_junctions.py $TMP/p $TMP/osm.net.xml
$NC -n $TMP/p.nod.xml -e $TMP/p.edg.xml -x $TMP/p.con.xml -i $TMP/p.tll.xml -t $TMP/p.typ.xml --lefthand true \
    --no-turnarounds.except-deadend --default.connection.cont-pos 0 --default.lanewidth 3.5 --offset.disable-normalization true \
    -o $OUT --no-warnings $EXTRA
# netconvert applies the pinned offset but writes its y with the wrong sign in <location>; write it back as
# applied (checked 29 Oct: node positions match the old network to 2 cm, latitude comes out right)
sed -i '' 's/netOffset="[^"]*"/netOffset="-699124.01,-3146552.76"/' $OUT
rm -r $TMP

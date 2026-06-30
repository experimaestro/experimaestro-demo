#!/bin/sh
set -e

CONFIG_DIR="$HOME/.config/experimaestro"
CONFIG_FILE="$CONFIG_DIR/settings.yaml"

mkdir -p "$CONFIG_DIR"

if [ -f "$CONFIG_FILE" ]; then
    echo "$CONFIG_FILE already exists !"
else
    cp xpm_settings.yaml "$CONFIG_FILE"
    echo "Installed $CONFIG_FILE"
fi
echo "Installation is ok, you may want to check the settings in $CONFIG_FILE" 
echo "See https://experimaestro-python.readthedocs.io/en/latest/settings.html for more information."

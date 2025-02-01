#!/bin/bash

# Set variables
PROJECT_ROOT=$(pwd)
TARGET_BINARY="$PROJECT_ROOT/target/release/raspberrypi_people_detection"
PACKAGE_DIR="$PROJECT_ROOT/linux/raspberrypi_people_detection_1.0.0-1_amd64"
BIN_DIR="$PACKAGE_DIR/usr/bin"
DEB_PACKAGE_NAME="raspberrypi_people_detection_v1.0.0_1_amd64.deb"

# Ensure the binary exists
if [ ! -f "$TARGET_BINARY" ]; then
    echo "Error: Compiled binary not found in target/release/. Please build the project first."
    exit 1
fi

# Ensure the destination directory exists
mkdir -p "$BIN_DIR"

# Copy the binary
cp "$TARGET_BINARY" "$BIN_DIR/"

# Build the Debian package
dpkg-deb --build "$PACKAGE_DIR"

echo "Debian package created: $PROJECT_ROOT/linux/raspberrypi_people_detection_1.0.0-1_amd64"

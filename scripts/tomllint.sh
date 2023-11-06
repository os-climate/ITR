#!/bin/bash

status_code="0"
TAPLO_URL=https://github.com/tamasfe/taplo/releases/download/0.8.1

# Process commmand-line arguments
if [ $# -eq 0 ]; then
    TARGET=$(pwd)
elif [ $# -eq 1 ]; then
    TARGET="$1"
fi

check_platform() {
    # Enumerate platform and set binary name appropriately
    PLATFORM=$(uname -a)
    if (echo "${PLATFORM}" | grep Darwin | grep arm64); then
        TAPLO_VER="taplo-darwin-aarch64.gz"
    elif (echo "${PLATFORM}" | grep Darwin | grep x86_64); then
        TAPLO_VER="taplo-darwin-x86_64.gz"
    elif (echo "${PLATFORM}" | grep Linux | grep aarch64); then
        TAPLO_VER="taplo-full-linux-aarch64.gz"
    elif (echo "${PLATFORM}" | grep Linux | grep x86_64); then
        TAPLO_VER="taplo-full-linux-x86_64.gz"
    else
        echo "Unsupported platform!"; exit 1
    fi
}

check_file() {
    local file_path="$1"
    cp "$file_path" "$file_path.original"
    "${TAPLO_BIN}" format "$file_path" >/dev/null
    diff "$file_path" "$file_path.original"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        status_code=$exit_code
        echo "::error file={$file_path},line={line},col={col}::{TOML file not formatted}"
    elif [ -f "$file_path.original" ]; then
        rm "$file_path.original"
    fi
}

check_all() {
    if [ -d "${TARGET}" ]; then
        echo "Scanning all the TOML files at folder: ${TARGET}"
    fi
    while IFS= read -r current_file; do
        echo "Check file $current_file"
        check_file "$current_file"
    done < <(find . -name '*.toml' -type f -not -path '*/.*')
}

download_taplo() {
    TAPLO_GZ=$(echo "${TAPLO_VER}" | sed "s/.gz//g")
    TAPLO_BIN=/tmp/"${TAPLO_GZ}"
    if [ ! -f /tmp/"${TAPLO_GZ}" ]; then
        "${WGET_BIN}" -q -e robots=off -P /tmp "${TAPLO_URL}"/"${TAPLO_VER}"
    fi
    if [ ! -x "${TAPLO_BIN}" ]; then
        gzip -d /tmp/"${TAPLO_VER}"
        chmod +x /tmp/"${TAPLO_GZ}"
    fi
}

check_wget() {
    # Pre-flight binary checks and download
    WGET_BIN=$(which wget)
    if [ ! -x "${WGET_BIN}" ]; then
        echo "WGET command not found"; exit 1
    fi
}

TAPLO_BIN=$(which taplo)
if [ ! -x "${TAPLO_BIN}" ]; then
    check_wget && check_platform && download_taplo
fi

if [ ! -x "${TAPLO_BIN}" ]; then
    echo "TOML linting binary not found [taplo]"; exit 1
fi

cleanup_tmp() {
    # Only clean the temp directory if it was used
    if (echo "${TAPLO_BIN}" | grep "/tmp" > /dev/null 2>&1)
    then
        echo "Cleaning up..."
        if [ -f "${TAPLO_BIN}" ]; then
            rm "${TAPLO_BIN}"
        fi
        if [ -f "${TAPLO_GZ}" ]; then
            rm "${TAPLO_GZ}"
        fi
    fi
}

# To avoid execution when sourcing this script for testing
[ "$0" = "${BASH_SOURCE[0]}" ] && check_all "$@"
cleanup_tmp
exit $status_code

#!/bin/bash

set -o pipefail
# set -xv

# Define variables

REPO_DIR=$(git rev-parse --show-toplevel)
FILE_SOURCE="pyproject.toml"

# Set these flags to false, then selectively enable based on arguments passed to script
unquoted=single=double=nocommas=brackets="false"

# Define functions

# Trim leading/trailing whitespace
remove_whitespace() {
    echo "$1" | awk '{$1=$1};1'
}

# No quotations
unquoted() {
    echo "${1//\"/}"
}

# Swaps quotes to single quotes
double_quotes_to_single() {
    echo "${1//\"/\'}"
}

# Wraps values in square brackets like an array
add_square_brackets() {
    echo "[$1]"
}

# Provides just raw comma separated numbers
strip_commas() {
    echo "${1//,/}"
}

# Provides the default return values, as below
# Example: "3.11", "3.10", "3.9"
return_result() {
    echo "$1"
}

check_for_arg_conflict() {
    COUNT="0"
    if [ "$unquoted" = "true" ]; then
        COUNT=$((COUNT+1))
    fi
    if [ "$single" = "true" ]; then
        COUNT=$((COUNT+1))
    fi
    if [ "$double" = "true" ]; then
        COUNT=$((COUNT+1))
    fi
    if [ "$COUNT" -gt 1 ]; then
        echo "ERROR: these arguments are mutually exclusive: unquoted|single|double"
        exit 1
    elif [ "$COUNT" -eq 0 ]; then
        echo "ERROR: mandatory to specify one of the following: unquoted|single|double"
        exit 1
    fi
}

# Main script entry point

# Process script arguments
if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage:    $0  unquoted|single|double [nocommas, brackets]"; exit 1
else
    NUM_ARGS="$#"
    for ARGUMENT in "$@"; do
        # Check for valid arguments
        if [ "$ARGUMENT" != "unquoted" ] && \
        [ "$ARGUMENT" != "single" ] && \
        [ "$ARGUMENT" != "double" ] && \
        [ "$ARGUMENT" != "nocommas" ] && \
        [ "$ARGUMENT" != "brackets" ]; then
            echo "ERROR: invalid arugment specified $ARGUMENT"
            echo "Valid arguments are unquoted|single|double, nocommas, brackets"
            exit 1
        fi

        # Set flags appropriately
        if [ "$ARGUMENT" = "unquoted" ]; then
            unquoted="true"
        elif [ "$ARGUMENT" = "single" ]; then
            single="true"
        elif [ "$ARGUMENT" = "double" ]; then
            double="true"
        elif [ "$ARGUMENT" = "nocommas" ]; then
            nocommas="true"
        elif [ "$ARGUMENT" = "brackets" ]; then
            brackets="true"
        fi



        # Count arguments and check for duplicates
        if [ -z "$ARGS" ]; then
            printf -v ARGS "%s" "$ARGUMENT"
        else
            printf -v ARGS "%s\n%s" "$ARGUMENT" "$ARGS"
        fi
    done

    # Makes sure contradictory options have not been requested
    # e.g. unquoted|single|double are mutually exclusive
    check_for_arg_conflict

    UNIQUE_ARGS=$(echo "$ARGS" | sort | uniq | wc -l)
    if [ "$NUM_ARGS" -ne "$UNIQUE_ARGS" ]; then
        echo "ERROR: you cannot use the same argument twice"; exit 1
    fi
fi

# Extract Python version string(s) from source
if [ -f "$REPO_DIR"/"$FILE_SOURCE" ]; then
    PYTHON_VERSIONS=$(grep "Programming Language :: Python :: " "$REPO_DIR"/"$FILE_SOURCE" | \
        sed "s/Programming Language :: Python :: //g" | \
        sed 's/"3 :: Only",//g' | \
        sed 's/"3",//g' | \
        sed '$s/.$//' | \
        tr '\n' ' ' | \
        sed 's/ \{2,\}/ /g')
else
    echo "Could not locate input file: $FILE_SOURCE"; exit 1
fi

# Trim leading/trailing whitespace
PYTHON_VERSIONS=$(remove_whitespace "$PYTHON_VERSIONS")

# Depending on arguments, modify returned string
if [ "$unquoted" = "true" ]; then
    PYTHON_VERSIONS=$(unquoted "$PYTHON_VERSIONS")
elif [ "$single" = "true" ]; then
    PYTHON_VERSIONS=$(double_quotes_to_single "$PYTHON_VERSIONS")
fi
if [ "$nocommas" = "true" ]; then
    PYTHON_VERSIONS=$(strip_commas "$PYTHON_VERSIONS")
fi
if [ "$brackets" = "true" ]; then
    PYTHON_VERSIONS=$(add_square_brackets "$PYTHON_VERSIONS")
fi

# Prints the Python strings in the requested format
return_result "$PYTHON_VERSIONS"

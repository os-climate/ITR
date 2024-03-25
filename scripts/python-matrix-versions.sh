#!/bin/sh

REPO_DIR=$(git rev-parse --show-toplevel)

if [ -f "$REPO_DIR"/pyproject.toml ]; then
    TOML_VERS=$(grep "Programming Language :: Python :: " "$REPO_DIR"/pyproject.toml | \
        sed "s/Programming Language :: Python :: //g" | \
        sed 's/"3 :: Only",//g' | \
        sed 's/"3",//g' | \
        sed '$s/.$//' | \
        tr '\n' ' ' | \
        sed 's/ \{2,\}/ /g')
else
    echo "Could not locate input file: pyproject.toml"; exit 1
fi

# Returns: "3.11", "3.10", "3.9"
echo "$TOML_VERS"

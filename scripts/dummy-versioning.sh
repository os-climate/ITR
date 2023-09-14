#!/bin/bash

#set -x

FILEPATH="pyproject.toml"

if [ $# -ne 1 ];then
	echo "Usage: $0 [version-string]"
	echo "Substitutes the version string in pyproject.toml"; exit 1
else
	VERSION=$1
	echo "Received version string: $VERSION"
fi

echo "Performing string substitution on: $FILEPATH"
sed -i "s/.*version =.*/version = \"$VERSION\"/" $FILEPATH
echo "Versioning set to:"
grep version $FILEPATH
echo "Script completed!"; exit 0

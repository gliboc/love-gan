#! /bin/bash

for d in $(find . -maxdepth 3 -type d) ; do
  echo "$d"
  touch .keep
done

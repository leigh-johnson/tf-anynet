#!/bin/zsh

function data_sanity(){
    total_files=$(find $1 -type f -name '*.pfm' | wc -l)
    echo "Found ${total_files} files in ${}"
    count=0

    color=Pf
    resolution=960[[:space:]]540
    endian=-1.0
    for file in $(find $1 -type f -name '*.pfm' -print0)
    do
        headers=$(head -n 3 $file | awk 'NR%3{printf "%s ",$0;next;}1')
            
        [[ ! $headers =~ $color || ! $headers =~ $resolution || ! $headers =~ $endian ]] && echo "oh god ${file} ${headers}"
        (( count++ ))
        if (( $count % 1000 == 0 ))
        then
            echo "Files remaining:$(($total_files-$count))"
        fi
    done
}
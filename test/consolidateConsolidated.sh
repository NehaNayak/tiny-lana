for i in `ls ../params/os*con*`
do
    echo -n -e $i'\t'| tr '_' '\t'

    cat $i | tr '_' '\t'
done

for i in `ls ../params/*Pr2*con*`
do
    echo -n -e $i'\t'| tr '_' '\t'

    cat $i | tr '_' '\t'
done

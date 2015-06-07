cd /afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/tiny-lana/test

for hiddenFactor in 0.75
do
    for learningRate in 0.05
    do
        # missing some values because i derped
        for regCoeff in 0.0001 0.0003 3e-05
        do
            for protoWeight in 0.001 0.003
            do
                for trainDev in Train Dev
                do
                    filename='../params/olPr2_EH_i100_h'$hiddenFactor'_lr'$learningRate'_rc'$regCoeff'_pw'$protoWeight'_el10_osgd_vg_s'$trainDev'.txt'
                    python findRank.py cosines/gloveRanks_EH_100.pickle $filename
                done
            done
        done
    done
done

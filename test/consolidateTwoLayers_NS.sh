cd /afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/tiny-lana/test

for hiddenFactor in 0.75 #0.75 1
do
    for learningRate in 0.05 #0.1 0.05 0.05
    do
        for regCoeff in 0 0.001 0.0003 0.0001 3e-05
        do
            for negSamples in 0 5 10
            do
                for trainDev in Train Dev
                do
                    filename='../params/tlNS_EH_i100_h'$hiddenFactor'_lr'$learningRate'_rc'$regCoeff'_ns'$negSamples'_el10_osgd_vg_s'$trainDev'.txt'
                    python findRank.py cosines/gloveRanks_EH_100.pickle $filename
                done
            done
        done
    done
done

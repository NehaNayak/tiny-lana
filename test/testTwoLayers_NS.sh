cd /afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/tiny-lana/test

for hiddenFactor in 0.75
do
    for learningRate in 0.05
    do
        for regCoeff in 3e-05 #0 0.001 0.0003 0.0001 0.00003
        do
            for negSamples in 0 5 10
            do
                for trainDev in Train Dev
                do
                    torch-lua testTwoLayers_NS.lua \
                    -hiddenFactor $hiddenFactor \
                    -learningRate $learningRate \
                    -regCoeff $regCoeff \
                    -negSamples $negSamples \
                    -trainDev $trainDev \
		            -endLimit 10
                done
            done
        done
    done
done

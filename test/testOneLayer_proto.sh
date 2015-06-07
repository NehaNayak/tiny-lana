cd /afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/tiny-lana/test

for hiddenFactor in 0.75
do
    for learningRate in 0.05
    do
        for regCoeff in 0 0.001 0.0003 0.0001 0.00003
        do
            for protoWeight in 0.1 0.01 0.03
            do
                for trainDev in Train Dev
                do
                    torch-lua testOneLayer_proto.lua \
                    -hiddenFactor $hiddenFactor \
                    -learningRate $learningRate \
                    -regCoeff $regCoeff \
                    -protoWeight $protoWeight\
                    -trainDev $trainDev \
		            -endLimit 10
                done
            done
        done
    done
done

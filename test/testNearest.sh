for trainDev in Train Dev Test
do
    for inputSize in 100 300
    do
        torch-lua printCos_nn.lua -trainDev $trainDev -inputSize $inputSize
    done
done

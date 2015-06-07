for trainDev in Train Dev Test
do
    for inputSize in 100 300
    do
        torch-lua newTest_os.lua -trainDev $trainDev -inputSize $inputSize
    done
done

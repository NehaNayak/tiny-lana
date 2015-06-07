require 'torch'
require 'optim'
require 'nn'
require 'torchnlp'

function cosine(v, w)
    return v:dot(w) / v:norm() / w:norm()
end

-- Command line arguments

cmd = torch.CmdLine()

cmd:option('-inputSize', 100, 'size of input layer')
cmd:option('-hiddenFactor', 1.0, 'relative size of hidden layer')
cmd:option('-learningRate', 0.5, 'learning rate')
cmd:option('-regCoeff', 0, 'regularisation coefficient')
cmd:option('-negSamples', 5, 'number of negative samples')
cmd:option('-endLimit', 100, 'maximum number of iterations with decreasing dev loss')
cmd:option('-pairSet', 'EH', 'which relation')
cmd:option('-trainDev', 'Train', 'which set to use')
cmd:option('-adaGrad', false, 'whether to use adaGrad')

cmd:option('-printFreq', 10e4, 'number of iterations after which to print loss')

cmdparams = cmd:parse(arg)

-- Load vectors
    emb_dir = '/scr/nayakne/wordvecs/glove/'
    emb_prefix = emb_dir .. 'glove.6B'
    emb_vocab, emb_vecs = torchnlp.read_embedding(
    emb_prefix .. '.vocab',
    emb_prefix .. '.' .. cmdparams.inputSize ..'d.th')

-- Create Train and Dev sets 

if cmdparams.adaGrad then opti='ag' else opti='sgd' end

modelPrefix = table.concat({
        '../params/olNS_',
        cmdparams.pairSet,
        '_i',
        cmdparams.inputSize,
        '_h',
        cmdparams.hiddenFactor,
        '_lr',
        cmdparams.learningRate,
        '_rc',
        cmdparams.regCoeff,
        '_ns',
        cmdparams.negSamples,
        '_el',
        cmdparams.endLimit,
        '_o',
        opti,
        '_vg'
        },"")

modelPath = modelPrefix .. '.th'        
tempModelPath = modelPrefix .. '_temp.th'


setPath = table.concat({
        '../pairs/pairs', 
        cmdparams.pairSet,
        '_',
        cmdparams.trainDev,
        '.txt'
        },"")
outPath = modelPrefix .. '_s' .. cmdparams.trainDev ..'.txt'

-- Define model
local m = torch.randn(cmdparams.inputSize):zero()
local outFile = assert(io.open(outPath,'w'))

local f = io.open(modelPath,'r')
if f==nil then 
    f = io.open(tempModelPath,'r')
    if f==nil then
        os.exit()
    else
        model = torch.load(tempModelPath)
    end
else
    model = torch.load(modelPath)
end


criterion = nn.MSECriterion()

-- Test
local f = assert(io.open(setPath, "r"))

devLoss = 0
devSize = 0


while true do

    line = f:read()
    if not line then break end

    for win,wout in string.gmatch(line, "(%S+)%s(%S+)") do

        local vin = emb_vecs[emb_vocab:index(win)]:typeAs(m)
        local vout = emb_vecs[emb_vocab:index(wout)]:typeAs(m)
        local myPrediction = model:forward(vin)
        local loss_x = criterion:forward(myPrediction,vout)
        devLoss =  devLoss + loss_x
        devSize = devSize + 1

	predictedCosine = cosine(myPrediction,vout)
	selfCosine = cosine(vin, vout)

	outputLine = table.concat({
	win,
	wout,
	predictedCosine,
	selfCosine
	},"\t")
	outFile:write(outputLine .. '\n')

    end
end

outFile:close()

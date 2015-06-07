require 'torch'
require 'optim'
require 'nn'
require 'torchnlp'

function cosine(v, w)
    return v:dot(w) / v:norm() / w:norm()
end

-- Command line arguments

cmd = torch.CmdLine()

cmd:option('-inputSize',100,'size of input layer')
cmd:option('-pairSet','EH','which relation')
cmd:option('-trainDev','Dev','which set to test')
cmd:option('-vecSet','g','glove vectors = \'g~~\'; word2vec vectors = \'v~~\'')

cmdparams = cmd:parse(arg)

if cmdparams.vecSet=='g' then
	emb_dir = '/scr/nayakne/wordvecs/glove/'
	emb_prefix = emb_dir .. 'glove.6B'
    emb_vocab, emb_vecs = torchnlp.read_embedding(
    emb_prefix .. '.vocab',
    emb_prefix .. '.' .. cmdparams.inputSize ..'d.th')
else
    emb_dir = '/scr/nayakne/wordvecs/word2vec/'
    emb_prefix = emb_dir .. 'wiki.bolt.giga5.f100.unk.neg5'
    emb_vocab, emb_vecs = torchnlp.read_embedding(
    emb_prefix .. '.vocab',
    emb_prefix .. '.' .. cmdparams.inputSize ..'.th')
end

-- Create Train and Dev sets 

setPath = table.concat({
        '../pairs/pairs', 
        cmdparams.pairSet,
        '_',
        cmdparams.trainDev,
        '.txt'
        },"")

modelPath = table.concat({
        '../params/os_', 
        cmdparams.pairSet,
        '_i',
        cmdparams.inputSize,
        '_v',
        cmdparams.vecSet,
        '.th'
        },"")

outPath = table.concat({
        '../params/os_', 
        cmdparams.pairSet,
        '_i',
        cmdparams.inputSize,
        '_v',
        cmdparams.vecSet,
        '_s',
        cmdparams.trainDev,
        '.txt'
        },"")

-- Define model
local m = torch.randn(cmdparams.inputSize):zero()
local outFile = assert(io.open(outPath,'w'))

offset = torch.load(modelPath)

-- Test
local f = assert(io.open(setPath, "r"))
while true do

    line = f:read()
    if not line then break end

    for win,wout in string.gmatch(line, "(%S+)%s(%S+)") do

        local vin = emb_vecs[emb_vocab:index(win)]:typeAs(m)
        local vout = emb_vecs[emb_vocab:index(wout)]:typeAs(m)
        local myPrediction = vin+offset

        predictedCosine = cosine(myPrediction,vout)
        selfCosine = cosine(vin, vout)

        outputLine = table.concat({
        win,
        wout,
        predictedCosine,
        selfCosine
        },"\t")
        
        outFile:write(outputLine .. "\n")

    end
end
outFile:close()

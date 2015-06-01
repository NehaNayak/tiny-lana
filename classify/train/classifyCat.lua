require 'torch'
require 'optim'
require 'nn'
require 'torchnlp'
require 'helpers'

-- Command line arguments

cmd = torch.CmdLine()

cmd:option('-inputSize',100,'size of input layer')
cmd:option('-hiddenFactor',1,'hiddenSize/inputSize')
cmd:option('-learningRate',0.5,'learning rate')
cmd:option('-regCoeff',0,'regularisation coefficient')
cmd:option('-endLimit',10,'maximum number of iterations with decreasing dev loss')
cmd:option('-pairSet','RNTN','which relation')
cmd:option('-vecSet','g.direct','glove vectors = \'g~~\'; word2vec vectors = \'v~~\'')
cmd:option('-printFreq',10e4,'number of iterations after which to print loss')
cmd:option('-model','af', 'which model to use')
cmdparams = cmd:parse(arg)

-- Load word embeddings

emb_vocab, emb_vecs = getEmbeddings()

-- Create Train and Dev sets 

devPath = '../pairs/pairs' .. cmdparams.pairSet .. '_Dev.txt'
dev = readSetCat(cmdparams.inputSize, devPath) 
dev_in=dev[1] ; dev_out=dev[2] ; devPairs=dev[3]


trainPath = '../pairs/pairs' .. cmdparams.pairSet .. '_Train.txt'
train = torch.load('../rntn/trainStuffCat.th')
--= readSet(cmdparams.inputSize, trainPath) 
train_in=train[1]:t() ; train_out=train[2]:t() ; trainPairs = train[3]

print('Loaded data')

outPrefix = table.concat({
        '../params/',
	    cmdparams.model,
	    'C_', 
        cmdparams.pairSet,
        '_i',
        cmdparams.inputSize,
        '_hf',
        cmdparams.hiddenFactor,
        '_lr',
        cmdparams.learningRate,
        '_rc',
        cmdparams.regCoeff,
        '_el',
        cmdparams.endLimit,
        '_v',
        cmdparams.vecSet,
        },"")

outPath = outPrefix .. '.th'
textOut =assert(io.open(outPrefix .. '.txt', "w"))

-- Define model

model = makeModel(cmdparams.model)
criterion = nn.ClassNLLCriterion()

-- Train

x, dl_dx = model:getParameters()

-- Define closure

feval = function()
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#train_in)[1] then _nidx_ = 1 end

   local input = train_in[_nidx_]:clone()
   local target = train_out[_nidx_]:clone()

   dl_dx:zero()

    local label = target[1]+2*target[2]
    pred = model:forward(input)
    local loss_x = criterion:forward(pred, label)
    loss_x = loss_x + cmdparams.regCoeff*torch.norm(x,2)^2/2
    model:backward(input, criterion:backward(pred, label))
    dl_dx:add(x:clone():mul(cmdparams.regCoeff))

   return loss_x, dl_dx

end

-- SGD

sgd_params = {
   learningRate = cmdparams.learningRate,
   learningRateDecay = 0,
   weightDecay = 0,
   momentum = 0
}

local prev_dev_loss=math.huge
local dev_loss=math.huge
bestLoss=math.huge

local prev_devError=math.huge
local devError=math.huge
bestDevError=math.huge

lessCount=0

for i = 1,10000 do

    train_loss = 0
    prev_model=model:clone()

    for j = 1,(#train_in)[1] do
        _,fs = optim.adagrad(feval,x,sgd_params)
        train_loss = train_loss + fs[1]
    end
    train_loss = train_loss / (#train_in)[1]
    
    prev_dev_loss=dev_loss
    dev_loss=0
    prev_devError = devError

    for j = 1,(#dev_in)[1] do
        local label = dev_out[j][1]+2*dev_out[j][2]
        if torch.norm(dev_in[j])>0 then
        local loss = criterion:forward(model:forward(dev_in[j]), label)
        dev_loss = dev_loss+loss
        end
    end
    dev_loss = dev_loss / (#dev_in)[1]
    devError = setError(model, dev_in, dev_out)

    if devError>prev_devError or (prev_devError - devError) < 10e-6 then
        if prev_devError<bestDevError then
            bestDevError = prev_devError
            bestModel=prev_model:clone()
            textOut:write("best model found at " .. lessCount .. "\n")
        end
        lessCount=lessCount+1
        if lessCount==cmdparams.endLimit then
            torch.save(outPath, bestModel)
            break
        end
    end

    if i%cmdparams.printFreq==0 then
        textOut:write('current train loss = ' .. train_loss .. "\n")
        textOut:write(setError(model, train_in, train_out) .. "\n")
        textOut:write('current dev loss = ' .. dev_loss .. "\n")
        textOut:write(setError(model, dev_in, dev_out) .. "\n")
    end
end


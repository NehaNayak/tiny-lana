require 'torch'
require 'optim'
require 'nn'
require 'torchnlp'
require 'ParallelCriterion'

-- Command line arguments

cmd = torch.CmdLine()

cmd:option('-inputSize', 100, 'size of input layer')
cmd:option('-learningRate', 0.5, 'learning rate')
cmd:option('-regCoeff', 0, 'regularisation coefficient')
cmd:option('-negSamples', 5, 'number of negative samples')
cmd:option('-endLimit', 100, 'maximum number of iterations with decreasing dev loss')
cmd:option('-pairSet', 'EH', 'which relation')
cmd:option('-adaGrad', false, 'whether to use adaGrad')

cmd:option('-printFreq', 10e4, 'number of iterations after which to print loss')

cmdparams = cmd:parse(arg)

-- Create Train and Dev sets 

trainPath = '../pairs/pairs'.. cmdparams.pairSet .. '_Train_i' ..cmdparams.inputSize .. '_vg.th'

train = torch.load(trainPath) 
train_in=train[1]
train_out=train[2]

devPath = '../pairs/pairs'.. cmdparams.pairSet .. '_Dev_i' ..cmdparams.inputSize .. '_vg.th'

dev = torch.load(devPath) 
dev_in=dev[1]
dev_out=dev[2]

-- Output files

if cmdparams.adaGrad then opti='ag' else opti='sgd' end

outPrefix = table.concat({
        '../params/afNS_',
        cmdparams.pairSet,
        '_i',
        cmdparams.inputSize,
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

bestOutPath = outPrefix .. '.th'        
tempOutPath = outPrefix .. '_temp.th'
textOut =assert(io.open(outPrefix .. '.txt', "w"))

-- Define model

model = nn.Sequential()                 
model:add(nn.Linear(cmdparams.inputSize, cmdparams.inputSize))

ptable = nn.ParallelTable()
ptable:add(model)
models = {}
for i=1,cmdparams.negSamples do
    models[i]=model:clone('weight','bias')
    ptable:add(models[i])
end

criterion = nn.ParallelCriterion()
directCriterion=nn.MSECriterion()
criterion:add(directCriterion,1)
criteria = {}
for i=1,cmdparams.negSamples do
    criteria[i]=nn.MSECriterion()
    criterion:add(criteria[i],-1)
end

-- Train

x, dl_dx = model:getParameters()

-- Define closure

feval = function()
	_nidx_ = (_nidx_ or 0) + 1
	if _nidx_ > (#train_in)[1] then _nidx_ = 1 end

	local directInput = train_in[_nidx_]:clone()
	local directTarget = train_out[_nidx_]:clone()

	inputs = {}
	targets = {}

	table.insert(inputs, directInput)
	table.insert(targets, directTarget)

    ridx=_nidx_
	for i = 1,cmdparams.negSamples do
		while ridx==_nidx_ do ridx = math.random((#train_in)[1]) end
		table.insert(inputs, directInput)
		table.insert(targets, train_in[ridx]:clone())
	end

	preds = ptable:forward(inputs)
	local loss_x = criterion:forward(preds, targets)
	loss_x = loss_x +cmdparams.regCoeff*torch.norm(x,2)^2/2

	dl_dx:zero()
	ptable:backward(inputs, criterion:backward(preds, targets))
	dl_dx:add(x:clone():mul(cmdparams.regCoeff))

	return loss_x, dl_dx

end

function calculateDevLoss()
    dev_loss=0
    for j = 1,(#dev_in)[1] do
	    local directInput = dev_in[j]:clone()
	    local directTarget = dev_out[j]:clone()

	    inputs = {}
	    targets = {}

	    table.insert(inputs, directInput)
	    table.insert(targets, directTarget)

	    for i = 1,cmdparams.negSamples do
        ridx=j
		while ridx==j do ridx = math.random((#dev_in)[1]) end
		table.insert(inputs, dev_in[ridx]:clone())
		table.insert(targets, dev_out[ridx]:clone())
	end

	preds = ptable:forward(inputs)
	local loss_x = criterion:forward(preds, targets)
	loss_x = loss_x +cmdparams.regCoeff*torch.norm(x,2)^2/2
     dev_loss = dev_loss+loss_x
    end
    dev_loss = dev_loss / (#dev_in)[1]
    return dev_loss

end

-- SGD

sgd_params = {
   learningRate = cmdparams.learningRate,
}

local prev_dev_loss=math.huge
local dev_loss=math.huge
bestLoss=math.huge
lessCount=0

for i = 1, 10000 do

    train_loss = 0
    prev_model=model:clone()
  
    for j = 1,(#train_in)[1] do
	if cmdparams.adaGrad then
        _,fs = optim.adagrad(feval,x,sgd_params)
	else
        _,fs = optim.sgd(feval,x,sgd_params)
	end
        train_loss = train_loss + fs[1]
    end
    train_loss = train_loss / (#train_in)[1]
    
    prev_dev_loss=dev_loss
    dev_loss=0

    dev_loss = calculateDevLoss()
    
    if dev_loss>prev_dev_loss then
        if prev_dev_loss<bestLoss then
            bestLoss = prev_dev_loss
            bestModel=prev_model:clone()
            torch.save(tempOutPath, bestModel)
            textOut:write("best model found at " .. lessCount .. "\n")
        end
        lessCount=lessCount+1
        if lessCount==cmdparams.endLimit then
            torch.save(bestOutPath, bestModel)
            break
        end
    end

    if i%cmdparams.printFreq==0 then
        textOut:write('current train loss = ' .. train_loss .. "\n")
        textOut:write('current dev loss = ' .. dev_loss .. "\n")
    end
end

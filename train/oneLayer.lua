require 'torch'
require 'optim'
require 'nn'
require 'torchnlp'

-- Command line arguments

cmd = torch.CmdLine()

cmd:option('-inputSize',100,'size of input layer')
cmd:option('-hiddenSize',100,'size of hidden layer')
cmd:option('-learningRate',0.5,'learning rate')
cmd:option('-regCoeff',0,'regularisation coefficient')
cmd:option('-endLimit',100,'maximum number of iterations with decreasing dev loss')
cmd:option('-pairSet','EH','which relation')
cmd:option('-adaGrad',false,'whether to use adaGrad')

cmd:option('-printFreq',10e4,'number of iterations after which to print loss')
cmdparams = cmd:parse(arg)

-- Create Train and Dev sets 

trainPath = table.concat({
        '../pairs/pairs', 
        cmdparams.pairSet,
        '_Train_i',
        cmdparams.inputSize,
        '_vg.th'
        },"")

train = torch.load(trainPath) 
train_in=train[1]
train_out=train[2]

devPath = table.concat({
        '../pairs/pairs', 
        cmdparams.pairSet,
        '_Dev_i',
        cmdparams.inputSize,
        '_vg.th'
        },"")

dev = torch.load(devPath) 
dev_in=dev[1]
dev_out=dev[2]

if cmdparams.adaGrad then
 opti='ag'
else
 opti='sgd'
end

tempOutPath = table.concat({
        '../params/ol_', 
        cmdparams.pairSet,
        '_i',
        cmdparams.inputSize,
        '_h',
        cmdparams.hiddenSize,
        '_lr',
        cmdparams.learningRate,
        '_el',
        cmdparams.endLimit,
	'_',
	opti,
        '_vg_temp.th'
        },"")


outPath = table.concat({
        '../params/ol_', 
        cmdparams.pairSet,
        '_i',
        cmdparams.inputSize,
        '_h',
        cmdparams.hiddenSize,
        '_lr',
        cmdparams.learningRate,
        '_el',
        cmdparams.endLimit,
	'_',
	opti,
        '_vg.th'
        },"")

-- Define model

model = nn.Sequential()                 
model:add(nn.Linear(cmdparams.inputSize, cmdparams.hiddenSize)) 
model:add(nn.Tanh())
model:add(nn.Linear(cmdparams.hiddenSize, cmdparams.inputSize))

criterion = nn.CosineEmbeddingCriterion()

-- Train

x, dl_dx = model:getParameters()

-- Define closure

feval = function()
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#train_in)[1] then _nidx_ = 1 end

   local input = train_in[_nidx_]:clone()
   local target = train_out[_nidx_]:clone()

   dl_dx:zero()

   local loss_x = criterion:forward({model:forward(input), target},1)
   loss_x=loss_x+cmdparams.regCoeff*torch.norm(x,2)^2/2
   model:backward(input, criterion:backward({model.output, target},1)[1])
   dl_dx:add(x:clone():mul(cmdparams.regCoeff))

   return loss_x, dl_dx

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

    for j = 1,(#dev_in)[1] do
        local loss = criterion:forward({model:forward(dev_in[j]), dev_out[j]},1)
        dev_loss = dev_loss+loss
    end
    dev_loss = dev_loss / (#dev_in)[1]
    
    if dev_loss>prev_dev_loss then
        if prev_dev_loss<bestLoss then
            bestLoss = prev_dev_loss
            bestModel=prev_model:clone()
            torch.save(tempOutPath, bestModel)
            print("best model found at " .. lessCount)
        end
        lessCount=lessCount+1
        if lessCount==cmdparams.endLimit then
            torch.save(outPath, bestModel)
            break
        end
    end

    if i%cmdparams.printFreq==0 then
        print('current train loss = ' .. train_loss)
        print('current dev loss = ' .. dev_loss)
    end
end

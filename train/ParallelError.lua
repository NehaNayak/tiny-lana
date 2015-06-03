require 'torch'
require 'optim'
require 'nn'
require 'torchnlp'
require('ParallelCriterion')

-- Define model

model = nn.Sequential()                 
model:add(nn.Linear(100,100)) 
model:add(nn.Tanh())
model:add(nn.Linear(100,100))

ptable = nn.ParallelTable()
ptable:add(model)

-- Define criterion

criterion = nn.ParallelCriterion()
directCriterion=nn.CosineEmbeddingCriterion()
criterion:add(directCriterion,1)

-- Data

input = torch.rand(100)
target = torch.rand(100)

preInputs = {}
table.insert(preInputs, input)

preds = ptable:forward(preInputs)

inputs = {}
targets = {}
table.insert(inputs, {preds[1],target})
table.insert(targets, 1)

-- next two lines execute fine
directCriterion:forward(inputs[1],targets[1])
directCriterion:backward(inputs[1],targets[1])

-- backward causes an error belo
criterion:forward(inputs, targets)
criterion:backward(inputs, targets)



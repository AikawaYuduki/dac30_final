import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.cuda import to_cpu
import pandas as pd
import random
import msvcrt
from santa import Santa, SantaE
from eve import Eve

n_epoch = 1000

print("which? full:f mini:m")
while True:
    if msvcrt.kbhit():
        kb = msvcrt.getch()
        if kb.decode() == "m":
            inp_file = "tv_dl_input_1000.csv"
            oup_file = "tv_dl_output_1000.csv"
            res_file = "result"
            train_size = 850
            batch_size = 32
            break
        if kb.decode() == "f":
            inp_file = "tv_dl_input.csv"
            oup_file = "tv_dl_output.csv"
            res_file = "result"
            train_size = 95000
            batch_size = 128
            break


inp = pd.read_csv(inp_file)
oup = pd.read_csv(oup_file)

X = inp.fillna(0)
Y = oup.fillna(0)
X = X.values.astype(np.float32)
Y = Y.values.astype(np.float32)
Y = np.reshape(Y,(X.shape[0],25))

train,test = datasets.split_dataset_random(chainer.datasets.TupleDataset(X,Y),train_size)
train_iter = chainer.iterators.SerialIterator(train, batch_size)
test_iter = chainer.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=True)

class DMMChain(Chain):
    def __init__(self):
        super(DMMChain,self).__init__()
        with self.init_scope():
            self.l1=L.Linear(None,128)
            self.l2=L.Linear(128,256)
            self.l3=L.Linear(256,25)
            

    def __call__(self,x):
        h = F.leaky_relu(self.l1(x))
        #h = F.dropout(F.leaky_relu(self.l2(h)))
        #h = F.dropout(F.leaky_relu(self.l3(h)))
        h = F.leaky_relu(self.l2(h))
        h = F.leaky_relu(self.l3(h))
        h1 = h.data
        h1[:,0] = F.relu(h1[:,0]).data
        h1[:,1:3] = F.sigmoid(h1[:,1:3]).data
        h1[:,3:9] = F.softmax(h1[:,3:9]).data
        h1[:,9:25] = F.softmax(h1[:,9:25]).data
        return Variable(h1)

model = L.Classifier(DMMChain(),lossfun=F.mean_squared_error)
model.compute_accuracy = False
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.005))
optimizer.add_hook(chainer.optimizer_hooks.Lasso(0.005))
updater = training.StandardUpdater(train_iter,optimizer,device=-1)
trainer = training.Trainer(updater,(n_epoch,"epoch"),out=res_file)

trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename="snapshot_epoch-{.updater.epoch}"),trigger=(100,'epoch'))
trainer.extend(extensions.Evaluator(test_iter,model,device=-1))
trainer.extend(extensions.PrintReport(["epoch","main/loss","validation/main/loss","elapsed_time"]))
trainer.extend(extensions.PlotReport(["main/loss","validation/main/loss"],x_key="epoch",file_name=res_file,trigger=training.triggers.ManualScheduleTrigger(list(range(10,n_epoch,(n_epoch//1000)+1)),'epoch')))
trainer.extend(extensions.dump_graph("main/loss"))

print("which? train:t predict:p")
while True:
    if msvcrt.kbhit():
        kb = msvcrt.getch()
        if kb.decode() == "t":
            trainer.run()
            break
        elif kb.decode() == "p":
            inf_net = L.Classifier(DMMChain(),lossfun=F.mean_squared_error)
            serializers.load_npz(
                res_file + '/snapshot_epoch-1000',
                inf_net, path='updater/model:main/predictor/',strict = False)
            out = oup
            test_data = inp.fillna(0)
            test_data = np.asarray(test_data).astype(np.float32)
            for i in range(inp.shape[0]):
                x = test_data[i,:]
                x = x.reshape((1,-1))
                x = chainer.Variable(x)
                y = model.predictor(x)
                #print(y.data)
                out.iloc[i,:] = y.data[0]
                if i % 1000 == 0:
                    print(i)

            out.to_csv("setsumei.csv")
            break

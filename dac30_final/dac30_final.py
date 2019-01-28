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
from chainer.dataset import iterator as iterator_module
from chainer.dataset import convert
from chainer import optimizer as optimizer_module
import pandas as pd
import random
import msvcrt
from santa import Santa, SantaE
from eve import Eve

n_epoch = 500

print("which? full:f mini:m")
while True:
    if msvcrt.kbhit():
        kb = msvcrt.getch()
        if kb.decode() == "m":
            inp_file = "tv_dl_input_1000.csv"
            oup_file = "tv_dl_output_1000.csv"
            res_file = "result_test"
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
test_iter = chainer.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

class DMMChain(Chain):
    def __init__(self):
        super(DMMChain,self).__init__()
        with self.init_scope():
            self.l1=L.Linear(None,128)
            self.l2=L.Linear(128,256)
            

    def __call__(self,x):
        h = F.leaky_relu(self.l1(x))
        h = F.dropout(F.leaky_relu(self.l2(h)))
        return h

class WatchChain(Chain):
    def __init__(self):
        super(WatchChain,self).__init__()
        with self.init_scope():
            self.l3=L.Linear(256,256)
            self.l4=L.Linear(256,1)
            

    def __call__(self,x):
        h = F.dropout(F.leaky_relu(self.l3(x)))
        h = F.leaky_relu(self.l4(h))
        return h

class ManChain(Chain):
    def __init__(self):
        super(ManChain,self).__init__()
        with self.init_scope():
            self.l3=L.Linear(256,256)
            self.l4=L.Linear(256,1)
            

    def __call__(self,x):
        h = F.dropout(F.leaky_relu(self.l3(x)))
        h = F.leaky_relu(self.l4(h))
        return F.sigmoid(h)

class MaryChain(Chain):
    def __init__(self):
        super(MaryChain,self).__init__()
        with self.init_scope():
            self.l3=L.Linear(256,256)
            self.l4=L.Linear(256,1)
            

    def __call__(self,x):
        h = F.dropout(F.leaky_relu(self.l3(x)))
        h = F.leaky_relu(self.l4(h))
        return F.sigmoid(h)

class OldChain(Chain):
    def __init__(self):
        super(OldChain,self).__init__()
        with self.init_scope():
            self.l3=L.Linear(256,256)
            self.l4=L.Linear(256,6)
            

    def __call__(self,x):
        h = F.dropout(F.leaky_relu(self.l3(x)))
        h = F.leaky_relu(self.l4(h))
        return F.softmax(h)

class JobChain(Chain):
    def __init__(self):
        super(JobChain,self).__init__()
        with self.init_scope():
            self.l3=L.Linear(256,256)
            self.l4=L.Linear(256,16)
            

    def __call__(self,x):
        h = F.dropout(F.leaky_relu(self.l3(x)))
        h = F.leaky_relu(self.l4(h))
        return F.softmax(h)

class MyUpdater(training.StandardUpdater):
    def __init__(self, iterator, base, base_optimizer, classifiers, cl_optimizers, converter=convert.concat_examples, device=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main':iterator}
        self._iterators = iterator
        self.base = base
        self.classifiers = classifiers

        self._optimizers = {}
        self._optimizers["base_opt"] = base_optimizer
        for i in range(0, len(cl_optimizers)):
            self._optimizers[str(i)] = cl_optimizers[i]

        self.converter = convert.concat_examples
        self.device = device
        self.iteration = 0

    def update_core(self):
        iterator = self._iterators['main'].next()
        in_arrays = self.converter(iterator, self.device)

        xp = np if int(self.device) == -1 else cuda.cupy
        x_batch = xp.array(in_arrays[0])
        t_batch = xp.array(in_arrays[1])
        y = self.base(x_batch)

        loss_dic = {}
        loss = self.classifiers[0](y, t_batch[:,0])
        loss_dir["0"] = loss

        loss = self.classifiers[1](y, t_batch[:,1])
        loss_dir["1"] = loss

        loss = self.classifiers[2](y, t_batch[:,2])
        loss_dir["2"] = loss

        loss = self.classifiers[3](y, t_batch[:,3:9])
        loss_dir["3"] = loss

        loss = self.classifiers[4](y, t_batch[:,9:])
        loss_dir["4"] = loss

        for name, optimizer in items(self._optimizers):
            optimizer.target.cleargrads()

        for name, loss in items(loss_dic):
            loss.backward()

        for name, optimizer in items(self._optimizers):
            optimizer.update()


model_base = L.Classifier(DMMChain(),lossfun=F.mean_squared_error)
model_watch = L.Classifier(WatchChain(),lossfun=F.mean_squared_error)
model_man = L.Classifier(ManChain(),lossfun=F.sigmoid_cross_entropy)
model_mary = L.Classifier(MaryChain(),lossfun=F.sigmoid_cross_entropy)
model_old = L.Classifier(OldChain(),lossfun=F.softmax_cross_entropy)
model_job = L.Classifier(JobChain(),lossfun=F.softmax_cross_entropy)

model_base.compute_accuracy = False
model_watch.compute_accuracy = False
model_man.compute_accuracy = False
model_mary.compute_accuracy = False
model_old.compute_accuracy = False
model_job.compute_accuracy = False

optimizer_base = chainer.optimizers.Adam()
optimizer_watch = chainer.optimizers.Adam()
optimizer_man = chainer.optimizers.Adam()
optimizer_mary = chainer.optimizers.Adam()
optimizer_old = chainer.optimizers.Adam()
optimizer_job = chainer.optimizers.Adam()

optimizer_base.setup(model_base)
optimizer_watch.setup(model_base)
optimizer_man.setup(model_base)
optimizer_mary.setup(model_base)
optimizer_old.setup(model_base)
optimizer_job.setup(model_base)

optimizer_base.add_hook(chainer.optimizer.WeightDecay(0.005))
optimizer_watch.add_hook(chainer.optimizer.WeightDecay(0.005))
optimizer_man.add_hook(chainer.optimizer.WeightDecay(0.005))
optimizer_mary.add_hook(chainer.optimizer.WeightDecay(0.005))
optimizer_old.add_hook(chainer.optimizer.WeightDecay(0.005))
optimizer_job.add_hook(chainer.optimizer.WeightDecay(0.005))

optimizer_base.add_hook(chainer.optimizer_hooks.Lasso(0.005))
optimizer_watch.add_hook(chainer.optimizer_hooks.Lasso(0.005))
optimizer_man.add_hook(chainer.optimizer_hooks.Lasso(0.005))
optimizer_mary.add_hook(chainer.optimizer_hooks.Lasso(0.005))
optimizer_old.add_hook(chainer.optimizer_hooks.Lasso(0.005))
optimizer_job.add_hook(chainer.optimizer_hooks.Lasso(0.005))

classifiers = [model_watch, model_man, model_mary, model_old, model_job]
cl_optimizers = [optimizer_watch, optimizer_man, optimizer_mary, optimizer_old, optimizer_job]

updater = MyUpdater(train_iter,model_base,optimizer_base,classifiers,cl_optimizers,device=-1)
trainer = training.Trainer(updater,(n_epoch,"epoch"),out=res_file)

trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename="snapshot_epoch-{.updater.epoch}"),trigger=(100,'epoch'))
#trainer.extend(extensions.Evaluator(test_iter,model,device=-1))
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

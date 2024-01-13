from mxnet.metric import MAE
from mxnet import npx, gluon, autograd
import time, os
import numpy as np

def train(model, data_train_ld, data_val_ld, configs):
    device = npx.gpu() if npx.num_gpus() > 0 else npx.cpu()
    model.initialize(ctx = device)
    model.hybridize()

    l2loss = gluon.loss.L2Loss()
    trainer = gluon.Trainer(model.collect_params(), configs["train"]["optim_method"], {'learning_rate': configs["train"]["learning_rate"]})

    print(f"____Start training on {device}____")

    for epoch in range(configs["train"]["num_epoch"] + 1):
      train_loss = 0.0
      tic = time.time()
      
      for data, label in data_train_ld:
        
        with autograd.record():
          output = model(data)
          loss = l2loss(output, label)
        
        autograd.backward(loss)
        trainer.step(configs["train"]["batch_size"])
        train_loss += loss.mean().sqrt().asscalar()
      
      toc = time.time()
      val_error = test(model, data_val_ld, configs)[3]

      if (epoch % 10 == 0):
        print('Epoch %d: train_loss %.3f - val_accuracy %.3f, in %.1f sec' %(epoch, train_loss/len(data_train_ld), val_error, toc - tic))

    print(f"____Stop training on {device}____")
    model.save_parameters(os.path.join(os.getcwd(), "pretrain", "model.params"))

def test(model, data_test_ld, configs, time_stamps = None):
    mean_absolute_error = MAE()
    mean_absolute_error.reset()
    truths = []
    preds = []
    for data, label in data_test_ld:
        predict = model(data)
        truth = label
        preds.append(predict)
        truths.append(truth)
    mean_absolute_error.update(truths, preds)
    error = mean_absolute_error.get()[1]

    if not time_stamps is None:
      select_stamp_idx = np.arange(configs["model"]["window_size"] + 1, len(time_stamps.tolist()), configs["model"]["window_size"])
      times = time_stamps[select_stamp_idx]
    else:
       times = None
    
    return (preds, truths, times, error)
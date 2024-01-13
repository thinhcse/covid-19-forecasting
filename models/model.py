from mxnet.gluon import nn, rnn

window_size = 7

class encoder(nn.HybridBlock):

  def __init__(self):
    super(encoder, self).__init__()

    with self.name_scope():
        self.conv = nn.Conv2D(channels = 10, kernel_size= 3)
        self.flat1 = nn.Flatten()
        self.lstm = rnn.LSTM(hidden_size=5, num_layers=2, layout='NTC')
        self.flat2 = nn.Flatten()
        self.drop = nn.Dropout(0.2)

        self.fc1 = nn.Dense(units=2, activation='relu')
        self.fc2 = nn.Dense(units=3, activation='sigmoid')

  def hybrid_forward(self, F, x):
    output = self.conv(x)
    output = self.flat1(output)
    output = F.reshape(output, (0, 10, -1))
    output = self.lstm(output)
    output = self.flat2(output)
    output = self.drop(output)

    output = self.fc1(output)
    output = self.fc2(output)

    output = F.reshape(output, (0, 1, -1))

    return output


class decoder(nn.HybridBlock):

  def __init__(self):
    super(decoder, self).__init__()

  def hybrid_forward(self, F, x, y):
    N = F.reshape(F.sum_axis(F.slice_axis(y, axis=2, begin=0, end=3), axis = 2), (0, 1, -1))
    S = F.elemwise_sub(F.slice_axis(y, axis = 2, begin = 0, end = 1),
                          F.elemwise_div(F.elemwise_mul(F.elemwise_mul(F.slice_axis(x, axis = 2, begin = 0, end = 1),
                                                                          F.slice_axis(y, axis = 2, begin = 0, end = 1)), F.slice_axis(y, axis = 2, begin = 1, end = 2)), N))
    I = F.elemwise_add(F.slice_axis(y, axis = 2, begin = 1, end = 2),
                        F.elemwise_sub(
                            F.elemwise_sub(
                                F.elemwise_div(
                                    F.elemwise_mul(
                                        F.elemwise_mul(
                                            F.slice_axis(x, axis = 2, begin = 0, end = 1), F.slice_axis(y, axis = 2, begin = 0, end = 1)),
                                            F.slice_axis(y, axis = 2, begin = 1, end = 2)),N), F.elemwise_mul(
                                                F.slice_axis(x, axis = 2, begin = 1, end = 2), F.slice_axis(y, axis= 2, begin = 1, end = 2))), F.elemwise_mul(
                                                    F.slice_axis(x, axis = 2, begin = 2, end = None), F.slice_axis(y, axis = 2, begin = 1, end = 2)
                                                )))
    R = F.elemwise_add(F.slice_axis(y, axis = 2, begin = 2, end = 3), F.elemwise_mul(F.slice_axis(x, axis = 2, begin = 1, end = 2), F.slice_axis(y, axis= 2, begin = 1, end = 2)))
    D = F.elemwise_add(F.slice_axis(y, axis = 2, begin = 3, end = None), F.elemwise_mul(F.slice_axis(x, axis = 2, begin = 2, end = None), F.slice_axis(y, axis = 2, begin = 1, end = 2)))
    
    return F.reshape(F.concat(S, I, R, D, dim = 2), (0, -1))

class model(nn.HybridSequential):
  def __init__(self):
    super(model, self).__init__()
    with self.name_scope():
      self.window_size = 7
      self.encoder = encoder()
      self.decoder = decoder()

  def set_configs(self, configs):
    self.window_size = configs["model"]["window_size"]
    return self

  def hybrid_forward(self, F, data):
    x = F.slice_axis(data, axis = 2, begin = 4, end = None)
    x = F.slice_axis(x, axis = 1, begin = 0, end = -1)
    x = F.reshape(x, (0, 1, self.window_size - 1, -1))
    y = F.slice_axis(data, axis = 2, begin = 0, end = 4)
    y = F.slice_axis(y, axis = 1, begin = -1, end = None)

    output = self.encoder(x)
    output = self.decoder(output, y)

    return output
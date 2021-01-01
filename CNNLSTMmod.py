#ConvLSTM to predict n steps ahead... Not sure how it will react when less than n steps are available during training


#https://raw.githubusercontent.com/ndrplz/ConvLSTM_pytorch/master/convlstm.py

#imports
import torch.nn as nn
import torch

class cnnLSTM(nn.Module):
    #not exactly a cnnLSTM but a convLSTM - giving time steps to a cnn
    def __init__(self, inp, hid, ks, bias):
        super(cnnLSTM, self).__init__()

        self.inp = inp #input dimensions
        self.hid = hid #hidden dimensions

        self.kernel_size = ks
        self.padding = ks[0] // 2, ks[1] // 2
        self.bias = bias

        #CNN
        self.conv = nn.Conv2d(in_channels=self.inp + self.hid,
                              out_channels = 4 * self.hid,
                              kernel_size=self.kernel_size,
                              padding = self.padding,
                              bias = self.bias
                              )

        def forward(self, input_tensor, cur_state):
            h_cur, c_cur = cur_state

            com = torch.cat([input_tensor,h_cur], dim= 1) #combined channel

            cc = self.conv(com) #shoving the combined channel into cnn

            cc_i, cc_f, cc_o, cc_g = torch.split(cc, self.hid, dim=1) #seperating the resulting values from the cnn
            i = torch.sigmoid(cc_i)
            f = torch.sigmoid(cc_f)
            o = torch.sigmoid(cc_o)
            g = torch.tanh(cc_g) #not really sure if this should be tan

            c_next = f * c_cur + i * g #getting the next step
            h_next = o * torch.tanh(c_next)

            return h_next, c_next

        def init_hidden(self,batch_size, image_size):
            #batch size is the number of frames we're passing in to make the predictions on

            height, width = image_size
            return (torch.zeros(batch_size, self.hid, height, width, device=self.conv.weight.device),
                    torch.zeros(batch_size, self.hid, height, width, device = self.conv.weight.device)) #trying to grab the weight from the conv2d



#Use the convLSTM above to create an input for 3dCNN
#idea is: predict next frame then shove it back in to get n more predictions - not sure if this will be very good0
class EncoderDecoder(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoder, self).__init__()

        self.e1_cnn = cnnLSTM(input_dim = in_chan,
                              hidden_dim = nf,
                              kernel_size=(3,3),
                              bias=True)
        self.e2_cnn = cnnLSTM(input_dim = in_chan,
                              hidden_dim=nf,
                              kernel_size=(3,3),
                              bias=True)
        self.d1_cnn = cnnLSTM(input_dim = nf,
                              hidden_dim = nf,
                              kernel_size=(3,3),
                              bias=True)
        self.d2_cnn = cnnLSTM(in_channels=nf,
                              out_channels=nf,
                              kernel_size=(3,3),
                              bias = True)
        self.d3CNN = nn.conv3d(in_channels = nf,
                               out_channels = 1,
                               kernel_size = (1,3,3),
                               padding = (0,1,1))

    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        #encode
        for t in range(seq_len):
            h_t, c_t = self.e1_cnn(input_tensor = x[:,t,:,:],
                                   cur_state = [h_t, c_t])
            h_t2, c_t2 = self.e2_cnn(input_tensor = h_t,
                                     cur_state=[h_t2,c_t2])

        encode_v = h_t2

        #decoder
        for t in range(future_step):
            h_t3, c_t3 = self.d1_cnn(input_tensor=encode_v, cur_state = [h_t3,c_t3])
            h_t4, c_t4 = self.d2_cnn(input_tensor = h_t3, cur_state = [h_t4, c_t4])

            encode_v = h_t4
            outputs += [h_t4] #these are the predictions

        outputs = torch.stack(outputs, 1) #stacking all the outputs into 1
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.d3CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x, future_seq=0, hidden_state=None):
        #input tensor should be of shape: batch, time, channel, height, width

        b, seq_len, _, h, w = x.size()

        #make the hidden states
        h_t, c_t = self.e1_cnn.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.e2_cnn.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.d1_cnn.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.d2_cnn.init_hidden(batch_size=b, image_size=(h, w))

        #get outputs from the encoder
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs


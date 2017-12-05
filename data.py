import numpy as np
import scipy.io as sio

class MNISTDataHandler(object):
  """
    Members :
      is_train - Options for sampling
      path - MNIST data path
      data - a list of np.array w/ shape [batch_size, 28, 28, 1]
  """
  def __init__(self, is_train):
    self.is_train = is_train
    self.data =self._get_data()

  def _get_data(self):
     
    b=[]
    d=[]
    data=[]
    for i in range (10):
        a=sio.loadmat('/home/yudeliang/Desktop/DIRNet/MNIST_data/phase'+str(i+1)+'.mat')
        b.append(a)

     
    for i in range(10):
        if i%2==0:
           d.append(b[i]['CT_phase1']) 
        if i%2==1:
           d.append(b[i]['CT_phase2']) 
    

    c=np.zeros([10,256,512])


    for i in range (272):
        for j in range (10):
            c[j,:,:] = d[j][100:356,:,i]

        e=np.reshape(c,[10,256,512,1])
        g=np.ones([10,256,512,1])
        e1=np.amin(e)
        e2=np.amax(e)

        h=(e-e1*g)/(e2*g-e1*g)

        data.append(h)
    
    return data

  def sample_pair(self, batch_size, label=None):
    label = np.random.randint(272)
    images = self.data[label]
    
    choice1 = np.random.choice(images.shape[0], batch_size)
    choice2 = np.random.choice(images.shape[0], batch_size)
    x = images[choice1]
    y = images[choice2]
    
    return x, y

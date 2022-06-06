import torch
import warnings
warnings.filterwarnings("ignore")
from utils.arguments import parse_arguments
from utils.gaussian import gaussian_blur
import matplotlib.pylab as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import kornia as K


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def prepare_baseline_input_grad(data,base,n_step):
        pass
        
        num_sample = 10
        # Generate m_steps intervals for integral_approximation() below
        alphas = torch.linspace(start=0.0, end=1.0, steps=n_step).unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(0)
        
        # base_line = gaussian_blur(data,(7,7), (1, 50))
        
        #interpolate now
        base_line = base
        data = data.unsqueeze(dim=1)
        base_line = base_line.unsqueeze(dim=1)
        # base_line = torch.zeros_like(data)
        
        # print("baseline",base_line.size())
        # data,base_line,alphas = data.to(device), base_line.to(device), alphas.to(device)
        new_data = data - base_line
        # print(alphas.size(),base_line.size(),new_data.size())
        imgs = alphas * new_data+ base_line 
        print(imgs.size())
        imgs = imgs.view(num_sample*n_step,data.size()[2],data.size()[3],data.size()[4])
        return imgs

if __name__ == "__main__":

    d = np.load('sample/000009.npy',allow_pickle = True)
    d=d[np.newaxis,:,:]
    base = gaussian_filter(d, sigma=15)
    # print(d.shape)
    # m = torch.nn.ZeroPad2d(2)
    
    data = torch.from_numpy(d).unsqueeze(0)
    print(data.size())
    x_blur = K.filters.gaussian_blur2d(data.float(), (51,51), (50.0, 50.0))
    print(x_blur.size())
    plt.imshow(x_blur.numpy()[0].transpose([1,2,0]))
    plt.show()
#     base = torch.from_numpy(base).unsqueeze(0)
#     # print(data.size())
#     # data = m(data)
#     # print(data.size())
#     a = prepare_baseline_input_grad(data[:,:,:224,:224],base[:,:,:224,:224],30)
#     print(data.size())
#     plt.subplot(221)
#     plt.imshow(a[0].numpy().transpose([1,2,0]))
#     plt.subplot(222)
#     plt.imshow(a[10].numpy().transpose([1,2,0]))
#     plt.subplot(223)
#     plt.imshow(a[20].numpy().transpose([1,2,0]))
#     plt.subplot(224)
#     plt.imshow(a[40].numpy().transpose([1,2,0]))
#     plt.show()
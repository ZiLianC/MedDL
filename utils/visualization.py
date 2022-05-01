import matplotlib.pyplot as plt
import torch
import numpy as np
def print_cut_samples(data,label,predict,save_path):
    '''z = np.any(torch.argmax(predict, dim=1).detach().cpu()[0,:,:,:], axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]
    cut=abs(start_slice-end_slice)/2'''
    cut = 39
    data0=data.cpu().numpy()[0, 0, :, :, cut]
    label0=label.cpu().numpy()[0, 0, :, :, cut]
    pred0=torch.argmax(predict, dim=1).detach().cpu()[0, :, :, cut]
    '''plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("image")
    plt.imshow(data, cmap="gray")
    plt.imshow(label,alpha=0.6)
    plt.imshow(contour[0],alpha=0.6)
    plt.savefig(save_path,dpi=300)'''
    np.save(save_path+"data66",data0)
    np.save(save_path+"label66",label0)
    np.save(save_path+"pred66",pred0)
    cut = 60
    data1=data.cpu().numpy()[0, 0, :, :, cut]
    label1=label.cpu().numpy()[0, 0, :, :, cut]
    pred1=torch.argmax(predict, dim=1).detach().cpu()[0, :, :, cut]
    '''plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("image")
    plt.imshow(data, cmap="gray")
    #plt.imshow(label,alpha=0.6)
    plt.imshow(contour[0],alpha=0.6)
    plt.savefig(save_path,dpi=300)'''
    np.save(save_path+"data60",data1)
    np.save(save_path+"label60",label1)
    np.save(save_path+"pred60",pred1)
    cut = 71
    data2=data.cpu().numpy()[0, 0, :, :, cut]
    label2=label.cpu().numpy()[0, 0, :, :, cut]
    pred2=torch.argmax(predict, dim=1).detach().cpu()[0, :, :, cut]
    '''plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("image")
    plt.imshow(data, cmap="gray")
    #plt.imshow(label,alpha=0.6)
    plt.imshow(contour[0],alpha=0.6)
    plt.savefig(save_path,dpi=300)'''
    np.save(save_path+"data69",data2)
    np.save(save_path+"label69",label2)
    np.save(save_path+"pred69",pred2)



def print_heatmap(data,label,predict,grad_graph,save_path):
    '''z = np.any(torch.argmax(predict, dim=1).detach().cpu(), axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]
    cut=abs(start_slice-end_slice)/2'''
    cut=66
    plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("image")
    plt.imshow(data.cpu().numpy()[0, 0, :, :, cut], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("label")
    plt.imshow(label.cpu().numpy()[0, 0, :, :, cut])
    plt.subplot(1, 3, 3)
    plt.title("heat")
    plt.imshow(grad_graph[0, :, :, cut])
    plt.savefig(save_path,dpi=300)
    




import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, CenteredNorm
from matplotlib.ticker import MaxNLocator
from .geom import WaveGeometryMs, WaveGeometry
from .solver import MMSolver
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore", message=".*No contour levels were found.*")


mpl.use('Agg',) # uncomment for plotting without GUI
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 600


def plot_loss(loss_iter, plotdir):
    fig = plt.figure()
    plt.plot(loss_iter, 'o-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir+'loss.png',dpi=600)
    plt.close(fig)
    
def plot_accuracy(accuracy_iter, plotdir):
    fig = plt.figure()
    plt.plot(accuracy_iter, 'o-')
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir+'training_accu.png',dpi=600)
    plt.close(fig)

def plot_output(u, p, num, batch, epoch, plotdir,thresh_line):
    ''' u: intensity (y-value)
    p: corresponding label
    num: sequence number of the signal within this batch'''
    fig = plt.figure()
    plt.bar(range(0,u.size()[0]), u.detach().cpu().squeeze(), color='k', label=f'label: {p}')
    #plt.bar(p, u.detach().cpu().squeeze()[p], color='r')
    #------------------------------------NEW--------------------------------------------------#
    for i in range(len(thresh_line)):
        if thresh_line[i] == True:
            plt.axhline(y=(i+1)*0.1*max(u.detach().cpu().squeeze()), linestyle=':', color='blue', linewidth = 3,label=f'{i+1}0% of max')
    plt.legend()
    #------------------------------------NEW--------------------------------------------------#
    plt.xlabel("output number")
    plt.ylabel("output")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(plotdir+'output_epoch%d_batch%d_X%d.png' % (epoch, batch, num),dpi=600) #original is(if error, just switch back): fig.savefig(plotdir+'output_epoch%d_X%d.png' % (epoch, p))
    plt.close(fig)


def _plot_probes(probes, ax):
    markers = []
    for i, probe in enumerate(probes):
        x,y = probe.coordinates()
        marker, = ax.plot(x,y,'.',markeredgecolor='none',markerfacecolor='k',markersize=4,alpha=0.8)
        markers.append(marker)
    return markers


def _plot_sources(sources, ax):
    markers = []
    for i, source in enumerate(sources):
        x,y = source.coordinates()
        marker, = ax.plot(x,y,'.',markeredgecolor='none',markerfacecolor='g',markersize=4,alpha=0.8)
        markers.append(marker)
    return markers


def geometry(model_input, ax=None, outline=False, outline_pml=True, epoch=0, plotdir=''):
    if ax is None:
        fig, axs_geom = plt.subplots(1, len(model_input), sharex=True, sharey=True,  constrained_layout=True)
        axs_geom = axs_geom.flatten()
    else:
        axs_geom = ax
        
    for i in range(len(model_input)): 
        model = model_input[i]
        geom = model.geom
        probes = model.probes
        sources = model.sources
        A = model.Alpha()[0, 0, ].squeeze()
        alph = A.min().cpu().numpy()
        B = geom.B[1,].detach().cpu().numpy().transpose()

        markers = []
        if not outline:
            if isinstance(model.geom, WaveGeometryMs):
                Msat = geom.Msat.detach().cpu().numpy().transpose()
                h1 = axs_geom[i].imshow(Msat, origin="lower", cmap=plt.cm.summer)
                plt.colorbar(h1, ax=axs_geom[i], label='Saturation magnetization (A/m)')
            else:
                h1 = axs_geom[i].imshow(B*1e3, origin="lower", cmap=plt.cm.summer)
                plt.colorbar(h1, ax=axs_geom[i], label='Magnetic field (mT)')
        else:
            if isinstance(model.geom, WaveGeometryMs):
                Msat = geom.Msat.detach().cpu().numpy().transpose()
                axs_geom[i].contour(Msat, levels=1, cmap=plt.cm.Greys, linewidths=[0.75], alpha=1)
            else:
                axs_geom[i].contour(B, levels=1, cmap=plt.cm.Greys, linewidths=[0.75], alpha=1)

        if outline_pml:
            b_boundary = A.cpu().numpy().transpose()
            axs_geom[i].contour(b_boundary, levels=[alph*1.0001], colors=['k'], linestyles=['dotted'], linewidths=[0.75], alpha=1)

        markers += _plot_probes(probes, axs_geom[i])
        markers += _plot_sources(sources, axs_geom[i])
            
    if plotdir:
        plt.legend()
        fig.savefig(plotdir+'geometry_epoch%d.png' % (epoch),dpi=600)
        plt.close(fig)


def wave_integrated(model, m_history, filename=''):
    fig, axs_int = plt.subplots(1, len(model), sharex=True, sharey=True, constrained_layout=True)
    axs_int = axs_int.flatten()

    m_int_list = [m_history[0].pow(2).sum(dim=0).numpy().transpose(),m_history[1].pow(2).sum(dim=0).numpy().transpose()]
    min_val = np.min([m_int_list[i] for i in range(len(model))])
    max_val = np.max([m_int_list[i] for i in range(len(model))])
    for i in range(len(model)):  
        m_int = m_int_list[i]
        vmax = m_int.max()
        h = axs_int[i].imshow(m_int, cmap=plt.cm.viridis, origin="lower", norm=LogNorm(vmin=vmax*0.01,vmax=vmax))
        geometry(model, ax=axs_int, outline=True)

    if filename:
        plt.colorbar(h, ax=axs_int[1])
        fig.savefig(filename,dpi=600)
        plt.close(fig)


def wave_snapshot(model, m_snap, filename='', clabel='m'):
    fig, axs_snap = plt.subplots(1, len(model), sharex=True, sharey=True, constrained_layout=True)
    axs_snap = axs_snap.flatten()
    m_t_list = [m_snap[0].cpu().numpy().transpose(), m_snap[1].cpu().numpy().transpose()]
    # Find the minimum and maximum values in the data
    min_val = np.min([m_t_list[i] for i in range(len(model))])
    max_val = np.max([m_t_list[i] for i in range(len(model))])

    for i in range(len(model)):
        m_t = m_t_list[i]
        norm = TwoSlopeNorm(vmin=min_val, vcenter=(min_val+max_val)/2, vmax=max_val)
        h = axs_snap[i].imshow(m_t, cmap=plt.cm.RdBu_r, origin="lower", norm=norm)
        geometry(model, ax=axs_snap, outline=True)
        axs_snap[i].axis('image')

    if filename:
        plt.colorbar(h, ax=axs_snap[1], label=clabel, shrink=0.80)
        fig.savefig(filename, dpi=600)
        plt.close(fig)

def Intensity_time(intensity_data, correct_label, epoch, Np, plotdir):
    ##### Probe Instnsity #####
    rt = []
    fig, ax = plt.subplots(int((Np+1)/2),2,sharex=True,sharey=True)
    ax = ax.flatten()
    for j in range(Np):
        info = [f'Target Output: {correct_label}']
        info.append('epoch: {}'.format(epoch))
        ax[j].plot(intensity_data[:,j],label='probe{}'.format(j))
        ax[j].legend(title="\n".join(info))
        rt.append(np.cumsum(intensity_data[:,j]))
    ax[0].set(ylabel='Intensity')
    ax[j].set(xlabel='timesteps')
    fig.tight_layout()
    plt.savefig(f'{plotdir}'+'label{}_IntensityEpoch{}.png'.format(correct_label, epoch))
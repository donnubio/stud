import librosa
import numpy as np
import scipy
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import LinearColorMapper, ColorBar
output_notebook() 


def AmplitudeSpectrum(t,y, t1=None,t2=None, remove_mean=True):
  '''
  знаходить спектр амплітуд.
  Вхідні аргументи:
  t,y   : вектори з моментами часу та виміряними значеннями сигналу
  t1,t2 : два моменти часу, початок та кінець фрагменту 
          який буде вирізаний з сигналу перед отриманням його спектру
  remove_mean: якщо True з сигналу видаляється його середнє
  Результати:
  частоти,амплітуди: вектори з частотами (Гц) та відповідними амплітудами
  '''

  if t1==None: t1=t[0]
  if t2==None: t2=t[-1]
  i=np.where((t>=t1)&(t<=t2))[0]
  t=t[i]
  y=y[i]
  if remove_mean:
    y=y-np.mean(y)
  частота_дискретиз = 1/(t[1]-t[0])
  амплітуди = scipy.fft.rfft(y, norm='forward')
  амплітуди = np.abs( амплітуди ) * 2
  частоти = scipy.fft.rfftfreq( len(y), 1 / частота_дискретиз)

  return(частоти,амплітуди)  


from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import LinearColorMapper, ColorBar
output_notebook()  # if you’re in a Jupyter notebook

def Spectrogram(t,y, 
                plot_spectrogram=True,
                pallete='Viridis256', 
                width=1000, height=300,
                n_fft=1024, hop_length=512):
  '''
  calculate and plot spectrogram.
  Parameters:
  t: time vector
  y: signal vector
  plot_spectrogram: if True, plot spectrogram
  pallete: color pallete (e.g. 'Viridis256','Plasma256','Inferno256','Rainbow256','Magma256')
  width, height: plot size
  n_fft, hop_length: parameters for STFT
  Returns:
  times, freqs, S_db: time, frequency, spectrogram
  '''
  # assume uniform spacing
  dt = t[1] - t[0]
  sr = 1.0 / dt
  # STFT
  S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
  # magnitude spectrogram (power)
  S_power = np.abs(S)**2
  # convert to dB
  S_db = librosa.power_to_db(S_power, ref=np.max)
  # time for each frame
  times = librosa.frames_to_time(np.arange(S_db.shape[1]),
                                sr=sr,
                                hop_length=hop_length)
  # frequency for each FFT bin
  freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

  if plot_spectrogram:
    # set up color mapper
    color_mapper = LinearColorMapper(palette=pallete,
                                    low=S_db.min(),
                                    high=S_db.max())

    # create figure
    p = figure(width=width, height=height,
              x_axis_label='Time (s)',
              y_axis_label='Frequency (Hz)',
              x_range=(times.min(), times.max()),
              y_range=(freqs.min(), freqs.max()))



    # draw the spectrogram
    p.image(image=[S_db],
            x=times.min(),
            y=freqs.min(),
            dw=times.max() - times.min(),
            dh=freqs.max() - freqs.min(),
            color_mapper=color_mapper)

    # add color bar
    color_bar = ColorBar(color_mapper=color_mapper,
                        label_standoff=12,
                        location=(0,0))
    p.add_layout(color_bar, 'right')

    show(p)  
  else:  
    return (times, freqs, S_db)

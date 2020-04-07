##########################################
# Basic Magnetization and Energy Plotter #
##########################################

use_t_for_x_axis = False # If True, uses d['t'] instead of step number for the x data



###################################
# No need to edit below this line #
###################################
if use_t_for_x_axis: 
  xlabels = ['Time (s)']
  x       = [d['t']]
else:                
  xlabels = 'Step'
  x       = None

y = []
ylabels = []
if solver['a/mode']:
  y.append(d['ax']); ylabels.append('ax')
  y.append(d['ay']); ylabels.append('ay')
  y.append(d['az']); ylabels.append('az')
  if solver['get_energy']:
    y.append(d['Ua']); ylabels.append('Ua')
if solver['b/mode']:
  y.append(d['bx']); ylabels.append('bx')
  y.append(d['by']); ylabels.append('by')
  y.append(d['bz']); ylabels.append('bz')
  if solver['get_energy']:
    y.append(d['Ub']); ylabels.append('Ub')

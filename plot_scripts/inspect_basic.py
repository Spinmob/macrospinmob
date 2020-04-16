##########################################
# Basic Magnetization and Energy Plotter #
##########################################

use_t = False # If True, uses d['t'] instead of step number for the x data



###################################
# No need to edit below this line #
###################################
if use_t: 
  xlabels = ['Time (s)']
  x       = [d['t']]
else:                
  xlabels = 'Step'
  x       = None

y = []
ylabels = []
if solver['a/enable']:
  for k in d.ckeys:
    if 'a' in k:
      ylabels.append(k)
      y.append(d[k])
if solver['b/enable']:
  for k in d.ckeys:
    if 'b' in k:
      ylabels.append(k)
      y.append(d[k])

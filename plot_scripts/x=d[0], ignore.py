# Shared x data and label
x = d[0]; xlabels='t'

# Keys to ignore
ignore = ['t', 'bx','by','bz','Ua','U','Ub']


##############################
# No need to edit below this #
##############################
y       = []
ylabels = []
for k in d.ckeys:
  if not k in ignore:
    y.append(d[k])
    ylabels.append(k)

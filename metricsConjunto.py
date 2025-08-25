import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": "mathptmx",})

plt.rc('axes', labelsize='x-large')
plt.rc('axes', titlesize='x-large')
plt.rc('xtick', labelsize='x-large')
plt.rc('ytick', labelsize='x-large')
#%%
#['Bantamweight', 'Middleweight', 'Heavyweight', 'Lightweight', 'Welterweight', 'Flyweight', 'LightHeavyweight', 'Featherweight', 'CatchWeight', 'OpenWeight', 
#'WomenStrawweight', 'WomenBantamweight', 'WomenFlyweight', 'WomenFeatherweight']

dataBantamweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/metrics_Bantamweight.csv')
dataMiddleweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/metrics_Middleweight.csv')
dataHeavyweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/metrics_Heavyweight.csv')
dataLightweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/metrics_Lightweight.csv')
dataWelterweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/metrics_Welterweight.csv')
dataFlyweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/metrics_Flyweight.csv')
dataLightHeavyweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/metrics_LightHeavyweight.csv')
dataFeatherweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/metrics_Featherweight.csv')
dataCatchWeight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/metrics_CatchWeight.csv')
dataOpenWeight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/metrics_OpenWeight.csv')
dataWomenStrawweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/metrics_WomenStrawweight.csv')
dataWomenBantamweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/metrics_WomenBantamweight.csv')
dataWomenFlyweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/metrics_WomenFlyweight.csv')
dataWomenFeatherweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/metrics_WomenFeatherweight.csv')

#%%

corrBantamweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/correlations_Bantamweight.csv')
corrMiddleweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/correlations_Middleweight.csv')
corrHeavyweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/correlations_Heavyweight.csv')
corrLightweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/correlations_Lightweight.csv')
corrWelterweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/correlations_Welterweight.csv')
corrFlyweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/correlations_Flyweight.csv')
corrLightHeavyweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/correlations_LightHeavyweight.csv')
corrFeatherweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/correlations_Featherweight.csv')
corrCatchWeight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/correlations_CatchWeight.csv')
corrOpenWeight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/correlations_OpenWeight.csv')
corrWomenStrawweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/correlations_WomenStrawweight.csv')
corrWomenBantamweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/correlations_WomenBantamweight.csv')
corrWomenFlyweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/correlations_WomenFlyweight.csv')
corrWomenFeatherweight = pd.read_csv('C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/plotsMetrics/correlations_WomenFeatherweight.csv')

#%%
savingPath = 'C:/Users/max_s/Desktop/MAGISTER-MAX/NOTES/UFC/paperImages/categoriesPPV/'
#%%
dfMetricsList = [dataBantamweight, dataMiddleweight, dataHeavyweight, dataLightweight, dataWelterweight, dataFlyweight, dataLightHeavyweight, dataFeatherweight, dataCatchWeight,dataOpenWeight, dataWomenStrawweight, dataWomenBantamweight, dataWomenFlyweight, dataWomenFeatherweight]
dfMetricsNames = ['Bantamweight', 'Middleweight', 'Heavyweight', 'Lightweight', 'Welterweight', 'Flyweight', 'LightHeavyweight', 'Featherweight', 'CatchWeight','OpenWeight', 'WomenStrawweight', 'WomenBantamweight', 'WomenFlyweight', 'WomenFeatherweight']
colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
]
#%%
for i, df in enumerate(dfMetricsList):
    plt.plot(df['deg'], '-o', lw = 0.5, ms = 1, color = f'{colors[i]}', label=f'{dfMetricsNames[i]}')
plt.legend(fontsize = 7)
plt.xlim(0, 301)
plt.ylim(1, 6)
plt.ylabel(r'$\langle k \rangle$')
plt.xlabel(r'window index')
plt.title('Evolution average degree')
#plt.savefig(f'{savingPath}degree.png', dpi = 1500)
plt.show()
#%%
for i, df in enumerate(dfMetricsList):
    plt.plot(df['density'], '-o', lw = 0.5, ms = 1, color = f'{colors[i]}', label=f'{dfMetricsNames[i]}')
plt.legend(fontsize = 7)
plt.xlim(0, 301)
#plt.ylim(1, 6)
plt.ylim(0, 3)
plt.ylabel(r'$\langle g \rangle$')
plt.xlabel(r'window index')
plt.title('Evolution norm. degree')
#plt.savefig(f'{savingPath}normDeg.png', dpi = 1500)
plt.show()

#%%
for i, df in enumerate(dfMetricsList):
    plt.plot(df['clustering'], '-o', lw = 0.5, ms = 1, color = f'{colors[i]}', label=f'{dfMetricsNames[i]}')
plt.legend(fontsize = 7)
plt.xlim(0, 301)
plt.ylim(0)
plt.ylabel(r'$\langle C \rangle$')
plt.xlabel(r'window index')
plt.title('Evolution averga clustering')
#plt.savefig(f'{savingPath}clustering.png', dpi = 1500)
plt.show()
#%%
for i, df in enumerate(dfMetricsList):
    plt.plot(df['Ginni'], '-o', lw = 0.5, ms = 1, color = f'{colors[i]}', label=f'{dfMetricsNames[i]}')
plt.legend(fontsize = 7)
plt.xlim(0, 301)
plt.ylim(0.1)
plt.ylabel(r'$G[p(k)]$')
plt.xlabel(r'window index')
plt.title('Evolution Ginni coeff.')
#plt.savefig(f'{savingPath}Ginni.png', dpi = 1500)
plt.show()
#%%
for i, df in enumerate(dfMetricsList):
    plt.plot(df['pathLenght'], '-o', lw = 0.5, ms = 1, color = f'{colors[i]}', label=f'{dfMetricsNames[i]}')
plt.legend(fontsize = 7)
plt.xlim(0, 301)
plt.ylim(0, 15)
plt.ylabel(r'$l$')
plt.xlabel(r'window index')
plt.title('Evolution average path lenght')
#plt.savefig(f'{savingPath}psthLenght.png', dpi = 1500)
plt.show()
#%%
for i, df in enumerate(dfMetricsList):
    plt.plot(df['betweenessCentrality'], '-o', lw = 0.5, ms = 1, color = f'{colors[i]}', label=f'{dfMetricsNames[i]}')
plt.legend(fontsize = 7)
plt.xlim(0, 301)
plt.ylim(0)
plt.ylabel(r'$b$')
plt.xlabel(r'window index')
plt.title('Evolution betweeness centrality')
#plt.savefig(f'{savingPath}between.png', dpi = 1500)
plt.show()
#%%
for i, df in enumerate(dfMetricsList):
    plt.plot(df['eigenvectorCentrality'], '-o', lw = 0.5, ms = 1, label=f'{dfMetricsNames[i]}')
plt.legend(fontsize = 7)
plt.xlim(0, 301)
plt.ylim(0.01, 2.5)
plt.ylabel(r'$\lambda$')
plt.xlabel(r'window index')
plt.title('Evolution eigenvector centrality')
#plt.savefig(f'{savingPath}eigenvector.png', dpi = 1500)
plt.show()
#%%

dfCorrList = [corrBantamweight, corrMiddleweight, corrHeavyweight, corrLightweight, corrWelterweight, corrFlyweight, corrLightHeavyweight, corrFeatherweight, corrCatchWeight,corrOpenWeight, corrWomenStrawweight, corrWomenBantamweight, corrWomenFlyweight, corrWomenFeatherweight]
dfMetricsNames = ['Bantamweight', 'Middleweight', 'Heavyweight', 'Lightweight', 'Welterweight', 'Flyweight', 'LightHeavyweight', 'Featherweight', 'CatchWeight','OpenWeight', 'WomenStrawweight', 'WomenBantamweight', 'WomenFlyweight', 'WomenFeatherweight']

xCorr = [r'$\langle C\rangle$', r'$\langle k \rangle$', r'$\langle g \rangle$', r'$ G[p(k)] $', '$l$', '$b$', '$\lambda$']
x_positions = np.arange(len(xCorr))

for i, df in enumerate(dfCorrList):
    plt.errorbar(x_positions, df.loc[0].values,  yerr=df.loc[1].values, fmt='o', alpha = 0.7, color = f'{colors[i]}', label=f'{dfMetricsNames[i]}')
#plt.figsize()
plt.title('correlations')
plt.xticks(x_positions, xCorr)
plt.ylim(-1,1)
plt.legend(loc=(1.02, 0.05))
plt.axhline(0, color='k')

plt.ylabel('Pearson correlation coefficient')
#plt.savefig(f'{savingPath}correlations.png', dpi = 1500)
plt.show()
#%%
def mean_and_errors_calculator(metric):
    values = []
    errors = []
    for i, df in enumerate(dfCorrList):
        
        value = df.at[0, f'{metric}']
        if math.isnan(value):
            print('nan')
        else:
            values.append(value)
            error = df.at[1, f'{metric}']
            errors.append(error)

    valuesA = np.array(values)
    errorsA = np.array(errors)
    
    weights = 1 / np.square(errorsA)
    weighted_sum = np.sum(valuesA * weights)
    sum_of_weights = np.sum(weights)

    weighted_average = weighted_sum / sum_of_weights
    uncertainty = 1 / np.sqrt(sum_of_weights)
    
    return weighted_average, uncertainty
#%%
columns = ['corrClust', 'corrDeg', 'corrDen', 'corrGin', 'corrPath', 'corrBet',
       'corrEigen']

averagesCorr = []

for index in columns:
    means = mean_and_errors_calculator(index)
    averagesCorr.append(means)
#%%
averagesCorrA = np.array(averagesCorr)

#%%
xCorr = [r'$\langle C\rangle$', r'$\langle k \rangle$', r'$\langle g \rangle$', r'$ G[p(k)] $', '$l$', '$b$', '$\lambda$']
x_positions = np.arange(len(xCorr))

plt.errorbar(x_positions, averagesCorrA[:,0],  yerr=averagesCorrA[:,1], fmt='o', alpha = 1, color = 'm')

plt.title('Averaged correlations categories')
plt.xticks(x_positions, xCorr)
plt.ylim(-1,1)
plt.axhline(0, color='k')
plt.ylabel('Pearson correlation coefficient')
    
    
    
    
    






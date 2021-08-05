Last Edited: August 5, 2021 11:47 AM

# README

This is the code for a webapp used to visualize theoretical allele frequency trajectories found here: 

- [https://neanderthal-heritability.herokuapp.com/](https://neanderthal-heritability.herokuapp.com/)

Specifically, it is made to build intuition for possible selective scenarios since interbreeding of Neanderthals and humans. It is meant to contextualize results from our Nature Communications paper and corresponding Nature Ecology & Evolution Community "Behind the paper" article.

- [McArthur, E., Rinker, D. C. & Capra, J. A. Quantifying the contribution of Neanderthal introgression to the heritability of complex traits. *Nat. Commun.* 2021 121 12, 1–14 (2021).](https://www.nature.com/articles/s41467-021-24582-y)
- McArthur, E. Uncovering the legacy of our Neanderthal ancestors. [https://natureecoevocommunity.nature.com/posts/uncovering-the-legacy-of-our-neanderthal-ancestors](https://natureecoevocommunity.nature.com/posts/uncovering-the-legacy-of-our-neanderthal-ancestors)

*DISCLAIMER!: This is a simple qualitative model. It is NOT a forward-time demographically-aware simulation of realistic evolution since introgression. We did not optimize any parameters, nor did we consider pleiotropy or LD. It is meant for illustrative purposes only and a tool for developing intuition for our model.*

**Please contact me if you have suggestions, problems, or if you find this useful. I am new to this!**

## Creation

This was created using Python and Javascript with tools such as:

- [Plotly Dash](https://plotly.com/dash/) (framework for building reactive web applications and interactive figures)
- [Dash Bootstrap](https://dash-bootstrap-components.opensource.faculty.ai/) (library of Bootstrap components for Dash)

It was a learning project for me (I am not a web developer at all!) and has some "advanced" Dash features including: 

- Client-side callbacks with Javascript (to speed up figure generation)
- Collapsible side bar
- Navigation bars and multiple pages

Check out [https://neanderthal-heritability.herokuapp.com/methods](https://neanderthal-heritability.herokuapp.com/methods) for more information on the scientific methods behind this app.

## Manifest

- `app.py` : The main python script for the app, callbacks, HTML layout, and CSS design
- `app_serverSideCallbacks.py` : An older version of the app with intact server side callback for the interactivity. I don't use this version anymore because of the client side callbacks but it might be useful for somebody learning how to implement server and client side callbacks for the same figures.
- `[type_of_figure]FigStorage.json` : An exported json version of initialized plotly figures that are read in and edited (3 types of figures: allele frequency figure, distribution figures, and arrow figure)
- `make_figures.py` : This is not used anymore but is the code for how the json plotly figures were generated, could remake and write/read these like this:

    ```python
    from make_figures import make_afFig, make_distFig, make_arrowFig
    afFigStorage = make_afFig()
    afFigStorage.write_json("afFigStorage.json")
    distFigStorage = make_distFig()
    distFigStorage.write_json("distFigStorage.json")
    arrowFigStorage = make_arrowFig()
    arrowFigStorage.write_json("arrowFigStorage.json")
    ```

- `simulations[_noPressure].csv` : Output for allele frequency trajectory simulations both under positive and negative selective pressures or no pressure at all, see code to generate this below
- `environment.yml` : conda environment specifications
- `Procfile`, `requirements.txt` , `runtime.txt` : files necessary to build environment on heroku, see heroku documentation or resources below for specifics
- `assets/callbacks.js` : Javascript clientside callbacks to update the figures
- `assets/[figure].[png/jpg/svg]` : Raster figures used in the app

## Resources that were helpful to me

### Collapsible side bar

- [https://stackoverflow.com/questions/62732631/how-to-collapsed-sidebar-in-dash-plotly-dash-bootstrap-components](https://stackoverflow.com/questions/62732631/how-to-collapsed-sidebar-in-dash-plotly-dash-bootstrap-components)
- [https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/](https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/)

### Client-side callbacks with Javascript

- Good example: [https://github.com/covid19-dash/covid-dashboard/blob/master/app.py](https://github.com/covid19-dash/covid-dashboard/blob/master/app.py)
- [https://replit.com/@StephenTierney/EnlightenedImpartialModule](https://replit.com/@StephenTierney/EnlightenedImpartialModule)
- [https://dash.plotly.com/clientside-callbacks](https://dash.plotly.com/clientside-callbacks)
- [https://community.plotly.com/t/is-it-possible-to-update-just-layout-not-whole-figure-of-graph-in-callback/8300/22](https://community.plotly.com/t/is-it-possible-to-update-just-layout-not-whole-figure-of-graph-in-callback/8300/22) : I was able to completely decouple any figure modification (e.g. adding descriptive shapes, traces, annotations etc.) while keeping the original (large) figure data in a dcc.Store object and not resend it on every update (of the auxiliary data), You have to make a deep copy of the figure object. Otherwise you might run into issues when adding elements to an existing list, e.g. such as traces.

### Other

- [https://stackoverflow.com/questions/60938972/dash-implementing-a-trace-highlight-callback](https://stackoverflow.com/questions/60938972/dash-implementing-a-trace-highlight-callback)
- [https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.LinearSegmentedColormap.html](https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.LinearSegmentedColormap.html)

### Tips for deploying

- [https://devcenter.heroku.com/articles/git](https://devcenter.heroku.com/articles/git)
- to push app to github & heroku `git push origin main` and `git push heroku main`

### Other code

```python
import numpy as np
import pandas as pd
from scipy.stats import norm

Ne = 1861
p0 = 0.05
ngen=2000

# Update these arrays:
fitness_weights = np.array([0 for x in range(50)]) # for no pressure (50 observations)
fitness_weights = norm.ppf(x) for x in np.linspace(0.01,0.99,50) # for normally distributed pressure (50 observations)
# positive if reference is better, negative if alt is better, 0 = no pressure

# Initialized dataframe:
af_df = pd.DataFrame(index=list(range(len(fitness_weights))), columns=['fitness_weight','af'])

for idx, weight in enumerate(fitness_weights):  # For each fitness weight, simulate gametes
    gametes = np.array([0 for i in range(2*Ne)])
    if p0 > 0:
        gametes[0:round(p0*2*Ne)] = 1

    np.random.shuffle(gametes)
    gametes = gametes.reshape(-1,2)

    af = [p0]
    for i in range(ngen-1): # for each generation
        probs = 1- np.apply_along_axis(sum, 1, gametes)*(-weight)
        probs = probs/sum(probs)
        parents = np.array([list(np.random.choice(range(Ne), size=2, replace=False, p=probs)) for i in range(Ne)])
        copy = np.array([list(np.random.choice([0,1],size=2,replace=True)) for i in range(Ne)])
        gametes = [[gametes[p[0]][c[0]],gametes[p[1]][c[1]]] for p,c in zip(parents,copy)]
        af_timepoint = sum([item for sublist in gametes for item in sublist])/(2*Ne) # allele frequency at time point
        #if (af_timepoint == 0) or (af_timepoint == 1):
        #    print('broke')
        #    break
        af.append(af_timepoint)
        
        
    af_df.loc[idx,'fitness_weight'] = weight
    af_df.loc[idx,'af'] = af

af_df[list(range(ngen))] = pd.DataFrame(af_df.af.tolist(), index= af_df.index)
af_df = af_df.drop(['af'], axis=1)
# Output
af_df.to_csv('simulations_noPressure.csv',sep=",",index=False)
af_df.to_csv('simulations_.csv',sep=",",index=False)
```
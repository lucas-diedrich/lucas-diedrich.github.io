# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| code-fold: true 

# Import modules
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t


# Parameters
color='#45B39D' 
plot_params = dict(
    marker='o', 
    linestyle='', 
    markeredgecolor='k'
)

n_individuals = 10000
length_of_year = 10
probability_of_death = 0.05
alpha_level = 0.05

n_replicates = 5
#
#
#
#| code-fold: true 

# Define functions 
def life(probability_of_death: float = 0.05):
    """ Simulate length of a life given a constant 
    probability of death every day """ 
    
    days_lived = 0
    while np.random.random() > probability_of_death: 
        days_lived += 1
    return days_lived 


def run_model(probability_of_death: float = 0.05, 
              length_of_year: int = 365, 
              n_individuals: int = 10000
              ): 
    """ Run model for a population of n_individuals and return probabilities to  
    die on day x of life"""

    # Simulate n_individuals lives 
    days_lived = [life(probability_of_death) for _ in range(n_individuals)]

    # Get day of year (0-birthday, 1-day after birthday ...)
    day_of_year = [dl % length_of_year for dl in days_lived]

    # Count deaths per day of year
    days, counts = np.unique(day_of_year, return_counts=True)

    # Return normalized values (probabilities)
    return pd.Series(counts/np.sum(counts), index=days)
#
#
#
#| code-fold: true 
# Run model multiple times to get estimate of uncertainty
np.random.default_rng(seed=15640423)

replicates = list()

for _ in range(n_replicates): 
    results = run_model(probability_of_death=probability_of_death, 
                        length_of_year=length_of_year, 
                        n_individuals=n_individuals
                        )
    replicates.append(results)

# Deal with missing values 
replicates = (pd.concat(replicates, axis=1)
             .reindex(range(length_of_year))
             .fillna(0)
             )

# Compute mean and confidence interval 
mean = replicates.mean(axis=1)
ci = t.ppf(1-alpha_level/2, n_replicates)*replicates.std(axis=1)/np.sqrt(n_replicates)

#
#
#
#| code-fold: true 
#| fig-cap: "Observed number of deaths at a day X after birthday divided by the total number of individuals. The fill indicates the confidence interval of the mean from 5 replications of the simulations. The gray line indicates a null model in which the likelihood to die is independent of relation to your birthday."

# Plot 
fig, ax = plt.subplots(1,1, figsize=(7,4))

ax.fill_between(range(length_of_year), mean-ci, mean+ci, 
                alpha=0.2, 
                color=color, 
                label=f'{(1-alpha_level)*100:.0f}% Confidence Interval n={n_replicates}'
                )

ax.plot(range(length_of_year), mean, 
        color=color, 
        **plot_params, 
        label='Simulated probability of death'
        )

ax.legend(loc='upper right')

ax.axhline(1/length_of_year, 0, 1, 
           color='#cccccc', 
           label='Expected by equal chance'
           )

ax.set_ylabel('Probability of death')
ax.set_xlabel(f'Day relative to birthday in {length_of_year}-day year')

ax.annotate('Birthday', xy=(0, mean[0]), xytext=(20,0),
            xycoords='data', 
            textcoords='offset points', 
            va='center', 
            ha='left', 
            fontsize=12,
            arrowprops=dict(width=1.5, headwidth=0)
            )

ax.set_ylim(0,0.15)

plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| code-fold: true
#| fig-cap: Overlay of our analytical solution and the simulation. 

def model(probability_of_death: float, length_of_year: int = 365): 
    """Implementation of our analytical model, assuming a constant probability of death """

        
    day_after_bday = np.arange(0, length_of_year, 1)

    pdeath = probability_of_death/(1-(1-probability_of_death)**length_of_year)*np.power(1-probability_of_death, day_after_bday)

    return day_after_bday, pdeath 




day_after_bday, pdeath = model(probability_of_death, length_of_year) 

# Plotting              
fig, ax = plt.subplots(1,1, figsize=(7,5))

# Confidence interval
ax.fill_between(range(length_of_year), mean-ci, mean+ci, 
                alpha=0.2, 
                color=color, 
                label=f'{(1-alpha_level)*100:.0f}% Confidence Interval n={n_replicates}'
                )
# Simulation
ax.plot(range(length_of_year), mean, 
        color=color, 
        **plot_params, 
        label='Simulation'
        )
# Model 
ax.plot(day_after_bday, pdeath, 
        marker='', linestyle='-', linewidth=3, color='#555555', label='Our model')

ax.set_ylabel('P(X)')
ax.set_xlabel(f'Day after birthday ({length_of_year} day year)')
ax.set_ylim(0,0.15)
ax.legend(loc='upper right')
plt.show()

#
#
#
#
#
#
#
#
#
#
#
#| code-fold: true

day_after_bday, pdeath = model(0.001/365, 365)


fig, ax = plt.subplots(1,1)

ax.plot(day_after_bday, pdeath, 
        marker='', linestyle='-', linewidth=3, color='#555555', label='Our model')

ax.annotate(text=f'Change {(pdeath[0] - pdeath[-1])*100:.2g} %', xy=(0.5,0.6), xytext=(0,0), xycoords='axes fraction', textcoords='offset points', ha='center', va='center')

ax.set_ylabel('P(X)')
ax.set_ylim(pdeath.min()*0.9,pdeath.max()*1.1)
ax.set_xlabel(f'Day after birthday ({length_of_year} day year)')
ax.legend(loc='upper right')
plt.show()

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

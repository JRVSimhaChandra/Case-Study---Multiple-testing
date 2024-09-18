#!/usr/bin/env python
# coding: utf-8

# ![ibm-cloud.png](attachment:ibm-cloud.png)
# 
# # Case Study - Multiple testing

# ## Synopsis
# 
# The management team at AAVAIL is preparing to deploy a large number of teams each tasked with integration into a different new market.  They claim to have a optimized the teams fairly with respect to skills and experience.  They are asking you to come up with a framework to evaluate the makeup of their teams.  They have not finished hiring and creating all of the teams so naturally before you even get the data you wanted to get a head start.
# 
# Getting a head start usually involves finding a similar dataset and writing the code in a way that the new data, once obtained can be added with little effort.

# When we perform a large number of statistical tests, some will have $p$-values less than the designated level of $\alpha$ (e.g. 0.05) purely by chance, even if all the null hypotheses are really true.  This is an inherent risk of using inferrential statistics.  Fortunately, there are several techniques to mitigate the risk.
# 
# We are going to look at the 2018 world cup data in this example.  
# 
# The case study is comprised of the following sections:
# 
# 1. Data Cleaning
# 2. Data Visualization
# 3. NHT
# 4. Adjust NHT results for multiple comparisons
# 
# Data science work that focuses on creating a predictive model is perhaps the hallmark of the field today, but there are still many use cases where [inferential statistics](https://en.wikipedia.org/wiki/Statistical_inference) are the best tool available. One issue with statistical inference is that there are situations where [performing multiple tests](https://en.wikipedia.org/wiki/Multiple_comparisons_problem) is a  logical way to accomplish a task, but it comes at the expense of an increased rate of false positives or Type I errors.
# 
# In this case study you will apply techniques and knowledge from all of the units in Module 2.

# ## Getting started
# 
# **This unit is interactive**.  During this unit we encourage you to [open this file as a notebook](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest).  Download the notebook from the following link then open it locally using a Jupyter server or use your IBM cloud account to login to Watson Studio.  Inside of Waston Studio cloud if you have not already ensure that this notebook is loaded as part of the *project* for this course. As a reminder fill in all of the places in this notebook marked with ***YOUR CODE HERE*** or ***YOUR ANSWER HERE***.  The data and notebook for this unit are available below.
# 
# * [m2-u7-case-study.ipynb](./m2-u7-case-study.ipynb)
# * [worldcup-2018.csv](../data/worldcup-2018.csv)
# 
# This unit is organized into the following sections:
# 
# 1. Data Processing
# 2. Data Summary
# 3. Investigative Visualization
# 4. Hypothesis testing
# 
# #### Resources
# 
# * [Creating or uploading a notebook in IBM cloud](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/creating-notebooks.html)
# * [Resources for multiple testing in Python](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)

# In[ ]:


get_ipython().run_cell_magic('capture', '', '! pip install pingouin\n!conda install -c conda-forge pingouin')


# In[ ]:


import os
import re
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels as sm
import pingouin


import matplotlib.pyplot as plt
plt.style.use('seaborn')

get_ipython().run_line_magic('matplotlib', 'inline')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
LARGE_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title


# ## Import the Data

# Before we jump into the data it can be useful to give a little background so that you can better understand the features.  Since the dawn of statistics practitioners have been trying to find advantages when it comes to games.  Much of this was motivated by gambling---here we will look at the results from this tournament in a different way.  We are going to ask the simple question
# 
#   >Was the tournament setup in a fair way?
# 
# Of course the findings from an investigation centering around this question could be used to strategically place bets, but lets assume that we are simply interested in whether or not the tournament organizers did an adequate job.  The reason for doing this is to prepare for the AAVAIL data that is coming.  This exercise is an important reminder that you do not have to wait until the day that data arrive to start your work. 
# 
# There are 32 teams, each representing a single country, that compete in groups or pools then the best teams from those groups compete in a single elimination tournament to see who will become world champions.  This is by far the world's most popular sport so one would hope that the governing organization FIFA did a good job composing the pools.  If for example there are 8 highly ranked teams then each of those teams should be in a different pool. 
# 
# In our data set we have more than just rank so we can dig in a little deeper than that, but first let's have a look at the data.

# In[ ]:


DATA_DIR = os.path.join("..","data")
df = pd.read_csv(os.path.join(DATA_DIR, 'worldcup-2018.csv'))
df.columns = [re.sub("\s+","_",col.lower()) for col in df.columns]
df.head()


# To limit the dataset for educational purposes we create a new data frame that consists of only the following columns: 
# 
# * team
# * group
# * previous_appearances
# * previous_titles
# * previous_finals
# * previous_semifinals
# * current_fifa_rank

# ## Data Processing
# 
# ### QUESTION 1
# 
# Using the column names below create a new dataframe that uses only them.

# In[ ]:


columns = ['team', 'group','previous_appearances','previous_titles','previous_finals',
           'previous_semifinals','current_fifa_rank']

### YOUR CODE HERE
df = df.loc[:, columns]
df.head()


# To help with this analysis we are going to engineer a feature that combines all of the data in the table.  This feature represents the past performance of a team.  Given the data we have it is the best proxy on hand for how good a team will perform.  Feel free to change the multipliers, but let's just say that `past_performance` will be a linear combination of the related features we have.
# 
# Let $X_{1}$,...,$X_{4}$ be `previous_titles`,`previous_finals`,`previous_semifinals`,`previous_appearances` and let the corresponding vector $\mathbf{\alpha}$ be the multipliers.  This will give us,
# 
# $$
# \textrm{past_performance} = \alpha_{1} X_{1} + \alpha_{2} X_{2} + \alpha_{3} X_{3} + \alpha_{4} X_{4}
# $$
# 
# Modify $\mathbf{\alpha}$ if you wish.  Then add to your dataframe the new feature `past_performance`.
# 
# ### QUESTION 2
# 
# create the engineered feature `past_performance` as a new column

# In[ ]:


alpha = np.array([16,8,4,1])

### YOUR CODE HERE
df['past_performance'] = alpha[0] * df['previous_titles'] + alpha[1] * df['previous_finals'] + alpha[2] * df['previous_semifinals'] + alpha[3] * df['previous_appearances']


# ## Data Summary

# ### QUESTION 3
# 
# Using the `pivot_table` function create one or more **tabular summaries** of the data

# In[ ]:


### YOUR CODE HERE
columns_to_show =['previous_appearances','previous_titles','previous_finals',
                  'previous_semifinals','current_fifa_rank','past_performance']
group_members = pd.pivot_table(df, index = ['group', 'team'], values=columns_to_show).round(3)
group_members


# In[ ]:


columns_to_show =['previous_appearances','previous_titles','previous_finals',
                  'previous_semifinals','current_fifa_rank','past_performance']
group_summary = pd.pivot_table(df, index = ['group'], values=columns_to_show, aggfunc='mean').round(3)
group_summary


# ### QUESTION 4
# 
# Check for missing data. Write code to identify if there is any missing data.

# In[ ]:


### YOUR CODE HERE

row_with_missing = [row_idx for row_idx,row in df.isnull().iterrows() if True in row.values]
if len(row_with_missing) > 0:
    print([df['team'].values[r] for r in row_with_missing])
else:
    print("There were no rows with missing data")
    
## missing values summary
print("\nMissing Value Summary\n{}".format("-"*35))
print(df.isnull().sum(axis = 0))


# ## Investigative Visualization
# 
# ### QUESTION 5
# 
# Come up with one or more plots that investigate the central question... Are the groups comprised in a fair way?

# In[ ]:


### YOUR CODE HERE

# The `group_summary` dataframe was created as part of Question 3's solution.
plt.bar(group_summary.index, group_summary['past_performance'].values)
plt.xlabel('Group')
plt.ylabel('Mean Past Performance')


# In[ ]:


# The mean Past Performance score of teams in each group has a pretty wide spread.
# (higher past performance scores indicate teams with more successful World Cup histories)
# Let's compare our metric with 'current_fifa_rank', where a low number indicates a stronger team.

plt.bar(group_summary.index, group_summary['current_fifa_rank'].values)
plt.xlabel('Group')
plt.ylabel('Mean FIFA Rank')


# In[ ]:


# In comparison the mean FIFA rank in each group seems much more tightly grouped, with the obvious exception
# of Group A, which seems to be a weak group by this metric (though average or above average in terms of 
# Past Performance).

# To have a better idea if these differences are likely due to randomness or a systematic issue, we need
# to apply some variant of hypothesis testing.


# ## Hypothesis Testing
# 
# There are a number of ways to use hypothesis testing in this situation.  There are certainly reasonable hypotheses tests and other methods like simulation approaches, that we have not discussed, but they would be appropriate here.  If you choose to explore some of the methods that are outside the scope of this course then we encourage you to first try the simple approach proposed here and compare the results to any further additional approaches you choose to use.
# 
# We could use an ANOVA approach here that would signify a difference between groups, but we would not know which and how many teams were different.  As we stated before there are a number of ways to approach the investigation, but lets use a simple approach.  We are going to setup our investigation to look at all pairwise comparisons to provide as much insight as possible.
# 
# Recall that there are $\frac{(N-1)(N)}{2}$ pairwise comparisons.

# In[ ]:


N = np.unique(df['group'].values).size
print("num comparisons: ",((N-1)*N) / 2.0)


# ### QUESTION 5
# 
# 
# 1. Choose a hypothesis test
# 2. State the null and alternative hypothesis, and choose a cutoff value $\alpha$
# 3. Run the test for all pairwise comparisons between teams. You can either loop over the different groups and use the [ttest_ind](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html) function provided by the stats library or explore the [pairwise_ttests](https://pingouin-stats.org/generated/pingouin.pairwise_ttests.html) function provided by the pingouin library.

# In[ ]:


### YOUR CODE HERE

# 1. Since there are only 4 teams in each group, we are well within the "small number statistics" range
#    when you would use a t-test. We are comparing two samples in each t-test, and are interested in 
#    whether the samples are *different* from each other -- meaning a two-tailed test. Finally, 
#    there are separate t-tests for when the variances of the samples are expected to be equal or not.
#    As described in the NHT unit, it is generally safest to NOT assume equal variance.

# 2. Null hypothesis: The mean FIFA rank (or past performance) within a group of teams is equal to the 
#    mean in another group in the tournament.
#    Alt. hypothesis: The mean FIFA rank (or past performance) within a group of teams is different than the 
#    mean in another group in the tournament.
#    Set alpha = 0.1 (other values could also be reasonably chosen as well)

# 3. Write pairwise test as a function so that it can be reused for both 'past_performance' and 'current_fifa_rank'

def worldcup_pairwise_ttest(data, test_column = 'current_fifa_rank'):
    """Performs t-tests of independence pairwise between each of the 8 groups in the
    world cup data set. Returns a dictionary of the associated p-values."""
    pair_p_vals = {}
    grps = 'ABCDEFGH'
    for grp1_index, grp1 in enumerate(grps):
        for grp2 in grps[grp1_index+1:]:
            grp_key = '-'.join([grp1, grp2])
            grp1_data = data.loc[data.loc[:, 'group'] == grp1, test_column].values
            grp2_data = data.loc[data.loc[:, 'group'] == grp2, test_column].values
            
            pval = stats.ttest_ind(grp1_data, grp2_data, equal_var = False).pvalue
            pair_p_vals[grp_key] = pval
            
    return pair_p_vals

past_perf_p_vals = worldcup_pairwise_ttest(df, test_column = 'past_performance')
fifa_rank_p_vals = worldcup_pairwise_ttest(df)

# Check that each dictionary has the right number of pairs:
print('past_performance pair count:', len(past_perf_p_vals))
print('current_fifa_rank pair count:', len(fifa_rank_p_vals))


# In[ ]:


# You can also use the pingouin library to do pairwise t-tests

test_results = pingouin.pairwise_ttests(data=df, dv='past_performance', between='group', alpha=0.1, correction=True)
test_results.head()

# See the documentation of the pingouin library to learn more about this function : 
# https://pingouin-stats.org/generated/pingouin.pairwise_ttests.html 

# p-unc is the Uncorrected p-value for this test.


# ### QUESTION 6
# 
# For all of the $p$-values obtained apply the Bonferroni and at least one other correction for multiple hypothesis tests.  Then comment on the results.

# In[ ]:


### YOUR CODE HERE
def test_pvals_w_bonferroni(pvals_dict, alpha):
    """Applies the Bonferroni correction to the cutoff value alpha as determined 
    by the number p-values contained in pvals_dict. Then tests whether those
    p-values are at least as extreme as the cutoff. Returns a new dict with boolean
    values. True: Reject the Null. False: Fail to reject the Null."""
    alpha_bonf = alpha / len(pvals_dict)
    return {k: v < alpha_bonf for k, v in pvals_dict.items()}

past_perf_bonf_p_vals = test_pvals_w_bonferroni(past_perf_p_vals, 0.1)
fifa_rank_bonf_p_vals = test_pvals_w_bonferroni(fifa_rank_p_vals, 0.1)

# In Python True evaluates to 1 and False evaluates to 0. So use that to count things up:
print("Reject the null count, past_performance:", sum(past_perf_bonf_p_vals.values()))
print("Reject the null count, current_fifa_rank:", sum(fifa_rank_bonf_p_vals.values()))


# In[ ]:


pingouin.pairwise_ttests(data=df, dv='past_performance', between='group', alpha=0.1, padjust='bonf', correction=True)


# In[ ]:


# Applying the Bonferroni correction in this case means just comparing our p-values
# with the adjusted alpha: 0.1 / 28 = 0.00357, or equivalently multipling each of the
# p-values in our dictionary by 28 and then comparing these to the original alpha = 0.1.
# Since this is a simple calculation, we don't really need to use a stats library.
# However for more sophisticated corrections, such as Benjamini-Hochberg, it can be
# convenient to use a library:

from statsmodels.stats.multitest import multipletests

# unpack dicts into lists of p-values
perf_pval_lst = list(past_perf_p_vals.values())
fifa_pval_lst = list(fifa_rank_p_vals.values())

perf_bh_tests = multipletests(perf_pval_lst, alpha = 0.1, method = 'fdr_bh')
fifa_bh_tests = multipletests(fifa_pval_lst, alpha = 0.1, method = 'fdr_bh')

# multipletests returns a tuple of items. The first item is an array of test results.
print("Reject the null count, past_performance:", sum(perf_bh_tests[0]))
print("Reject the null count, current_fifa_rank:", sum(fifa_bh_tests[0]))


# In[ ]:


pingouin.pairwise_ttests(data=df, dv='past_performance', between='group', alpha=0.1, padjust='fdr_bh', correction=True)


# In[ ]:


# The full contents returned from multipletests (and a similar tuple for 
# for past performance):
fifa_bh_tests


# In[ ]:


# This tuple contains: Test results, corrected p-values, corrected alpha 
# using Sidak correction, corrected alpha using Bonferroni correction.  


# In[ ]:


# Let's unpack these results. When comparing the strength of teams between groups, no pair of groups meets
# our criteria of having statistically significantly different means in terms of either FIFA Rank or their
# Past Performance scores. Which is good. This is the result we would expect to see if the tournament is set
# up fairly.


# In[ ]:


# EXTRA:
# When we plotted the FIFA Rankings between groups we noted that Group A's mean was higher than the rest. 
# The groups with the lowest mean with this metric (meaning they consist of strong teams) are Group C, and 
# Group E. Even though we have determined that the difference between Groups A-C and Groups A-E, DID NOT
# meet our threshold for rejecting the Null, the difference is still somewhat notable. So going back 
# and examining these values is a reasonable thing to do.

df.loc[df['group'].isin(list('ACE')), ['team', 'group', 'current_fifa_rank']]


# In[ ]:


# One thing that stands out in these data is that Group A has 2 teams with ranks in the 60s
# while the worst ranked teams in groups C and E are in the high 30s.

# For the curious, Wikipedia sheds some light on this somewhat surprising situation:
# https://en.wikipedia.org/wiki/2018_FIFA_World_Cup_seeding
# According to Wikipedia: "The hosts [were] placed in Pot 1 and treated as a seeded team, 
# and therefore Pot 1 consisted of hosts Russia and the seven highest-ranked teams that 
# qualify for the tournament."

# The host nation, Russia, got special treatment in the grouping method, which wss otherwise
# based on their FIFA rank at the time. -- This treatment showed up in our earlier plot.


# ## Additional Approaches 
# 
# There is an [allpairtest function in statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.sandbox.stats.multicomp.MultiComparison.allpairtest.html) that could be used here to combine the work from QUESTION 5 and QUESTION 6.
# 
# Generalized Linear Models (GLMs) are an appropriate tool to use here if we wanted to include the results of the tournament (maybe a ratio of wins/losses weighted by the final position in the tournament).  `statsmodels` supports [R-style formulas to fit generalized linear models](https://www.statsmodels.org/stable/examples/notebooks/generated/glm_formula.html). One additional variant of GLMs are hierarchical or multilevel models that provide even more insight into this types of dataset.  See the [tutorial on multilevel modeling](https://docs.pymc.io/notebooks/multilevel_modeling.html).

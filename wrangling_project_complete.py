import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

pbp = pd.read_csv("C:/Users/asaar/Downloads/Machine Learning/nfl_pbp_2024.csv")


#Initial Exploration, do "riskier passes" correlate to higher EPA?
sns.scatterplot(pbp, x = 'air_yards', y = 'epa')
plt.title("Do Air Yards Predict EPA?")
plt.ylabel("EPA")
plt.xlabel("Air Yards")
plt.show()

risky_ols = sm.formula.ols('air_yards ~ epa', pbp).fit()

risky_ols.summary()

#Conclusion: Almost no relationship between "riskier passes" and higher EPA

pbp['deep_yes'] = pbp['desc'].str.contains('pass deep',regex = True)

pbp['deep_yes'] = pbp['deep_yes'].apply(lambda x: 1 if x else 0)

deep_logit = sm.formula.logit('deep_yes ~ epa', pbp).fit()
deep_logit.summary()

#Plays with higher EPA are significantly more likely to be described as “pass deep,” 
#with each additional EPA point increasing the odds of a deep play by more than 2×

pbp['short_yes'] = pbp['desc'].str.contains('pass short',regex = True)

pbp['short_yes'] = pbp['short_yes'].apply(lambda x: 1 if x else 0)

deep_logit = sm.formula.logit('short_yes ~ epa', pbp).fit()
deep_logit.summary()

#Short passes are slightly more likely to have higher EPA, but the relationship is weak especially compared to deep passes


#Lets explore leaguewide scramble direction

pbp["desc_lower"] = pbp["desc"].str.lower()

# Create boolean indicators for scramble direction (for all QBs)
pbp["scramble_left_all"] = pbp["desc_lower"].str.contains("scrambles left")
pbp["scramble_right_all"] = pbp["desc_lower"].str.contains("scrambles right")
pbp["scramble_middle_all"] = pbp["desc_lower"].str.contains("scrambles up the middle")

# Keep only scramble plays
pbp_scrambles = pbp[pbp["desc_lower"].str.contains("scrambles")].copy()

# Compute average EPA across directions (league-wide)
scramble_results_all = {
    "left": pbp_scrambles.loc[pbp_scrambles["scramble_left_all"], "epa"].mean(),
    "right": pbp_scrambles.loc[pbp_scrambles["scramble_right_all"], "epa"].mean(),
    "middle": pbp_scrambles.loc[pbp_scrambles["scramble_middle_all"], "epa"].mean(),
}

scramble_df_all = pd.DataFrame.from_dict(
    scramble_results_all, orient="index", columns=["mean_epa"]
)
scramble_df_all.index.name = "scramble_direction"
scramble_df_all = scramble_df_all.reset_index()

print("League-wide scramble EPA by direction:")
print(scramble_df_all)

plt.figure(figsize=(8, 6))
plt.bar(scramble_df_all["scramble_direction"], scramble_df_all["mean_epa"])
plt.title("League-wide – EPA by Scramble Direction (2024)")
plt.ylabel("Mean EPA")
plt.xlabel("Scramble Direction")
plt.axhline(0, color="black", linewidth=1)
plt.tight_layout()
plt.show()


# Lets explore league-wide EPA by pass depth (Deep vs Short)

# Use lowercase descriptions
pbp["pass_deep_all"] = pbp["desc_lower"].str.contains("pass deep")
pbp["pass_short_all"] = pbp["desc_lower"].str.contains("pass short")

# Keep only plays with either deep or short tag
pbp_pass_depth = pbp[(pbp["pass_deep_all"]) | (pbp["pass_short_all"])].copy()

pass_depth_results_all = {
    "Deep Pass": pbp_pass_depth.loc[pbp_pass_depth["pass_deep_all"], "epa"].mean(),
    "Short Pass": pbp_pass_depth.loc[pbp_pass_depth["pass_short_all"], "epa"].mean(),
}

pass_depth_df_all = pd.DataFrame.from_dict(
    pass_depth_results_all, orient="index", columns=["mean_epa"]
)
pass_depth_df_all.index.name = "pass_type"
pass_depth_df_all = pass_depth_df_all.reset_index()

print("League-wide mean EPA by pass depth:")
print(pass_depth_df_all)

plt.figure(figsize=(8, 6))
plt.bar(pass_depth_df_all["pass_type"], pass_depth_df_all["mean_epa"],
        color=["#4C72B0", "#55A868"])
plt.title("League-wide – Mean EPA: Deep Pass vs Short Pass (2024)")
plt.ylabel("Mean EPA")
plt.xlabel("Pass Depth Type")
plt.axhline(0, color="black", linewidth=1)
plt.tight_layout()
plt.show()

# Lets take a closer look at MVP candidate Lamar Jackson

pbp["desc_lower"] = pbp["desc"].str.lower()

# Filtering for only lamar jackson plays
lamar = pbp[pbp["desc_lower"].str.contains("l.jackson")].copy()

# Creating boolean indicators for scramble direction sorting by left, right and center
lamar["scramble_left"] = lamar["desc_lower"].str.contains("scrambles left")
lamar["scramble_right"] = lamar["desc_lower"].str.contains("scrambles right")
lamar["scramble_middle"] = lamar["desc_lower"].str.contains("scrambles up the middle")

# Ensuring it's a scramble play
lamar_scrambles = lamar[lamar["desc_lower"].str.contains("scrambles")].copy()


results = {
    "left": lamar_scrambles.loc[lamar_scrambles["scramble_left"], "epa"].mean(),
    "right": lamar_scrambles.loc[lamar_scrambles["scramble_right"], "epa"].mean(),
    "middle": lamar_scrambles.loc[lamar_scrambles["scramble_middle"], "epa"].mean(),
}

results_df = pd.DataFrame.from_dict(results, orient="index", columns=["mean_epa"])
results_df.index.name = "scramble_direction"
results_df = results_df.reset_index()

print(results_df)

plt.figure(figsize=(8,6))
plt.bar(results_df["scramble_direction"], results_df["mean_epa"])
plt.title("Lamar Jackson – EPA by Scramble Direction (2024)")
plt.ylabel("Mean EPA")
plt.xlabel("Scramble Direction")
plt.axhline(0, color="black", linewidth=1)
plt.show()

lamar["pass_deep"] = lamar["desc_lower"].str.contains("pass deep")
lamar["pass_short"] = lamar["desc_lower"].str.contains("pass short")

# Keep only passing plays with clear deep/short information
lamar_pass = lamar[(lamar["pass_deep"]) | (lamar["pass_short"])].copy()

# Compute mean EPA by pass depth classification
results = {
    "Deep Pass": lamar_pass.loc[lamar_pass["pass_deep"], "epa"].mean(),
    "Short Pass": lamar_pass.loc[lamar_pass["pass_short"], "epa"].mean()
}

results_df = pd.DataFrame.from_dict(results, orient="index", columns=["mean_epa"])
results_df.index.name = "pass_type"
results_df = results_df.reset_index()

print(results_df)

# Lets visualize the difference between deep and short passes for Lamar
plt.figure(figsize=(8,6))
plt.bar(results_df["pass_type"], results_df["mean_epa"], color=["#4C72B0", "#55A868"])
plt.title("Lamar Jackson – Mean EPA: Deep Pass vs Short Pass (2024)")
plt.ylabel("Mean EPA")
plt.xlabel("Pass Depth Type")
plt.axhline(0, color="black", linewidth=1)
plt.tight_layout()
plt.show()


# Now lets explore Josh Allen

allen = pbp[pbp["desc_lower"].str.contains("j.allen")].copy()

# Scramble detection
allen["scramble_left"] = allen["desc_lower"].str.contains("scrambles left")
allen["scramble_right"] = allen["desc_lower"].str.contains("scrambles right")
allen["scramble_middle"] = allen["desc_lower"].str.contains("scrambles up the middle")

# Keep only scramble plays
allen_scrambles = allen[allen["desc_lower"].str.contains("scrambles")].copy()

# Compute mean EPA across directions
scramble_results = {
    "left": allen_scrambles.loc[allen_scrambles["scramble_left"], "epa"].mean(),
    "right": allen_scrambles.loc[allen_scrambles["scramble_right"], "epa"].mean(),
    "middle": allen_scrambles.loc[allen_scrambles["scramble_middle"], "epa"].mean(),
}

scramble_df = pd.DataFrame.from_dict(scramble_results, orient="index", columns=["mean_epa"])
scramble_df.index.name = "scramble_direction"
scramble_df = scramble_df.reset_index()

print(scramble_df)

# Plot results
plt.figure(figsize=(8,6))
plt.bar(scramble_df["scramble_direction"], scramble_df["mean_epa"], color=["#00338D", "#C60C30", "#4C72B0"])
plt.title("Josh Allen – EPA by Scramble Direction (2024)")
plt.ylabel("Mean EPA")
plt.xlabel("Scramble Direction")
plt.axhline(0, color="black", linewidth=1)
plt.tight_layout()
plt.show()

# Identify deep vs short passes
allen["pass_deep"] = allen["desc_lower"].str.contains("pass deep")
allen["pass_short"] = allen["desc_lower"].str.contains("pass short")

# Keep only plays with a clear deep/short label
allen_pass = allen[(allen["pass_deep"]) | (allen["pass_short"])].copy()

# Compute mean EPA by pass depth
pass_results = {
    "Deep Pass": allen_pass.loc[allen_pass["pass_deep"], "epa"].mean(),
    "Short Pass": allen_pass.loc[allen_pass["pass_short"], "epa"].mean()
}

pass_df = pd.DataFrame.from_dict(pass_results, orient="index", columns=["mean_epa"])
pass_df.index.name = "pass_type"
pass_df = pass_df.reset_index()

print(pass_df)

# Plot results
plt.figure(figsize=(8,6))
plt.bar(pass_df["pass_type"], pass_df["mean_epa"], color=["#00338D", "#7FB5F5"])
plt.title("Josh Allen – Mean EPA: Deep Pass vs Short Pass (2024)")
plt.ylabel("Mean EPA")
plt.xlabel("Pass Depth Type")
plt.axhline(0, color="black", linewidth=1)
plt.tight_layout()
plt.show()
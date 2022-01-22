import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("src\labels\AllLabelsOld.csv")
print(df)
df["Engagement"] = df["Engagement"].replace(1,0)
df["Engagement"] = df["Engagement"].replace(2,1)
df["Engagement"] = df["Engagement"].replace(3,1)

df["Engagement"].plot(kind="hist")
plt.grid(color='white', lw = 1, axis='x')
plt.xlabel("Enaggement Levels")
plt.ylabel("Number of video clips")
plt.title("DAiSEE dataset")

plt.show()
df.to_csv("src\labels\AllLabels.csv", index=False)

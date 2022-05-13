import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("src\labels\AllLabels.csv")
print(df)
df["Engagement"] = df["Engagement"].replace(1,0)
df["Engagement"] = df["Engagement"].replace(2,1)
df["Engagement"] = df["Engagement"].replace(3,1)

df["ClipID"] = df["ClipID"].replace("avi","jpg")
df["ClipID"] = df["ClipID"].replace("mp4","jpg")

df.to_csv("src\labels\AllLabels.csv", index=False)

df["Engagement"].plot(kind="hist")
plt.grid(color='white', lw = 1, axis='x')
plt.xlabel("Engagement Levels")
plt.ylabel("Number of video clips")
plt.title("DAiSEE dataset")
plt.show()
import pandas as pd

mfcc = pd.read_csv("mfcc_features.csv")
lpc = pd.read_csv("LPC.csv")
plp = pd.read_csv("PLP.csv")

print("MFCC:", mfcc.shape)
print("LPC:", lpc.shape)
print("PLP:", plp.shape)

print(mfcc.head())
print(lpc.head())
print(plp.head())
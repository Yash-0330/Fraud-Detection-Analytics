
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 1. Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Create a folder for charts
Path("charts").mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# 2. Display basic statistics
print("\n--- DATA INFORMATION ---")
print(df.info())

print("\n--- TRANSACTION AMOUNT STATISTICS ---")
print(df["Amount"].describe())

print("\n--- FRAUD AND GENUINE COUNTS ---")
print(df["Class"].value_counts())

# 3. Chart 1: Class distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Class")
plt.title("Genuine vs Fraudulent Transactions")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("charts/class_distribution.png")
plt.show()

# 4. Chart 2: Transaction amount distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Amount", bins=50)
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("charts/amount_distribution.png")
plt.show()

# 5. Chart 3: Amount by class
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Class", y="Amount")
plt.title("Transaction Amount by Class")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Transaction Amount")
plt.tight_layout()
plt.savefig("charts/amount_by_class.png")
plt.show()

print("\nEDA completed successfully!")
print("Charts saved in the charts folder.")

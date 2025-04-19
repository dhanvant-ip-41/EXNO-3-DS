## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
NAME : Dhanvant Kumar V

REG NO : 212224040070
```python
import pandas as pd
```
```python
df=pd.read_csv("/content/Encoding Data (2).csv")
df
```
![Screenshot 2025-04-19 195854](https://github.com/user-attachments/assets/25ac052c-7816-41a2-9c51-f0340b0584f4)


## ORDINAL ENCODER
```python
from sklearn.preprocessing import OrdinalEncoder
e1=OrdinalEncoder(categories=[["Hot","Warm","Cold"]])
e1.fit_transform(df[['ord_2']])
```
![Screenshot 2025-04-19 195905](https://github.com/user-attachments/assets/ffce3fc9-f22b-4d35-a88f-44886a39aa62)


```python
df['bo2']=e1.fit_transform(df[['ord_2']])
df
```
![Screenshot 2025-04-19 195913](https://github.com/user-attachments/assets/f75827e3-389f-473b-85ec-cb303b330bfb)


## LABEL ENCODER
```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
```
```python
dfc
```
![Screenshot 2025-04-19 195922](https://github.com/user-attachments/assets/dd66c565-e4ab-45f0-acf0-1820c8a41ae5)

## OneHotEncoder
```python
from sklearn.preprocessing import OneHotEncoder
```
```python
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
```
```python
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
```
```python
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2025-04-19 195933](https://github.com/user-attachments/assets/96fa0583-601c-4dcc-b58e-6c0234a493f7)

```python
pd.get_dummies(df2,columns=["nom_0"],dtype=float)
```
![Screenshot 2025-04-19 195945](https://github.com/user-attachments/assets/924a7238-f321-4422-b82c-65d78e564918)

## BINARY ENCODER
```python
pip install --upgrade category_encoders
```
![Screenshot 2025-04-19 200106](https://github.com/user-attachments/assets/2471739d-6b6c-490e-94a4-d59d8a519c48)

```python
from category_encoders import BinaryEncoder
```
```python
df=pd.read_csv("/content/data (2).csv")
df
```
![Screenshot 2025-04-19 200118](https://github.com/user-attachments/assets/cf6e8819-7869-4d80-9acf-5630882fc39f)

```python
be=BinaryEncoder()
nb=be.fit_transform(df['Ord_2'])
```
```python
dfb=pd.concat([df,nb],axis=1)
dfb1=df.copy()
dfb
```
![Screenshot 2025-04-19 200129](https://github.com/user-attachments/assets/9167fab6-6dd2-4dc6-8a8f-cd17cde5882c)

## TARGET ENCODER
```python
from category_encoders import TargetEncoder
```
```python
te=TargetEncoder()
cc=df.copy()
```
```python
new=te.fit_transform(X=cc['City'],y=cc['Target'])
new
```
![Screenshot 2025-04-19 200140](https://github.com/user-attachments/assets/b26cb0b0-ed72-402b-ba38-ccbdb655c381)

```python
cc=pd.concat([cc,new],axis=1)
cc
```
![Screenshot 2025-04-19 200150](https://github.com/user-attachments/assets/fa7b6b95-f5b7-4b02-bc4a-9cf72c354ddb)

## FEATURE TRANSFORMATION
```python
import pandas as pd
from scipy import stats
import numpy as np
```
```python
df=pd.read_csv("/content/Data_to_Transform (1).csv")
df
```
![Screenshot 2025-04-19 200201](https://github.com/user-attachments/assets/2703b1ec-fc12-44f0-88a7-a70d8e691f5d)

```python
df.skew()
```
![Screenshot 2025-04-19 200209](https://github.com/user-attachments/assets/f2e02d77-31fd-4431-9c0c-7459091c80f6)

```python
np.log(df["Highly Positive Skew"])
```
![Screenshot 2025-04-19 200218](https://github.com/user-attachments/assets/93b50ee7-2a0e-47f8-a32f-83d85168d354)

```python
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2025-04-19 200226](https://github.com/user-attachments/assets/ef7c1c4c-57f0-49b3-bb6b-235be36109fd)

```python
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2025-04-19 200233](https://github.com/user-attachments/assets/36c18372-4274-4c21-bf82-b342bdab0157)

```python
np.square(df["Highly Positive Skew"])
```
![Screenshot 2025-04-19 200240](https://github.com/user-attachments/assets/18858f01-4c24-49b4-a98a-de892b7525c8)

```python
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2025-04-19 200251](https://github.com/user-attachments/assets/ee0c9043-203e-41f3-a804-673d932c00d4)

```python
df.skew()
```
![Screenshot 2025-04-19 200303](https://github.com/user-attachments/assets/d710ce54-fc59-4fd8-903e-d856838ee58f)

```python
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
```
```python
df.skew()
```
![Screenshot 2025-04-19 200309](https://github.com/user-attachments/assets/ff979e47-f9aa-4a12-8836-f6b7914f32cc)

```python
df
```
![Screenshot 2025-04-19 200324](https://github.com/user-attachments/assets/3aef1e03-636d-46bc-915f-b74fc4e61bfa)

```python
from sklearn.preprocessing import QuantileTransformer
```
```python
qt=QuantileTransformer(output_distribution="normal")
df["Moderate Negative Skew_1"]=qt.fit_transform(df[['Moderate Negative Skew']])
df
```
![Screenshot 2025-04-19 200338](https://github.com/user-attachments/assets/2a08e497-d44b-4725-a6f2-4ac0aa2f13f0)

```python
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
```
```python
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
```
![Screenshot 2025-04-19 200348](https://github.com/user-attachments/assets/a55c45b1-5e8f-41d9-bee7-103abe05a2e1)

```python
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2025-04-19 200355](https://github.com/user-attachments/assets/770d0c05-8973-4230-8240-13a4c05490ca)

```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution="normal",n_quantiles=891)
```
```python
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
```
```python
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-19 200402](https://github.com/user-attachments/assets/634730fd-36f9-444b-a9f1-02be78ef4def)

```python
dt=pd.read_csv("/content/titanic_dataset (2).csv")
```
```python
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![Screenshot 2025-04-19 200411](https://github.com/user-attachments/assets/ef63cc03-920e-4962-aed2-8a059db74101)

```python
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![Screenshot 2025-04-19 200417](https://github.com/user-attachments/assets/138245a8-4914-46bc-afab-46550b5e4bdc)

```python
dt=pd.read_csv("/content/titanic_dataset (2).csv")
```
```python
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
```
```python
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
```
```python
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![Screenshot 2025-04-19 200424](https://github.com/user-attachments/assets/55d6a957-4dab-457c-9d26-36c181335041)

```python
sm.qqplot(dt['Age_1'],line='45')
plt.show()
```
![Screenshot 2025-04-19 200431](https://github.com/user-attachments/assets/077a10b0-914b-4342-a4f5-5d82e48bffa1)

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       

# Ex.No-3b-DATA PREPROCESSING
## Aim
To perform data preprocessing on a dataset using Python and Scikit-learn by handling missing values, encoding categorical data, splitting the dataset, and applying feature scaling. 
## Procedure
1.	Import the required Python libraries. 
2.	Mount Google Drive and load the dataset using Pandas. 
3.	Display the first few records of the dataset. 
4.	Inspect the dataset using df.info() and df.shape. 
5.	Separate the independent variables (X) and dependent variable (Y). 
6.	Convert the independent variables into an array. 
7.	Identify and handle missing values using SimpleImputer with the mean strategy. 
8.	Encode the categorical Country column using LabelEncoder. 
9.	Apply One-Hot Encoding to convert categorical country values into dummy variables. 
10.	Encode the dependent variable Purchased using LabelEncoder. 
11.	Split the dataset into training and testing sets using train_test_split. 
12.	Apply StandardScaler for feature scaling. 
13.	Display the preprocessed training and testing datasets. 
### Program
```
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
df = pd.read_csv('/content/drive/My Drive/Data.csv')
df.head()


print("Dataset Information:")
df.info()

print("\nDataset Shape:")
print(df.shape)


x = df[['Country', 'Age', 'Salary']]
y = df[['Purchased']].values

x = df[['Country', 'Age', 'Salary']].values

print("Independent Variable X:")
print(x)

print("\nDependent Variable Y:")
print(y)


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
missing_values=np.nan,
strategy='mean'
)

imputer.fit(x[:, 1:3])

x[:, 1:3] = imputer.transform(x[:, 1:3])

print("After Handling Missing Values:")
print(x)


from sklearn.preprocessing import LabelEncoder

label_encoder_x = LabelEncoder()

x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

print("After Label Encoding:")
print(x)


from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()

x_country = onehotencoder.fit_transform(
df.Country.values.reshape(-1, 1)
).toarray()

print("One-Hot Encoded Country:")
print(x_country)

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

 print("\nEncoded Dependent Variable:")
print(y)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
x,
y,
test_size=0.2,
random_state=0
)

print("X Training Data:")
print(x_train)

print("\nX Testing Data:")
print(x_test)

print("\nY Training Data:")
print(y_train)

print("\nY Testing Data:")
print(y_test)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)

print("Scaled X Training Data:")
print(x_train)

print("\nScaled X Testing Data:")
print(x_test)
```
## Output
<img width="357" height="246" alt="image" src="https://github.com/user-attachments/assets/943b8729-45a6-4a00-8174-7534c4e0eaf1" />
<img width="422" height="327" alt="image" src="https://github.com/user-attachments/assets/3760d42a-895c-4525-af5c-f1534364e17f" />
<img width="280" height="522" alt="image" src="https://github.com/user-attachments/assets/f1f17291-6f77-4608-82da-a102ad16bb2d" />
<img width="412" height="251" alt="image" src="https://github.com/user-attachments/assets/8c21794a-5f14-4386-b10b-06ba213bd7f8" />
<img width="281" height="311" alt="image" src="https://github.com/user-attachments/assets/c81090df-fd5d-487f-ad8c-1635f8e55a3c" />
<img width="310" height="422" alt="image" src="https://github.com/user-attachments/assets/191b9ad6-4a07-4940-9def-6f43a2e324f3" />
<img width="440" height="291" alt="image" src="https://github.com/user-attachments/assets/3525b746-75df-4d82-92b6-44df7cec3b82" />

## Conclusion
Thus, the given dataset was successfully preprocessed by handling missing values, encoding categorical variables, splitting the data into training and testing sets, and performing feature scaling.


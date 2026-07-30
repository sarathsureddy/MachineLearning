```python
import numpy as np
```


```python
    import numpy as np
    arr1 = np.array([10, 20, 30, 40, 50])
    print("1D Array:", arr1)
    arr2 = np.array([[1, 2, 3],
                     [4, 5, 6]])
    print("\n2D Array:\n", arr2)
    print("\nZeros Array:")
    print(np.zeros((2, 3)))
    print("\nOnes Array:")
    print(np.ones((3, 3)))
    print("\nIdentity Matrix:")
    print(np.eye(4))
    print("\nArray with Range:")
    print(np.arange(1, 11))
    print("\nEven Numbers:")
    print(np.arange(2, 21, 2))
    print("\nLinearly Spaced Values:")
    print(np.linspace(0, 10, 5))
    print("\nShape:", arr2.shape)
    print("Size:", arr2.size)
    print("Dimensions:", arr2.ndim)
    print("Data Type:", arr2.dtype)
    print("\nAddition:", arr1 + 5)
    print("Subtraction:", arr1 - 5)
    print("Multiplication:", arr1 * 2)
    print("Division:", arr1 / 2)
    print("Square:", arr1 ** 2)
    print("Square Root:", np.sqrt(arr1))
    print("\nSum:", np.sum(arr1))
    print("Mean:", np.mean(arr1))
    print("Median:", np.median(arr1))
    print("Maximum:", np.max(arr1))
    print("Minimum:", np.min(arr1))
    print("Standard Deviation:", np.std(arr1))
    print("Variance:", np.var(arr1))
```

    1D Array: [10 20 30 40 50]
    
    2D Array:
     [[1 2 3]
     [4 5 6]]
    
    Zeros Array:
    [[0. 0. 0.]
     [0. 0. 0.]]
    
    Ones Array:
    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    
    Identity Matrix:
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    
    Array with Range:
    [ 1  2  3  4  5  6  7  8  9 10]
    
    Even Numbers:
    [ 2  4  6  8 10 12 14 16 18 20]
    
    Linearly Spaced Values:
    [ 0.   2.5  5.   7.5 10. ]
    
    Shape: (2, 3)
    Size: 6
    Dimensions: 2
    Data Type: int32
    
    Addition: [15 25 35 45 55]
    Subtraction: [ 5 15 25 35 45]
    Multiplication: [ 20  40  60  80 100]
    Division: [ 5. 10. 15. 20. 25.]
    Square: [ 100  400  900 1600 2500]
    Square Root: [3.16227766 4.47213595 5.47722558 6.32455532 7.07106781]
    
    Sum: 150
    Mean: 30.0
    Median: 30.0
    Maximum: 50
    Minimum: 10
    Standard Deviation: 14.142135623730951
    Variance: 200.0
    


```python

print("\nFirst Element:", arr1[0])
print("Last Element:", arr1[-1])
print("Elements from Index 1 to 3:", arr1[1:4])
a = np.arange(1, 13)
print("\nOriginal Array:")
print(a)

print("\nReshaped to 3x4:")
print(a.reshape(3, 4))
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("\nConcatenated Array:")
print(np.concatenate((a, b)))
arr = np.array([8, 3, 9, 1, 5])
print("\nOriginal:", arr)
print("Sorted:", np.sort(arr))
print("\nRandom Integers:")
print(np.random.randint(1, 100, 5))

print("\nRandom Decimal Numbers:")
print(np.random.rand(3))
A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

print("\nMatrix Addition:")
print(A + B)

print("\nMatrix Subtraction:")
print(A - B)

print("\nMatrix Multiplication:")
print(np.dot(A, B))
print("\nTranspose of Matrix A:")
print(A.T)

print("\nElements Greater Than 25:")
print(arr1 > 25)

print("\nElements Greater Than 25:")
print(arr1[arr1 > 25])
matrix = np.array([[2, 4, 6],
                   [8, 10, 12]])

print("\nRow-wise Sum:")
print(np.sum(matrix, axis=1))

print("\nColumn-wise Sum:")
print(np.sum(matrix, axis=0))
```

    
    First Element: 10
    Last Element: 50
    Elements from Index 1 to 3: [20 30 40]
    
    Original Array:
    [ 1  2  3  4  5  6  7  8  9 10 11 12]
    
    Reshaped to 3x4:
    [[ 1  2  3  4]
     [ 5  6  7  8]
     [ 9 10 11 12]]
    
    Concatenated Array:
    [1 2 3 4 5 6]
    
    Original: [8 3 9 1 5]
    Sorted: [1 3 5 8 9]
    
    Random Integers:
    [26 34 24  9 92]
    
    Random Decimal Numbers:
    [0.00996651 0.75438094 0.52338487]
    
    Matrix Addition:
    [[ 6  8]
     [10 12]]
    
    Matrix Subtraction:
    [[-4 -4]
     [-4 -4]]
    
    Matrix Multiplication:
    [[19 22]
     [43 50]]
    
    Transpose of Matrix A:
    [[1 3]
     [2 4]]
    
    Elements Greater Than 25:
    [False False  True  True  True]
    
    Elements Greater Than 25:
    [30 40 50]
    
    Row-wise Sum:
    [12 30]
    
    Column-wise Sum:
    [10 14 18]
    


```python

import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
print("First 5 Rows:")
print(df.head())
print("\nLast 5 Rows:")
print(df.tail())
print("\nRandom 5 Rows:")
print(df.sample(5))
print("\nShape of Dataset:")
print(df.shape)

print("\nNumber of Rows:")
print(df.shape[0])

print("\nNumber of Columns:")
print(df.shape[1])

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)
print("\nDataset Information:")
df.info()
print("\nStatistical Summary:")
print(df.describe())
print("\nSummary Including Categorical Columns:")
print(df.describe(include='all'))
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:")
print(df.duplicated().sum())
print("\nUnique Values in Each Column:")
print(df.nunique())
print("\nUnique Passenger Classes:")
print(df["class"].unique())
```

    First 5 Rows:
       survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
    0         0       3    male  22.0      1      0   7.2500        S  Third   
    1         1       1  female  38.0      1      0  71.2833        C  First   
    2         1       3  female  26.0      0      0   7.9250        S  Third   
    3         1       1  female  35.0      1      0  53.1000        S  First   
    4         0       3    male  35.0      0      0   8.0500        S  Third   
    
         who  adult_male deck  embark_town alive  alone  
    0    man        True  NaN  Southampton    no  False  
    1  woman       False    C    Cherbourg   yes  False  
    2  woman       False  NaN  Southampton   yes   True  
    3  woman       False    C  Southampton   yes  False  
    4    man        True  NaN  Southampton    no   True  
    
    Last 5 Rows:
         survived  pclass     sex   age  sibsp  parch   fare embarked   class  \
    886         0       2    male  27.0      0      0  13.00        S  Second   
    887         1       1  female  19.0      0      0  30.00        S   First   
    888         0       3  female   NaN      1      2  23.45        S   Third   
    889         1       1    male  26.0      0      0  30.00        C   First   
    890         0       3    male  32.0      0      0   7.75        Q   Third   
    
           who  adult_male deck  embark_town alive  alone  
    886    man        True  NaN  Southampton    no   True  
    887  woman       False    B  Southampton   yes   True  
    888  woman       False  NaN  Southampton    no  False  
    889    man        True    C    Cherbourg   yes   True  
    890    man        True  NaN   Queenstown    no   True  
    
    Random 5 Rows:
         survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
    38          0       3  female  18.0      2      0  18.0000        S  Third   
    65          1       3    male   NaN      1      1  15.2458        C  Third   
    725         0       3    male  20.0      0      0   8.6625        S  Third   
    863         0       3  female   NaN      8      2  69.5500        S  Third   
    510         1       3    male  29.0      0      0   7.7500        Q  Third   
    
           who  adult_male deck  embark_town alive  alone  
    38   woman       False  NaN  Southampton    no  False  
    65     man        True  NaN    Cherbourg   yes  False  
    725    man        True  NaN  Southampton    no   True  
    863  woman       False  NaN  Southampton    no  False  
    510    man        True  NaN   Queenstown   yes   True  
    
    Shape of Dataset:
    (891, 15)
    
    Number of Rows:
    891
    
    Number of Columns:
    15
    
    Column Names:
    Index(['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
           'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town',
           'alive', 'alone'],
          dtype='object')
    
    Data Types:
    survived          int64
    pclass            int64
    sex              object
    age             float64
    sibsp             int64
    parch             int64
    fare            float64
    embarked         object
    class          category
    who              object
    adult_male         bool
    deck           category
    embark_town      object
    alive            object
    alone              bool
    dtype: object
    
    Dataset Information:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 15 columns):
     #   Column       Non-Null Count  Dtype   
    ---  ------       --------------  -----   
     0   survived     891 non-null    int64   
     1   pclass       891 non-null    int64   
     2   sex          891 non-null    object  
     3   age          714 non-null    float64 
     4   sibsp        891 non-null    int64   
     5   parch        891 non-null    int64   
     6   fare         891 non-null    float64 
     7   embarked     889 non-null    object  
     8   class        891 non-null    category
     9   who          891 non-null    object  
     10  adult_male   891 non-null    bool    
     11  deck         203 non-null    category
     12  embark_town  889 non-null    object  
     13  alive        891 non-null    object  
     14  alone        891 non-null    bool    
    dtypes: bool(2), category(2), float64(2), int64(4), object(5)
    memory usage: 80.7+ KB
    
    Statistical Summary:
             survived      pclass         age       sibsp       parch        fare
    count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
    mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
    std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
    min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
    25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
    50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
    75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
    max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
    
    Summary Including Categorical Columns:
              survived      pclass   sex         age       sibsp       parch  \
    count   891.000000  891.000000   891  714.000000  891.000000  891.000000   
    unique         NaN         NaN     2         NaN         NaN         NaN   
    top            NaN         NaN  male         NaN         NaN         NaN   
    freq           NaN         NaN   577         NaN         NaN         NaN   
    mean      0.383838    2.308642   NaN   29.699118    0.523008    0.381594   
    std       0.486592    0.836071   NaN   14.526497    1.102743    0.806057   
    min       0.000000    1.000000   NaN    0.420000    0.000000    0.000000   
    25%       0.000000    2.000000   NaN   20.125000    0.000000    0.000000   
    50%       0.000000    3.000000   NaN   28.000000    0.000000    0.000000   
    75%       1.000000    3.000000   NaN   38.000000    1.000000    0.000000   
    max       1.000000    3.000000   NaN   80.000000    8.000000    6.000000   
    
                  fare embarked  class  who adult_male deck  embark_town alive  \
    count   891.000000      889    891  891        891  203          889   891   
    unique         NaN        3      3    3          2    7            3     2   
    top            NaN        S  Third  man       True    C  Southampton    no   
    freq           NaN      644    491  537        537   59          644   549   
    mean     32.204208      NaN    NaN  NaN        NaN  NaN          NaN   NaN   
    std      49.693429      NaN    NaN  NaN        NaN  NaN          NaN   NaN   
    min       0.000000      NaN    NaN  NaN        NaN  NaN          NaN   NaN   
    25%       7.910400      NaN    NaN  NaN        NaN  NaN          NaN   NaN   
    50%      14.454200      NaN    NaN  NaN        NaN  NaN          NaN   NaN   
    75%      31.000000      NaN    NaN  NaN        NaN  NaN          NaN   NaN   
    max     512.329200      NaN    NaN  NaN        NaN  NaN          NaN   NaN   
    
           alone  
    count    891  
    unique     2  
    top     True  
    freq     537  
    mean     NaN  
    std      NaN  
    min      NaN  
    25%      NaN  
    50%      NaN  
    75%      NaN  
    max      NaN  
    
    Missing Values:
    survived         0
    pclass           0
    sex              0
    age            177
    sibsp            0
    parch            0
    fare             0
    embarked         2
    class            0
    who              0
    adult_male       0
    deck           688
    embark_town      2
    alive            0
    alone            0
    dtype: int64
    
    Duplicate Rows:
    107
    
    Unique Values in Each Column:
    survived         2
    pclass           3
    sex              2
    age             88
    sibsp            7
    parch            7
    fare           248
    embarked         3
    class            3
    who              3
    adult_male       2
    deck             7
    embark_town      3
    alive            2
    alone            2
    dtype: int64
    
    Unique Passenger Classes:
    ['Third', 'First', 'Second']
    Categories (3, object): ['First', 'Second', 'Third']
    


```python
import pandas as pd
from sklearn.datasets import load_iris
iris_raw = load_iris()
iris_df = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)

iris_df['species_id'] = iris_raw.target
species_mapping = {i: name for i, name in enumerate(iris_raw.target_names)}
iris_df['species_name'] = iris_df['species_id'].map(species_mapping)
print("Iris dimensions:", iris_df.shape)
print("Missing entries checking:\n", iris_df.isnull().sum())
print(iris_df.head(3))
for col in iris_raw.feature_names:
    mean_val = iris_df[col].mean()
    median_val = iris_df[col].median()
    mode_val = iris_df[col].mode()[0]
    print(f"Feature: {col}")
    print(f"  Mean:   {mean_val:.4f} cm")
    print(f"  Median: {median_val:.4f} cm")
    print(f"  Mode:   {mode_val:.4f} cm")
    print("-" * 40)
for col in iris_raw.feature_names:
    variance = iris_df[col].var()
    std_dev = iris_df[col].std()
    data_range = iris_df[col].max() - iris_df[col].min()
    q1 = iris_df[col].quantile(0.25)
    q3 = iris_df[col].quantile(0.75)
    print(f"Feature: {col}")
    print(f"  Variance:           {variance:.4f}")
    print(f"  Std Deviation:      {std_dev:.4f}")
    print(f"  Range (Max - Min):  {data_range:.4f}")
    print(f"  IQR (Q3 - Q1):      {q3 - q1:.4f}")
    print("-" * 40)
for col in iris_raw.feature_names:
    skewness = iris_df[col].skew()
    kurtosis = iris_df[col].kurt()
    print(f"Feature: {col}")
    print(f"  Skewness (Asymmetry): {skewness:.4f}")
    print(f"  Kurtosis (Tails):     {kurtosis:.4f}")
    print("-" * 40)
```

    Iris dimensions: (150, 6)
    Missing entries checking:
     sepal length (cm)    0
    sepal width (cm)     0
    petal length (cm)    0
    petal width (cm)     0
    species_id           0
    species_name         0
    dtype: int64
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \
    0                5.1               3.5                1.4               0.2   
    1                4.9               3.0                1.4               0.2   
    2                4.7               3.2                1.3               0.2   
    
       species_id species_name  
    0           0       setosa  
    1           0       setosa  
    2           0       setosa  
    Feature: sepal length (cm)
      Mean:   5.8433 cm
      Median: 5.8000 cm
      Mode:   5.0000 cm
    ----------------------------------------
    Feature: sepal width (cm)
      Mean:   3.0573 cm
      Median: 3.0000 cm
      Mode:   3.0000 cm
    ----------------------------------------
    Feature: petal length (cm)
      Mean:   3.7580 cm
      Median: 4.3500 cm
      Mode:   1.4000 cm
    ----------------------------------------
    Feature: petal width (cm)
      Mean:   1.1993 cm
      Median: 1.3000 cm
      Mode:   0.2000 cm
    ----------------------------------------
    Feature: sepal length (cm)
      Variance:           0.6857
      Std Deviation:      0.8281
      Range (Max - Min):  3.6000
      IQR (Q3 - Q1):      1.3000
    ----------------------------------------
    Feature: sepal width (cm)
      Variance:           0.1900
      Std Deviation:      0.4359
      Range (Max - Min):  2.4000
      IQR (Q3 - Q1):      0.5000
    ----------------------------------------
    Feature: petal length (cm)
      Variance:           3.1163
      Std Deviation:      1.7653
      Range (Max - Min):  5.9000
      IQR (Q3 - Q1):      3.5000
    ----------------------------------------
    Feature: petal width (cm)
      Variance:           0.5810
      Std Deviation:      0.7622
      Range (Max - Min):  2.4000
      IQR (Q3 - Q1):      1.5000
    ----------------------------------------
    Feature: sepal length (cm)
      Skewness (Asymmetry): 0.3149
      Kurtosis (Tails):     -0.5521
    ----------------------------------------
    Feature: sepal width (cm)
      Skewness (Asymmetry): 0.3190
      Kurtosis (Tails):     0.2282
    ----------------------------------------
    Feature: petal length (cm)
      Skewness (Asymmetry): -0.2749
      Kurtosis (Tails):     -1.4021
    ----------------------------------------
    Feature: petal width (cm)
      Skewness (Asymmetry): -0.1030
      Kurtosis (Tails):     -1.3406
    ----------------------------------------
    


```python
grouped_species = iris_df.groupby('species_name')
print("--- Group-Wise Mean Values ---")
print(grouped_species[iris_raw.feature_names].mean())
print("\n--- Group-Wise Standard Deviation Profiles ---")
print(grouped_species[iris_raw.feature_names].std())
aggregated_stats = iris_df.groupby('species_name')['petal length (cm)'].agg(['mean', 'median', 'std', 'skew'])
print("Aggregated Statistics (Petal Length):\n", aggregated_stats)
```

    --- Group-Wise Mean Values ---
                  sepal length (cm)  sepal width (cm)  petal length (cm)  \
    species_name                                                           
    setosa                    5.006             3.428              1.462   
    versicolor                5.936             2.770              4.260   
    virginica                 6.588             2.974              5.552   
    
                  petal width (cm)  
    species_name                    
    setosa                   0.246  
    versicolor               1.326  
    virginica                2.026  
    
    --- Group-Wise Standard Deviation Profiles ---
                  sepal length (cm)  sepal width (cm)  petal length (cm)  \
    species_name                                                           
    setosa                 0.352490          0.379064           0.173664   
    versicolor             0.516171          0.313798           0.469911   
    virginica              0.635880          0.322497           0.551895   
    
                  petal width (cm)  
    species_name                    
    setosa                0.105386  
    versicolor            0.197753  
    virginica             0.274650  
    Aggregated Statistics (Petal Length):
                    mean  median       std      skew
    species_name                                   
    setosa        1.462    1.50  0.173664  0.106394
    versicolor    4.260    4.35  0.469911 -0.606508
    virginica     5.552    5.55  0.551895  0.549445
    


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset("iris")
print(df.head())
```

       sepal_length  sepal_width  petal_length  petal_width species
    0           5.1          3.5           1.4          0.2  setosa
    1           4.9          3.0           1.4          0.2  setosa
    2           4.7          3.2           1.3          0.2  setosa
    3           4.6          3.1           1.5          0.2  setosa
    4           5.0          3.6           1.4          0.2  setosa
    


```python

plt.figure(figsize=(6,4))
sns.histplot(df["sepal_length"], bins=15, kde=True)
plt.title("Histogram of Sepal Length 24EU01124")
plt.show()
plt.figure(figsize=(6,4))
sns.countplot(x="species", data=df)
plt.title("Species Count 24EU01124")
plt.show()
plt.figure(figsize=(6,4))
sns.boxplot(y="petal_length", data=df)
plt.title("Box Plot of Petal Length 24EU01124")
plt.show()
plt.figure(figsize=(6,4))
sns.violinplot(y="petal_width", data=df)
plt.title("Violin Plot of Petal Width 24EU01124")
plt.show()
plt.figure(figsize=(6,4))
sns.kdeplot(df["sepal_width"], fill=True)
plt.title("Density Plot of Sepal Width 24EU01124")
plt.show()
```


    
![png](output_7_0.png)
    



    
![png](output_7_1.png)
    



    
![png](output_7_2.png)
    



    
![png](output_7_3.png)
    



    
![png](output_7_4.png)
    



```python

plt.figure(figsize=(6,4))
sns.scatterplot(x="sepal_length", y="petal_length",
                hue="species", data=df)
plt.title("Sepal Length vs Petal Length 24EU01124")
plt.show()
sns.pairplot(df, hue="species")
plt.show()
plt.figure(figsize=(7,5))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap 24EU01124")
plt.show()
plt.figure(figsize=(6,4))
sns.boxplot(x="species", y="petal_length", data=df)
plt.title("Petal Length by Species 24EU01124")
plt.show()
plt.figure(figsize=(6,4))
sns.violinplot(x="species", y="sepal_width", data=df)
plt.title("Sepal Width by Species 24EU01124")
plt.show()
plt.figure(figsize=(6,4))
sns.barplot(x="species", y="petal_width", data=df)
plt.title("Average Petal Width by Species 24EU01124")
plt.show()
```


    
![png](output_8_0.png)
    



    
![png](output_8_1.png)
    



    
![png](output_8_2.png)
    



    
![png](output_8_3.png)
    



    
![png](output_8_4.png)
    



    
![png](output_8_5.png)
    



```python
import pandas as pd
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
df = pd.read_csv(url, header=None)
df.rename(columns={60: "Target"}, inplace=True)
print(df.head())
```

            0       1       2       3       4       5       6       7       8  \
    0  0.0200  0.0371  0.0428  0.0207  0.0954  0.0986  0.1539  0.1601  0.3109   
    1  0.0453  0.0523  0.0843  0.0689  0.1183  0.2583  0.2156  0.3481  0.3337   
    2  0.0262  0.0582  0.1099  0.1083  0.0974  0.2280  0.2431  0.3771  0.5598   
    3  0.0100  0.0171  0.0623  0.0205  0.0205  0.0368  0.1098  0.1276  0.0598   
    4  0.0762  0.0666  0.0481  0.0394  0.0590  0.0649  0.1209  0.2467  0.3564   
    
            9  ...      51      52      53      54      55      56      57  \
    0  0.2111  ...  0.0027  0.0065  0.0159  0.0072  0.0167  0.0180  0.0084   
    1  0.2872  ...  0.0084  0.0089  0.0048  0.0094  0.0191  0.0140  0.0049   
    2  0.6194  ...  0.0232  0.0166  0.0095  0.0180  0.0244  0.0316  0.0164   
    3  0.1264  ...  0.0121  0.0036  0.0150  0.0085  0.0073  0.0050  0.0044   
    4  0.4459  ...  0.0031  0.0054  0.0105  0.0110  0.0015  0.0072  0.0048   
    
           58      59  Target  
    0  0.0090  0.0032       R  
    1  0.0052  0.0044       R  
    2  0.0095  0.0078       R  
    3  0.0040  0.0117       R  
    4  0.0107  0.0094       R  
    
    [5 rows x 61 columns]
    


```python
print("Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:")
print(df.duplicated().sum())
print("\nClass Distribution:")
print(df["Target"].value_counts())
```

    Shape: (208, 61)
    
    Data Types:
    0         float64
    1         float64
    2         float64
    3         float64
    4         float64
               ...   
    56        float64
    57        float64
    58        float64
    59        float64
    Target     object
    Length: 61, dtype: object
    
    Statistical Summary:
                   0           1           2           3           4           5   \
    count  208.000000  208.000000  208.000000  208.000000  208.000000  208.000000   
    mean     0.029164    0.038437    0.043832    0.053892    0.075202    0.104570   
    std      0.022991    0.032960    0.038428    0.046528    0.055552    0.059105   
    min      0.001500    0.000600    0.001500    0.005800    0.006700    0.010200   
    25%      0.013350    0.016450    0.018950    0.024375    0.038050    0.067025   
    50%      0.022800    0.030800    0.034300    0.044050    0.062500    0.092150   
    75%      0.035550    0.047950    0.057950    0.064500    0.100275    0.134125   
    max      0.137100    0.233900    0.305900    0.426400    0.401000    0.382300   
    
                   6           7           8           9   ...          50  \
    count  208.000000  208.000000  208.000000  208.000000  ...  208.000000   
    mean     0.121747    0.134799    0.178003    0.208259  ...    0.016069   
    std      0.061788    0.085152    0.118387    0.134416  ...    0.012008   
    min      0.003300    0.005500    0.007500    0.011300  ...    0.000000   
    25%      0.080900    0.080425    0.097025    0.111275  ...    0.008425   
    50%      0.106950    0.112100    0.152250    0.182400  ...    0.013900   
    75%      0.154000    0.169600    0.233425    0.268700  ...    0.020825   
    max      0.372900    0.459000    0.682800    0.710600  ...    0.100400   
    
                   51          52          53          54          55          56  \
    count  208.000000  208.000000  208.000000  208.000000  208.000000  208.000000   
    mean     0.013420    0.010709    0.010941    0.009290    0.008222    0.007820   
    std      0.009634    0.007060    0.007301    0.007088    0.005736    0.005785   
    min      0.000800    0.000500    0.001000    0.000600    0.000400    0.000300   
    25%      0.007275    0.005075    0.005375    0.004150    0.004400    0.003700   
    50%      0.011400    0.009550    0.009300    0.007500    0.006850    0.005950   
    75%      0.016725    0.014900    0.014500    0.012100    0.010575    0.010425   
    max      0.070900    0.039000    0.035200    0.044700    0.039400    0.035500   
    
                   57          58          59  
    count  208.000000  208.000000  208.000000  
    mean     0.007949    0.007941    0.006507  
    std      0.006470    0.006181    0.005031  
    min      0.000300    0.000100    0.000600  
    25%      0.003600    0.003675    0.003100  
    50%      0.005800    0.006400    0.005300  
    75%      0.010350    0.010325    0.008525  
    max      0.044000    0.036400    0.043900  
    
    [8 rows x 60 columns]
    
    Missing Values:
    0         0
    1         0
    2         0
    3         0
    4         0
             ..
    56        0
    57        0
    58        0
    59        0
    Target    0
    Length: 61, dtype: int64
    
    Duplicate Rows:
    0
    
    Class Distribution:
    Target
    M    111
    R     97
    Name: count, dtype: int64
    


```python
import matplotlib.pyplot as plt
import seaborn as sns
corr = df.iloc[:, :-1].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Matrix 24EU01124")
plt.show()
```


    
![png](output_11_0.png)
    



```python
print(df.corr(numeric_only=True)[0].sort_values(ascending=False))
```

    0     1.000000
    1     0.735896
    2     0.571537
    3     0.491438
    57    0.368132
    58    0.357116
    7     0.355523
    51    0.355299
    8     0.353420
    59    0.347078
    4     0.344797
    10    0.344058
    53    0.322299
    45    0.319354
    9     0.318276
    56    0.313725
    54    0.312067
    52    0.311729
    14    0.304878
    44    0.279968
    49    0.269287
    6     0.260815
    13    0.256278
    50    0.254450
    48    0.247560
    15    0.239079
    5     0.238921
    46    0.230343
    55    0.220642
    41    0.213592
    11    0.210861
    12    0.210722
    37    0.209873
    38    0.208371
    42    0.206057
    47    0.203234
    43    0.157949
    19    0.156760
    16    0.137845
    40    0.127313
    36    0.119565
    20    0.117663
    39    0.099993
    34    0.098118
    35    0.080722
    18    0.055227
    17    0.041817
    33    0.031319
    31   -0.030444
    32   -0.031939
    30   -0.048370
    21   -0.056973
    29   -0.077430
    22   -0.163426
    28   -0.199099
    23   -0.218093
    27   -0.224340
    24   -0.295683
    26   -0.341703
    25   -0.342865
    Name: 0, dtype: float64
    


```python
import sys
print(sys.executable)  # Should show ...envs\ML\python.exe
import numpy
print(numpy.__version__)  # Should show 1.26.4
```

    C:\ProgramData\anaconda3\python.exe
    1.26.4
    


```python
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target
df['Species'] = df['Species'].map({
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
})
print("\n===== First 5 Rows =====")
print(df.head())
print("\n===== Last 5 Rows =====")
print(df.tail())
print("\n===== Dataset Shape =====")
print(df.shape)
print("\nRows :", df.shape[0])
print("Columns :", df.shape[1])
print("\n===== Column Names =====")
print(df.columns.tolist())
print("\n===== Data Types =====")
print(df.dtypes)
print("\n===== Dataset Information =====")
df.info()
print("\n===== Statistical Description =====")
print(df.describe())
print("\n===== Missing Values =====")
print(df.isnull().sum())
print("\n===== Duplicate Records =====")
print(df.duplicated().sum())
print("\n===== Random 5 Samples =====")
print(df.sample(5))
print("\n===== Unique Values =====")
print(df.nunique())
print("\n===== Species Distribution =====")
print(df['Species'].value_counts())
```

    
    ===== First 5 Rows =====
       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \
    0                5.1               3.5                1.4               0.2   
    1                4.9               3.0                1.4               0.2   
    2                4.7               3.2                1.3               0.2   
    3                4.6               3.1                1.5               0.2   
    4                5.0               3.6                1.4               0.2   
    
      Species  
    0  Setosa  
    1  Setosa  
    2  Setosa  
    3  Setosa  
    4  Setosa  
    
    ===== Last 5 Rows =====
         sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \
    145                6.7               3.0                5.2               2.3   
    146                6.3               2.5                5.0               1.9   
    147                6.5               3.0                5.2               2.0   
    148                6.2               3.4                5.4               2.3   
    149                5.9               3.0                5.1               1.8   
    
           Species  
    145  Virginica  
    146  Virginica  
    147  Virginica  
    148  Virginica  
    149  Virginica  
    
    ===== Dataset Shape =====
    (150, 5)
    
    Rows : 150
    Columns : 5
    
    ===== Column Names =====
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'Species']
    
    ===== Data Types =====
    sepal length (cm)    float64
    sepal width (cm)     float64
    petal length (cm)    float64
    petal width (cm)     float64
    Species               object
    dtype: object
    
    ===== Dataset Information =====
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   sepal length (cm)  150 non-null    float64
     1   sepal width (cm)   150 non-null    float64
     2   petal length (cm)  150 non-null    float64
     3   petal width (cm)   150 non-null    float64
     4   Species            150 non-null    object 
    dtypes: float64(4), object(1)
    memory usage: 6.0+ KB
    
    ===== Statistical Description =====
           sepal length (cm)  sepal width (cm)  petal length (cm)  \
    count         150.000000        150.000000         150.000000   
    mean            5.843333          3.057333           3.758000   
    std             0.828066          0.435866           1.765298   
    min             4.300000          2.000000           1.000000   
    25%             5.100000          2.800000           1.600000   
    50%             5.800000          3.000000           4.350000   
    75%             6.400000          3.300000           5.100000   
    max             7.900000          4.400000           6.900000   
    
           petal width (cm)  
    count        150.000000  
    mean           1.199333  
    std            0.762238  
    min            0.100000  
    25%            0.300000  
    50%            1.300000  
    75%            1.800000  
    max            2.500000  
    
    ===== Missing Values =====
    sepal length (cm)    0
    sepal width (cm)     0
    petal length (cm)    0
    petal width (cm)     0
    Species              0
    dtype: int64
    
    ===== Duplicate Records =====
    1
    
    ===== Random 5 Samples =====
         sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \
    33                 5.5               4.2                1.4               0.2   
    139                6.9               3.1                5.4               2.1   
    36                 5.5               3.5                1.3               0.2   
    73                 6.1               2.8                4.7               1.2   
    64                 5.6               2.9                3.6               1.3   
    
            Species  
    33       Setosa  
    139   Virginica  
    36       Setosa  
    73   Versicolor  
    64   Versicolor  
    
    ===== Unique Values =====
    sepal length (cm)    35
    sepal width (cm)     23
    petal length (cm)    43
    petal width (cm)     22
    Species               3
    dtype: int64
    
    ===== Species Distribution =====
    Species
    Setosa        50
    Versicolor    50
    Virginica     50
    Name: count, dtype: int64
    


```python

```

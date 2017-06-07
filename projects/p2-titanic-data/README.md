
# 泰坦尼克号数据的分析项目
&emsp;&emsp;泰坦尼克号（RMS Titanic，又称铁达尼号）是一艘奥林匹克级邮轮，于1912年4月处女航时撞上冰山后沉没。泰坦尼克号由位于爱尔兰岛贝尔法斯特的哈兰德与沃尔夫造船厂兴建，是当时最大的客运轮船。在它的处女航中，泰坦尼克号从英国南安普敦出发，途经法国瑟堡-奥克特维尔以及爱尔兰昆士敦，计划中的目的地为美国纽约。1912年4月14日，船上时间夜里11时40分，泰坦尼克号撞上冰山；4月15日凌晨2时20分，船体断裂成两截后沉入大西洋，船上1500多人丧生。泰坦尼克号海难为和平时期死伤人数最惨重的海难之一。（摘至百度百科）

&emsp;&emsp;本文分析的数据：泰坦尼克号上 2224 名乘客和船员中 891 名的人口学数据和乘客基本信息。这个数据集来自 Kaggle，该网站上能够查看这个数据集的详细描述。

泰坦尼克的2224成员中有1500多人在事故中丧生，有哪些因素会让船上的人生还率更高？我们先来看看数据：


```python
# 引入 numpy 和 pandas 模块
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
titanic_data = pd.DataFrame(pd.read_csv('titanic-data.csv'))
# 查看下数据样本的前3行数据
titanic_data.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




数据集的字段注释

字段 | 注释 | 字段 | 注释 
---|---|---|---
PassengerId | 乘客ID | Survived | 存活(1存活、0死亡)
Pclass | 乘客等级(1/2/3等舱位) | Name | 乘客姓名
Sex | 性别 | Age | 年龄
SibSp | 堂兄弟/妹个数 | Parch | 父母与小孩个数
Ticket | 船票信息 | Fare | 票价
Cabin | 客舱     |       Embarked | 登船港口

根据现有的数据来看：
- PassengerId，是乘客ID应该类似于票根号，可能分析的意义不大，暂时忽略。
- Survived，是本文分析生还率最重要的数据。
- Pclass，是船舱等级，泰坦尼克号不同等级享受的待遇是天壤之别的，高等级的船舱拥有更多的福利；**船舱等级越高，生存率越高？**
- Sex和Age，在逃生时是否遵循老人儿童女士优先，**性别、年龄跟生存率的关系？**
- SibSp和Parch，逃生船是有限的，**堂兄弟/妹个数或者父母与小孩个数与生存率的关系？**
- Ticket，该数据分析的意义可能不大，暂时忽略。
- Cabin，客轮是撞了冰山海水从船底涌入，所以在底层的客舱逃生时间应该比较短，或者有些客舱离逃生出口比较远等等。**哪些客舱的生存率高？**
- Embarked，该数据可能分析的意义不确定，不同的登船港口，似乎不能说明什么，暂时忽略。


```python
# 查看下数据样本的相关信息
titanic_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB
    

通过info()可以得知，Age、Cabin、Embarked存在数据缺失。

Cabin的数据实在太少，暂时不知道如何填充，暂时忽略。Embarked是登船港口，根据之前的分析，暂时忽略。有Age信息的数据只有714，但是为了能够使用更多的数据，这里采用现有的714名有的Age数据的平均值作为余下的177条数据的Age值。


```python
# 计算714名有的Age数据的平均值
mean_age_714 = titanic_data['Age'].mean()
print "mean_age_714:", mean_age_714
# 填充平均值
titanic_data['Age'].fillna(mean_age_714,inplace=True)
```

    mean_age_714: 29.6991176471
    


```python
# 查看下Sex的分类
titanic_data.groupby(['Sex'])['Sex'].count()
```




    Sex
    female    314
    male      577
    Name: Sex, dtype: int64



Sex的数据类型是object，又根据Sex的分类得知，性别可以转换为男为1、女为0；如果男女人数相等，那么mean=0.5，如果女士比男士多，那么则mean趋于0，反之趋于1。


```python
# 男为1，女为0
titanic_data['Sex'].replace(['male','female'],[1,0],inplace=True)
```


```python
# 查看下数据样本的相关信息
# titanic_data.info()
# 查看数据集的 统计汇总
titanic_data.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>0.647587</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>0.477990</td>
      <td>13.002015</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>22.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>29.699118</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>35.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 保存已经处理的数据
titanic_data.to_csv('titanic-data.v1.csv')
```

### 从已处理的数据统计汇总信息中，可以得出以下结论：
- Survived.mean=0.38，Survived的值为1或0，所以这次事故中乘客的存活率为0.38
- Sex.mean=0.65, 说明男性乘客较多。
- Age.min最小年龄的乘客不到1岁，而Age.max最大年龄的乘客为80岁，乘客平均年龄为29岁
- Pclass.mean=2.3，舱位分为1/2/3等，而mean大于2，所以2、3等舱位数量最多

### 针对该数据样本数据探索

**a1**.年龄最小不到1岁的乘客和年龄最大80岁的乘客，他们是否在事故中幸存下来？


```python
age_for_min = titanic_data['Age'].argmin()
age_for_max = titanic_data['Age'].argmax()
# print age_for_min, age_for_max
age_min_Survived = titanic_data.iloc[[age_for_min]]
age_max_Survived = titanic_data.iloc[[age_for_max]]
print age_min_Survived['Survived']
print age_max_Survived['Survived']
```

    803    1
    Name: Survived, dtype: int64
    630    1
    Name: Survived, dtype: int64
    

年龄最大和最小的两位乘客在那次事故中幸存。

**a2**.船舱等级越高，生存率越高？


```python
# 891名乘客中，舱位1/2/3等，分别各是多少？所占比例？
by_Pclass = titanic_data.groupby(['Pclass'])['Pclass'].count()
plt.pie(by_Pclass, labels = ['Pclass1({})'.format(by_Pclass[1]),'Pclass2({})'.format(by_Pclass[2]), 'Pclass3({})'.format(by_Pclass[3])], autopct='%.1f%%')
plt.title('Pie Chart for Pclass of Passengers')
```




    <matplotlib.text.Text at 0xa3dd630>




![png](output_17_1.png)


1/2/3等舱位数量分别为216、184、491，如上图所示。


```python
# 对比3种舱位的生存情况
titanic_data.groupby(['Pclass', 'Survived'])['Survived'].count().unstack().plot(kind = 'bar')
plt.title('The number of Passengers survival group by Pclass')
```




    <matplotlib.text.Text at 0xa365588>




![png](output_19_1.png)


上面根据船舱等级分类的柱形图中，蓝色为遇难人数，橙色为存活人数；橙色柱形与蓝色柱形相差越大，存活率越低。


```python
# 定义一个生存率的绘柱状图的函数
def survival_rate(key):
    plt.title("the survival rate in " + key)
    titanic_data.groupby([key])['Survived'].mean().plot(kind = 'bar')
# 根据 Pclass乘客等级 绘制柱状图
survival_rate('Pclass')
```


![png](output_21_0.png)


综上得知，**乘客等级越高，生存率越高。**

**a3**.性别、年龄跟生存率的关系？


```python
# 根据 Sex性别 绘制柱状图
# 注：数据已处理，0为女性，1为男性
survival_rate('Sex')
```


![png](output_24_0.png)


由上图得知，**女性的生存率远高于男性**，说明逃生时执行了女士优先。

现在要探索年龄与生存率的关系，由于不知道当时评判幼年、中年、老年等年龄段的划分，已知最大年龄为80岁则不妨设置5年为年龄组距，暂且查看现有的年龄数据的直方图：


```python
# 891名乘客的年龄直方图
# plt.subplot2grid((1,2),(0,0))
titanic_data['Age'].plot.hist(bins=16,title='All Passenger Age')
# 生存的乘客年龄直方图
# plt.subplot2grid((1,2),(0,1))
titanic_data[titanic_data.Survived==1]['Age'].plot.hist(bins=16,title='Survived Passenger Age')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xa5f4dd8>




![png](output_27_1.png)


通过观察上图，蓝色是全体乘客年龄情况，而橙色是存活乘客年龄情况，两种颜色的柱形图越接近，表明存活率越高。
以下是计算具体存活的代码：


```python
# 设置年龄的分组,同上组距为5,最大年龄段为80+5
age_width = 5
bins = np.arange(0, 80+age_width, age_width)
titanic_data['Age_group'] = pd.cut(titanic_data['Age'], bins)
titanic_data.groupby(['Age_group', 'Survived'])['Survived'].count().unstack().plot(kind = 'bar')
plt.title('Age groups, width={}'.format(age_width))
```




    <matplotlib.text.Text at 0xb44b4e0>




![png](output_29_1.png)


观察以上的年龄段的分类柱形图，橙色为存活，蓝色为遇难，蓝色与橙色柱形图相差约悬殊表示存活率越低。以下是计算具体存活率的代码：


```python
titanic_data.groupby('Age_group')['Survived'].mean().plot(kind = 'bar')
plt.title("the survival rate in Age's group")
```




    <matplotlib.text.Text at 0xb952c50>




![png](output_31_1.png)



```python
print "每个龄段分组的数据如下:"
titanic_data.groupby('Age_group')['Survived'].count()
```

    每个龄段分组的数据如下:
    




    Age_group
    (0, 5]       44
    (5, 10]      20
    (10, 15]     19
    (15, 20]     96
    (20, 25]    122
    (25, 30]    285
    (30, 35]     88
    (35, 40]     67
    (40, 45]     47
    (45, 50]     39
    (50, 55]     24
    (55, 60]     18
    (60, 65]     14
    (65, 70]      3
    (70, 75]      4
    (75, 80]      1
    Name: Survived, dtype: int64



年龄在75~80之间的乘客只有1位，是幸存者，所以这个区间的存活率是100%，现在需要重新调整年龄组距：


```python
# 设置年龄的分组,同上组距为10,最大年龄段为80+10
age_width = 10
bins = np.arange(0, 80+age_width, age_width)
titanic_data['Age_group'] = pd.cut(titanic_data['Age'], bins)
titanic_data.groupby(['Age_group', 'Survived'])['Survived'].count().unstack().plot(kind = 'bar')
plt.title('Age groups, width={}'.format(age_width))
```




    <matplotlib.text.Text at 0xaf7f320>




![png](output_34_1.png)



```python
print "每个龄段分组的数据如下:"
titanic_data.groupby('Age_group')['Survived'].count()
```

    每个龄段分组的数据如下:
    




    Age_group
    (0, 10]      64
    (10, 20]    115
    (20, 30]    407
    (30, 40]    155
    (40, 50]     86
    (50, 60]     42
    (60, 70]     17
    (70, 80]      5
    Name: Survived, dtype: int64




```python
# 以下是计算具体存活率的代码：
titanic_data.groupby('Age_group')['Survived'].mean().plot(kind = 'bar')
plt.title("the survival rate in Age's group")
```




    <matplotlib.text.Text at 0xba7ac88>




![png](output_36_1.png)


从这个年龄分组组距为10的图，可以看出儿童的生存率较高，而老年人的生存率略低，中年人的生存率在50%左右。

**a4**.堂兄弟/妹个数或者父母与小孩个数与生存率的关系？


```python
# 堂兄弟/妹个数 SibSp 的 分析
# 根据堂兄弟/妹个数分组的柱形图
titanic_data.groupby(['SibSp', 'Survived'])['Survived'].count().unstack().plot(kind = 'bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xb9dc860>




![png](output_39_1.png)



```python
# 堂兄弟/妹个数分组的存活率
survival_rate('SibSp')
```


![png](output_40_0.png)



```python
# 父母与小孩个数 Parch 的分析
# 父母与小孩个数分组的柱形图
titanic_data.groupby(['Parch', 'Survived'])['Survived'].count().unstack().plot(kind = 'bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xb7e4c88>




![png](output_41_1.png)



```python
# 父母与小孩个数分组的存活率
survival_rate('Parch')
```


![png](output_42_0.png)


由图得知，堂兄弟/妹个数为2~8之间的数据太少无法下结论，可以推测个数为1时生存率较高；父母与小孩个数为3~6之间的数据太少无法下结论，可以推测个数为1~2时生存率最高。

### 结论
泰坦尼克号的事故乘客的存活率比较低为*0.38*。船舱的等级越高，存活的几率越大，反之等级越低存活的可能性越小，这也与实际相符，因为高等级的船舱地理位置、逃生空间较好。乘客中男性的存活率较低为*0.19*，而女性的存活率达到*0.74*，说明逃生时执行了女士优先。0~5岁的乘客生存率较高，很可能是因为他们从登上船到上救生艇，就没有离开过父母的怀抱，在逃生时能够被优先照顾。堂兄弟/妹个数和父母与小孩个数的越少，生存率越高，是否体现为优先逃生，尽可能减少对家庭的破坏呢？这个还暂时无法确定。

**思考**：通过Age的直方图，得知用平均值填充缺失的数据，会导致接近平均值的年龄段乘客数量剧增，也应该会影响到分析的结果。Cabin客舱的数据缺失严重，这个数据应该对客舱的设计起到很大作用，什么位置的客舱逃生条件最好，或者怎么样设计出口才更合理等等。该数据样本的的乘客信息也比较少，缺乏阶级地位、身体素质等信息。所以本文只是分析了若干因素与生存率之间的相关性。

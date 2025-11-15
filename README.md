기초통계(Basic statistics)

    - 통계는 아직 발생하지 않은 일을 예측하기 위해서 사용한다.
    - 수량적인 비교를 기초로 많은 사실을 관찰하고 처리하는 방법을 연구하는 학문이다.
    - 불균형 데이터를 대상으로 규칙성과 불규칙성을 발견한 뒤 실생활에 적용할 수 있다.

Data Analysis (데이터 분석)

<img width="915" height="373" alt="스크린샷 2025-11-12 오후 7 32 06" src="https://github.com/user-attachments/assets/314e24a7-0df9-46f3-9736-a29e915fd6de" />

    - 원시(원본 그대로) 데이터를 분석하여 인사이트(가시성 증가 및 깊은 이해)로 변환하는 작업이다.
    - 문제를 해결하기 위해서 데이터를 사용해 흐름 및 방향을 찾는 기술이다.

변량(Variable)

    - 자료의 수치를 변량이라고 하며, 데이터의 값을 의미한다.

 !pip install numpy pandas

    - numpy ,pandas 설치

    Numpy: 수치계산과 배열 연산에 특화된 라이브러리 -> import numpy as np
    Pandas: 표 형태의 데이터(행과열)를 다루는 라이브러리 -> import pandas as pd

계급(Class)

    - 변량을 일정 간격으로 나눈 구간을 의미
    - 변향의 최소값, 최대값을 잘 고려해서 계급을 정해야한다.
    - 소괄호는 구간 미포함, 대괄호는 구간 포함 ex) ( ] -> (): 미포함, []: 포함

도수(Frequency)

    - 각 계급에 속하는 변량의 개수를 의미한다.

상대 도수(Relatice frequency)

    - 각 계급에 속하는 변향의 비율을 의미한다.

도수분포표(Frequency table)

    - 주어진 자료를 계급별로 나눈 뒤 각 계급에 속하는 도수 및 상대 도수를 조사한 표이다.
    - 구간별 분포를 한 번에 알아보기 좋지만 계급별 각 변량의 정확한 값이 생략되어 있다.

히스토그램(Histogram)

    - 도수분포표를 시각화한 그래프이다.
    # !pip install matplotlib
    matplotlib.pyplot:  그래프 쉽게 그릴 수 있게 해주는 모듈

산술평균(Mean)

    - 변량의 합을 변량의 수로 나눈 값을 의미한다.

편차(Deviation)

    - 변량에서 평균을 뺀 값
    - 각 변량의 편차를 구한 뒤 모두 합하면 0이 되기 때문에 편차의 평균은 구할 수 없다.

분산(Variance)

    - 변향이 평균으로부터 떨어져있는 정도를 보기 위한 통계량이다.
    - 편차에 제곱하여 그 합을 구한 뒤 산술 평균을 낸다.

표준편차(Standard deviation)

    - 분산의 제곱근이며, 관측된 변량의 흩어진 정도를 하나의 수치로 나타내는 통계량이다.
    - 표준편차가 작을 수록 평균 값에서 변량들의 거리가 가깝다고 판단한다.

확률변수(Random variable)

    - 머신러닝, 딥러닝 등 확률을 다루는 분야에 있어서 필수적인 개념이다.
    - 확률(probability)이 있다는 뜻은 사건(event)이 있다는 뜻이며,
    - 시행(trial)을 해야 시행의 결과인 사건(event)이 나타난다.
    - 시행(trial)을 해서 어떤 사건(event)이 나타났는지에 따라 값이 정해지는 변수이다.
    - 알파벳 대문자로 표현하며, X, Y, Z 또는 X1, X2, X3과 같이 표현한다.
    - 확률변수는 집합이며, 원소를 확률변수값(Value of random variable)이라고 표현한다.
    - 확률변수에서 사용한 알파벳의 소문자를 사용한다.
    - Y = { y1, y2, y3 }, 이 때 Y는 확률변수이고 원소인 y1 ~ y3은 확률변수값이다.
    - 종류로는 범주형, 이산형, 연속형이 있다.

범주형 확률변수(Categorical random variable)

    - 수치가 아닌 기호나 언어, 숫자등으로 표현한다. (순서를 가질 수 있다.)
    - 유한집합(원소 수가 유한한 집합)으로 표현한다.

이산형 확률변수(Discrete random variable)

    - 수치로 표현하고 셀 수 있는 값이다. (양적 확률변수 혹은 수치형 확률변수라고도 부른다.)
    - 유한집합 또는 셀 수 있는 무한집합으로 표현한다.

연속형 확률변수(Continuous random variable)

    - 구간을 나타내는 수치로 표현한다.
    - 셀 수 없는 무한집합으로 표현한다.

확률분포(Probability distribution)

    - 사건에 대한 확률변수에서 정의된 모든 확률값의 분포이며, 서로 다른 모든 결과의 출현 확률을 제공한다.

확률분포표(Probability distribution table)

    - 확률변수의 모든 값(원소)에 대해 확률을 표로 표시한 것이다.
    - 범주형 또는 이산형 확률변수의 확률분포를 표현하기에 적합한 방식이다.

확률분포함수(Probability distribution function)

    - 확률변수의 분포를 나타내는 함수로서, 확률변수의 확률변수값이 나올 확률을 나타내는 함수이다.
    - 확률질량함수, 확률밀도함수 등의 함수가 있다.

확률질량 함수(Probability mass function, pmf)

    - 확률변수 X의 분포를 나타내는 함수로서, xi가 나올 확률이다.
    - 확률변수의 값을 매개변수로 전달받고, 해당 값이 나타날 확률을 구해서 리턴하는 함수이다.
    - 범주형 확률변수와 이산형 확률변수에서 사용된다.
    - 확률변수에서 각 값에 대한 확률을 나타내는 것이 마치 각 값이 "질량"을 가지고 있는 것처럼 보이기 때문에 확률질량 함수로 불린다.
    - 확률질량 함수 f는 확률변수 X가 x를 변수값으로 가질 때의 확률이다.

확률밀도 함수(Probability density function, pdf)

    - 확률변수 X의 분포를 나타내는 함수로서, 특정 구간에 속할 확률이고 이는 특정 구간을 적분한 값이다.
    - 확률변수값의 범위(구간)를 매개변수로 전달받고, 범위의 넓이를 구해서 리턴하는 함수이다.
    - 연속형 확률변수에서 사용된다.
    - 전체에 대한 확률이 아닌 구간에 포함될 확률을 나타내기 때문에 구간에 따른 밀도를 구하는 것이고, 이를 통해 확률밀도 함수라 불린다.
    - 확률밀도 함수 f는 특정 구간에 포함될 확률을 나타낸다.
    ※ CDF(cumulative distribution function): 이하 확률

정규분포(Normal distribution)
    
    - 모든 독립적인 확률변수들의 평균은 어떠한 분포에 가까워지는데, 이 분포를 정규분포라고 한다.
    - 즉, 비정규분포의 대부분은 극한상태에 있어서 정규분포에 가까워진다.

표준 정규분포(Standard normal distribution)

    - 정규분포는 평균과 표준편차에 따라서 모양이 달라진다.
    - 정규분포를 따르는 분포는 많지만 각 평균과 표준편차가 달라서 일반화할 수 없다.
    - N(μ, σ) = N(0, 1)로 만든다면 모두 같은 특성을 가지는 동일한 확률분포로 바꿔서 일반화할 수 있다.
    - 따라서 일반 정규분포를 표준 정규분포로 바꾼 뒤 표준 정규분포의 특정 구간의 넓이를 이용해서 원래 분포의 확률을 구할 수 있다.

표준화(Standardization)

    - 다양한 형태의 정규분포를 표준 정규분포로 변환하는 방법이다.
    - 표준 정규분포에 대한 값(넓이)를 이용해 원래 분포의 확률을 구할 수 있다.

모집단과 모수 (Population and population parameter)

    - 모집단이란, 정보를 얻고자 하는 대상의 전체 집합을 의미한다.
    - 모수란, 모집단의 수치적 요약값을 의미한다. 평균 또는 표준편차와 같은 모집단의 통계값을 모수라고 한다.

표본과 샘플링 (Sample and Sampling)

    - 표본이란, 모집단의 부분집합으로서 표본의 통계량을 통해 모집단의 통계량을 추론할 수 있다.
    - 모집단의 통계량을 구할 수 없는 상황 즉, 전수 조사가 불가능한 상황에서 임의의 표본을 추출하여 분석한다.
    - 이렇게 표본(sample)을 추출하는 작업을 샘플링(sampling)이라고 한다.

Numpy

    - 머신러닝 애플리케이션에서 데이터 추출, 가공, 변환과 같은 데이터 처리 부분을 담당한다.
    - numpy 기반의 사이킷런을 이해하기 위해서 필수이다.

ndarray

    - N차원(n-dimension) 배열 객체이다.
    - 파이썬 list를 array() 메소드에 전달하면 ndarray로 변환되고 numpy의 다양한 기능들을 사용할 수 있다.
    - 반드시 같은 자료형의 데이터만 담아야 한다.

shape

    - 배열의 형태 (차원별 크기) - ex) output : (2,3) - 2행 3열

ndim 
    
    - 배열의 차원 수

astype()

    - ndarray에 저장된 요소의 타입을 변환시킬 때 사용
    - 대용량 데이터 처리 시, 메모리 절약

axis

    - 축의 방향성을 표현할 때
    - 행 : 0 , 열: 1 -> 순서대로 0,1,2

arange(), zeros(), ones()

    - ndarray의 요소를 원하는 범위의 연속값, 0 또는 1로 초기화 할때 사용 

reshape()

    - ndarrary의 기존 shape를 다른 shape로 변경

flatten()
    
    - 다차원 배열을 1차원 배열로 변경(평평하게 만들어줌)


indexing (특정위치의 데이터를 가져오기)

    - 위치 인덱싱(Location indexing) 
    - 슬라이싱 (slicing) 
    - 팬시 인덱싱 (Fancy indexing)
    - 불린 인덱싱 (Boolean indexing)

sorting

    - 모두 오름차순 정렬, 내림차순은 오름차순 정렬 후 [::-1]

    행방향(0), 세로 -> axis=0
    열방향(1), 가로 -> axis=1

argsort()

    - 배열 정리했을때 인덱스 번호로 정렬


벡터 

    - 데이터 과학에서 벡터란 숫자 자료를 나열한 것을 의미한다.
    - 벡터는 공간에서 한 점을 나타낸다.
    - feature(특성) 1개당 1차원 


내적 (Dot product)

    - 두 벡터의 성분들의 곱의 합
    - w: 가중치 
    - print(np.dot(A, w))

선형대수 (Linear Algebra)

    - 연립 방정식을 표현할 때 쉽게 표현하기 위해서 선형대수를 배운다.


전치 행렬 

    - .T : transpose (행과 열을 바꿈)
    - linalg: 선형대수 inv: inverse (역행렬)

역행렬 

    - np.linalg.inv(A)
    - 행렬의 A의 역행렬을 구해주는 함수

단위행렬

    - 대각선: 1, 나머지: 0 
    - 1 0 1 0

과결정계(Overdetermined system)

    * 차원을 넘어간 평면에 위치한 점들 중, 교점과 가장 가까운 거리의 점을 찾고, 투영을 통해 해당 차원으로 축소해야 해를 찾을 수 있으며, 이 때 해의 근사치를 구할 수 있다.
    * 투영시, 원본 값에서 어느정도의 loss(손실)가 발생하지만 이를 감안하더라도 근사값을 구한다.
    * x, residuals, _, _ = np.linalg.lstsq(A, b)
    np.linalg.lstsq(A, b) -> 최소제곱근사값
    lstsq : Least Squares(최소제곱근사법)
    residuals: 잔차 (오차제곱합)


판다스(Pandas)

    - 2차원 데이터(테이블, 엑셀, CSV 등)을 효율적으로 가공 및 처리할 수 있다.

판다스 구성요소 

    - DataFrame: 행과 열로 구성된 2차우너 Dataset을 의미한다.
    - Series: 1개의 열로만 구성된 열벡터 Dataset을 의미한다.
    - Index: DataFrame과 Series에서 중복없는 행 번호를 의미한다.

DataFrame()

    - dict를 DataFrame으로 변환하고자 할 때 DataFrame 생성자에 전달한다.
    - 컬럼명을 추가하거나 인덱스명을 변경하는 등 다양하게 설정할 수 있다.

drop

    - 기존 인덱스 삭제 여부

inplace

    - 원본 DataFrame에 적용 여부 

read_csv() 

    - csv 파일을 DataFrame으로 읽어 온다.

head()

    - 전체 데이터 중 앞부분 일부를 가져온다.

tail()

    - 전체 데이터 주 뒷부분 일부를 가져온다.

iloc[], loc[]

    - 원하는 행 또는 열을 가져온다.
    - iloc은 인덱스 번호를 가져오고, loc은 인덱스 값 또는 컬럼명으로 가져온다.
    
    print(happiness_df.iloc[0])
    print(happiness_df.loc[1])

    # 한 개의 feature를 가져오면 Series이다.
    # to_frame()을 사용하면 다시 Dataframe으로 변경된다.
    print(happiness_df.iloc[:, -1])
    print(type(happiness_df.iloc[:, -1].to_frame()))
    
    print(happiness_df.loc[:, 'income'])
    print(type(happiness_df.loc[:, 'income'].to_frame()))
    
    # 대괄호로 가져올 때에는 두 번 써서 Dataframe으로 가져올 수 있다.
    print(happiness_df['income'])
    print(type(happiness_df[['income']]))

describe()

    - 숫자형 데이터의 개수, 평균, 표준편차, 최솟값, 사분위 분포도(중앙값:50%), 최댓값을 제공한다.
    - 25번째 백분위수와 75번째 백분위수를 기준으로 정상치의 범위를 설정할 수 있디.

#### 📝 실습

    import pandas as pd
    
    happiness_df = pd.read_csv('./datasets/happiness_report_2023.csv')
    display(happiness_df)

<img width="983" height="626" alt="스크린샷 2025-11-15 오후 8 25 47" src="https://github.com/user-attachments/assets/340aedaa-e43a-4625-9041-d927aca46c76" />

    - 행복지수 csv파일을 불러와서 그래프를 시각화해서 데이터 분석하기

그래프 시각화

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.scatter(x=happiness_df.gdp_per_capita, y=happiness_df.happiness_score)
    plt.xlabel('GDP')
    plt.ylabel('happiness_score')
    plt.show()

<img width="554" height="432" alt="image" src="https://github.com/user-attachments/assets/f06df18a-9b51-46a8-8348-d35645f08d57" />

#### 행복지수와 GDP 관계 그래프 해석 

    gdp_per_capita(1인당 국내총생산(GDP)을 측정)에 따른 행복지수 그래프 x축은 gdp_per_capita(1인당 국내총생산(GDP), y축은 행복지수 점수
    
    GDP가 높을 수록 행복지수 점수가 높다.
    GDP가 낮은 국가들(왼쪽 구간, 0 ~ 0.8)은 대체로 행복 점수가 3~5점 수준으로 낮다.
    GDP가 중간 정도인 국가들(1.0 ~ 1.5)은 행복 점수가 5~6점 사이에서 분포한다.
    GDP가 높은 국가들(1.8 이상)은 대부분 행복 점수가 7점 이상이다.
    그래프가 우상향하는 것을 보면 경제 수준이 높은 나라일수록 국민들의 행복감이 높게 나타난다.

#### RFM 분석

    - 사용자 별로 얼마나 최근에, 얼마나 자주, 얼마나 많은 금액을 지출했는지에 따라 사용자들의 분포를 확인하거나 사용자 그룹(또는 등급)을 나누어 분류하는 분석 기법이다.
    - 구매 가능성이 높은 고객을 선정할 때 용이한 데이터 분석 방법이며, 사용자들의 평소 구매 패턴을 기준으로 분류를 진행하기 때문에 각 사용자 그룹의 특성에 따라 차별화된 마케팅 메시지를 전달 할 수 있다.
    
    - Recency : 얼마나 최근에 구매 했는가
    - Frequency: 얼마나 자주 구매했는가
    - Monetary: 얼마나 많은 금액을 지출했는가

데이터 정보 확인

    - info()

duplicated().sum()

    - 데이터프레임에서 중복된 행의 갯수

isna().sum()

    - null값의 개수 확인

describe().T

    - 통계를 컬럼기준으로 정리

정상치 범위 구하기 

    import numpy as np
    
    Q1 = cp_df.describe().T.loc['Income', '25%']
    Q3 = cp_df.describe().T.loc['Income', '75%']
    
    iqr = Q3 - Q1
    
    lower_bound = Q1 - iqr * 1.5
    upper_bound = Q3 + iqr * 1.5
    
    if lower_bound < 0:
        lower_bound = 0
    
    print(f'정상치 범위: {lower_bound} ~ {upper_bound}')
    
정규화 (Nomalization)

    값의 범위를 0~1 사이로 변환시켜 모든 컬럼의 데이터가 평등해진다. 서로 다른 단위의 값은 비교 대상이 될 수 없다.      예를 들어, 80kg과 180cm는 비교할 수 없기 때문에 정규화를 사용해서 비교한다.

- 정규화 

        from sklearn.preprocessing import MinMaxScaler
    
        normalization = MinMaxScaler()
        rfm_normalization = normalization.fit_transform(cp_rfm_df)
        rfm_normalization

- 정규화된 코드 데이터프레임으로 변경
  
        cp_rfm_norm_df = pd.DataFrame(rfm_normalization, columns=cp_rfm_df.columns)
        cp_rfm_norm_df

#### 📝 rfm 실슴

- 쇼핑 csv파일 읽어오기

        import pandas as pd 
        
        cs_df = pd.read_csv('./datasets/customer_shopping_data.csv')
        cs_df

<img width="824" height="367" alt="스크린샷 2025-11-15 오후 9 02 36" src="https://github.com/user-attachments/assets/b293a76b-8e82-40e3-995c-005cf8eeba1c" />

- DataFrame rfm 시각화

* 회원 등급 수

<img width="589" height="455" alt="image" src="https://github.com/user-attachments/assets/cc0d41f1-15cf-4009-a833-e9b28c77f3fb" />

* 쇼핑몰의 거래 수

<img width="1028" height="398" alt="스크린샷 2025-11-15 오후 9 06 11" src="https://github.com/user-attachments/assets/1b659d12-6786-41d7-a962-2ed931786fc7" />

* 성별에 따른 등급 수 

<img width="1040" height="438" alt="스크린샷 2025-11-15 오후 9 06 30" src="https://github.com/user-attachments/assets/c2bf8432-c7aa-448a-923b-58024623dcf8" />


#### 마케팅 전략

- 등급 마케팅

        등급(Bronze, Silver, Gold, VIP, VVIP)을 만들었을 때 중 Gold 등급과 VIP등급이 비중이 높다. 등급들을 지정하고 등급별로 이벤트들을 준비해서 고객들의 만족도를 높여 고객들을 유지해야 한다.

- 집중 쇼핑몰 마케팅

        kanyon, mall of istanbul 두개의 쇼핑몰이 거래 수가 많아서 집중인 타켓 마케팅을 할 필요가 있다. 다른 쇼핑몰들은 이벤트나 프로모션을 준비해 고객들이 더 많이 찾아올 수 있도록 마케팅 전략을 가져가야 한다.

- 여성타겟 마케팅

        전체적으로 여성 회원들이 쇼핑몰을 자주 방문하고 등급이 올라갈 수록 여성이 많은 것을 봐서 여성 회원들을 집중적으로 마케팅을 할 필요가 있다.













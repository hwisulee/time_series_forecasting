# time_series_forecasting
>Python, Keras, Tensorflow 기반 LSTM을 사용한 시계열예측 머신러닝

<pre>
<code>
주제 : 미세먼지와 호흡계통 질환의 사망자 수 상관관계 분석 및 예측</br>
설명 : 대기오염 물질 농도 데이터와 호흡계통 질환의 사망자 수 데이터를 사용해 머신러닝 분석 및 예측을 통한 상관관계 탐구</br>
개발 인원 : 1명</br>
맡은 역할 : 데이터 수집 및 재가공, 관련 논문 탐구, 시스템 설계</br>
개발 기간 : '22.05.18 ~ '22.06.02</br>
개발 언어 : Python</br>
개발 환경 : Windows 10 pro, Python 3.9.13, tensorflow, Keras 2.9.0
</code>
</pre>

The supported format for this report files is hwp.<br>
[Report-download](https://drive.google.com/file/d/1HHa-d-Bx-HVgbcb2ZrPYt3a845zEzKoG/view?usp=sharing)

## Table of Contents
1. [Preview](#preview)
2. [References](#references)
3. [License](#license)

<h2 id="preview">Preview</h2>

1. 만들어진 데이터셋 Dataframe화(+ index=df['year'])

![캡처1](https://user-images.githubusercontent.com/62528282/170856101-e0e12c86-8596-4525-b426-cc35a4a10633.PNG)

2. 데이터셋에 대한 산점도행렬(year)

![sns_pairplot_year](https://user-images.githubusercontent.com/62528282/170855982-b8952376-fc80-4fef-a0ce-23f6438cac92.png)

3. 데이터셋에 대한 산점도행렬(month)

![sns_pairplot_month](https://user-images.githubusercontent.com/62528282/170855984-e76cef7b-5f33-4926-bd17-ca11c0554a79.png)

4. 데이터셋의 각 column에 대한 연도별 차트

![Figure_4](https://user-images.githubusercontent.com/62528282/170856059-dfc297b1-30bd-489c-bf43-36bdb537676e.png)

5. 데이터셋의 특성 분포 확인

![Figure_3](https://user-images.githubusercontent.com/62528282/171444660-ea7f17a7-28e8-4a3a-a5c0-9f3298566461.png)

6. LSTM 시계열예측 (12개월)

![Figure_6](https://user-images.githubusercontent.com/62528282/171444723-5f133b17-06bc-46f3-b02d-2bd6be8e6480.png)

7. LSTM 시계열예측 (24개월)

![Figure_6](https://user-images.githubusercontent.com/62528282/171444748-4547671a-58ad-460c-8991-18628b82c92d.png)

8. 모델 평가

![Figure_9](https://user-images.githubusercontent.com/62528282/171445065-f4648207-3bc5-4111-b82f-c08b33b128d8.png)

<h2 id="references">References</h2>

1. [Tensorflow - timeseries](https://www.tensorflow.org/tutorials/structured_data/time_series)
2. [Tensorflow - Linear regression](https://www.tensorflow.org/tutorials/keras/regression?hl=ko)

<h2 id="license">License</h2>

>My Project has this License

MIT License

Copyright (c) 2022 hwisulee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

# time_series_forecasting
>Python, Keras, Tensorflow 기반 LSTM을 사용한 시계열예측 머신러닝

<pre>
<code>
주제 : 미세먼지와 호흡계통 질환의 사망자 수 상관관계 분석 및 예측</br>
설명 : 대기오염 물질 농도 데이터와 호흡계통 질환의 사망자 수 데이터를 사용해 머신러닝 분석 및 예측을 통한 상관관계 탐구</br>
개발 인원 : 1명</br>
맡은 역할 : 데이터 수집 및 재가공, 관련 논문 탐구, 시스템 설계</br>
개발 기간 : '22.05.18 ~ 개발중</br>
개발 언어 : Python</br>
개발 환경 : Windows 10 pro, Python 3.9.13, tensorflow & tensorflow-gpu 2.9.1, Keras 2.9.0, CUDA 11.5, CUDNN 8.3.3
</code>
</pre>

## Table of Contents
1. [Preview](#preview)
2. [Library](#library)
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

<h2 id="library">Library</h2>

>This Project included this Library.



<h2 id="license">License</h2>

>Library has this Licenses



<br>

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

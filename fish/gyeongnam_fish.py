# 한글 지원
from matplotlib import font_manager, rc
font_path = "C:\Windows\Fonts\\gulim.ttc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('경상남도_수산생물 질병 발생 현황_20230731.csv', encoding='cp949')

# "없음"이 있는 행 삭제
fish = data[data['병명'] != '없음']

# 질병 발생 빈도를 계산
disease_counts = fish['병명'].value_counts()

# 차트 생성
plt.figure(figsize=(10, 6))
disease_counts.plot(kind='bar', color='skyblue')
plt.title('질병 발생 빈도')
plt.xlabel('질병명')
plt.ylabel('빈도')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# 차트 출력
plt.show()

# # 월별 질병 발생 건수를 질병명으로 그룹화하여 계산
# monthly_disease_counts = fish.groupby(['월', '병명'])['발생건수'].sum().unstack()
#
# # 막대 그래프 생성
# plt.figure(figsize=(12, 8))
# colors = plt.cm.tab20.colors  # 색상 팔레트
# monthly_disease_counts.plot(kind='bar', stacked=True, color=colors)
# plt.title('월별 질병 발생 건수')
# plt.xlabel('월')
# plt.ylabel('발생건수')
# plt.xticks(rotation=45)  # x축 레이블 회전
# plt.legend(title='질병', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
#
# # 그래프 출력
# plt.show()
#
#
# # 월별로 제일 많이 발생한 질병명 구하기
# most_common_diseases = fish.groupby(['연도', '월'])['병명'].agg(lambda x: x.value_counts().index[0]).reset_index()
#
# # 월별 질병 발생건수를 구하여 막대 그래프 그리기
# plt.figure(figsize=(12, 6))
# for year in most_common_diseases['연도'].unique():
#     for month in range(1, 13):
#         subset = most_common_diseases[(most_common_diseases['연도'] == year) & (most_common_diseases['월'] == month)]
#         if not subset.empty:
#             x = str(year) + '-' + str(month)
#             y = fish[(fish['연도'] == year) & (fish['월'] == month)]['발생건수'].sum()
#             label = subset.iloc[0]['병명']
#             plt.bar(x, y, label=label)
#             plt.text(x, y, label, ha='center', va='bottom')
#
# plt.xlabel('연도-월')
# plt.ylabel('발생건수')
# plt.title('월별 질병 발생건수 및 가장 많이 발생한 질병')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# 월별로 제일 많이 발생한 질병명 구하기
most_common_diseases = fish.groupby('월')['병명'].agg(lambda x: x.value_counts().index[0]).reset_index()

# 월별 질병 발생건수를 구하여 막대 그래프 그리기
plt.figure(figsize=(10, 6))
for index, row in most_common_diseases.iterrows():
    month = row['월']
    label = row['병명']
    total_cases = fish[fish['월'] == month]['발생건수'].sum()
    plt.bar(month, total_cases, label=label)
    plt.text(month, total_cases, label, ha='center', va='bottom')

plt.xlabel('월')
plt.ylabel('발생건수')
plt.title('월별 질병 발생건수 및 가장 많이 발생한 질병')
plt.xticks(range(1, 13))
plt.legend()
plt.tight_layout()
plt.show()


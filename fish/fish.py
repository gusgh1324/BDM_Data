import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 데이터 로드
data = pd.read_csv('경상남도_수산생물 질병 발생 현황_20230731.csv', encoding='cp949')

# '연쇄구균증'과 '연쇄구균'을 같은 병으로 취급
data['병명'] = data['병명'].replace('연쇄구균', '연쇄구균증').replace('비브리오병', '비브리오증')

# 상위 수산생물 5종 추출
top_species = data['품종'].value_counts().head(5).index.tolist()

# 상위 수산생물 5종의 질병 데이터 추출
top_species_data = data[data['품종'].isin(top_species)]

# 전체 수산생물에서 많이 발생하는 질병 Top 3 출력
overall_disease_counts = data[data['병명'] != '없음']['병명'].value_counts().head(5)
overall_top_diseases = overall_disease_counts.index.tolist()
print(f"전체 수산생물에서 많이 발생하는 질병 Top 5: {overall_top_diseases}")

# 품종별로 가장 많이 발생하는 질병 그래프
for species in top_species:
    species_data = top_species_data[top_species_data['품종'] == species]
    disease_counts = species_data['병명'].value_counts()

    # 2% 미만의 데이터는 '기타'로 처리
    total_count = disease_counts.sum()
    threshold = total_count * 0.02
    other_diseases = disease_counts[disease_counts < threshold].index.tolist()
    disease_counts = disease_counts[disease_counts >= threshold]
    disease_counts['기타'] = species_data['병명'].isin(other_diseases).sum()

    # 그래프 그리기
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕 폰트 경로
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(disease_counts, labels=disease_counts.index, autopct='%1.1f%%')
    ax.set_title(f"{species}의 질병 발생 현황")

    plt.show()

# 품종별로 많이 발생하는 질병 3가지씩 출력
for species in top_species:
    species_data = top_species_data[top_species_data['품종'] == species]
    disease_counts = species_data['병명'].value_counts().head(3)

    top_diseases = disease_counts.index.tolist()
    print(f"{species}에서 많이 발생하는 질병 Top 3: {top_diseases}")

# 전체 질병 발생 현황 계산 (단, "없음" 제외)
all_disease_counts = data[data['병명'] != '없음']['병명'].value_counts()

# 기타 질병 합치기
threshold = all_disease_counts.sum() * 0.02  # 2% 기준
other_diseases = all_disease_counts[all_disease_counts < threshold].sum()
all_disease_counts = all_disease_counts[all_disease_counts >= threshold]
all_disease_counts['기타'] = other_diseases

plt.figure(figsize=(10, 10))
plt.pie(all_disease_counts, labels=all_disease_counts.index, autopct='%1.1f%%')
plt.title('전체 수산생물의 질병 발생 현황')
plt.show()
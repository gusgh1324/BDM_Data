from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import urllib.request
import os

# 검색쿼리
searchKey = input('검색할 키워드 입력 : ')
saveKey=input('파일로 저장할 키워드 입력 : ')


# 폴더 생성
def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print('Error')


createFolder(f'train_dataset/{searchKey}')

driver = webdriver.Chrome()
driver.get('https://www.google.co.kr/imghp')

# 쿼리 검색 및 검색 버튼 클릭
elem = driver.find_element('name', 'q')
elem.send_keys(searchKey)
elem.send_keys(Keys.RETURN)
# 이미지 스크롤링
while True:
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')  # 브라우저 끝까지 스크롤
    time.sleep(1.5)  # 쉬어주기
    try:
        button = driver.find_element(By.XPATH, '//*[@id="islmp"]/div/div/div/div/div[1]/div[2]/div[2]/input')
        button.click()  # 스크롤을 내리다보면 '결과 더보기'가 있는 경우 버튼 클릭
        time.sleep(1)
    except:
        pass
    if driver.find_element(By.CLASS_NAME, 'OuJzKb.Yu2Dnd').text == '더 이상 표시할 콘텐츠가 없습니다.':  # class 이름으로 가져오기
        break

# 이미지 수집 및 저장
images = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")  # 각 이미지들의 class
count = 1
for image in images:
    try:
        image.click()
        time.sleep(1)
        imgUrl = driver.find_element(By.XPATH,
                                     '//*[@id="Sva75c"]/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div/div[3]/div[1]/a/img[1]').get_attribute(
            "src")
        imgUrl = imgUrl.replace('https', 'http')  # https로 요청할 경우 보안 문제로 SSL에러가 남
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]  # https://docs.python.org/3/library/urllib.request.html 참고
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(imgUrl, f'train_dataset/{saveKey}/{saveKey}_{str(count)}.jpg')  # url을
        print(f'--{count}번째 이미지 저장 완료--')
        count = count + 1
    except Exception as e:
        print('Error : ', e)
        pass

driver.close()

from django.utils import timezone
import pandas as pd
import os
import pickle
from django.conf import settings

# 장고 서버시간
def get_server_time():
    return timezone.now()    

# react에 JsonResponse하는데, NaN response를 막기 위한 none 처리 함수
def convert_none_to_null_1(value):
    if pd.isna(value):
        return None
    return value

# react에 JsonResponse하는데, NaN response를 막기 위한 none 처리 함수
def convert_none_to_null(program):
    for key, value in program.items():
        if pd.isna(value):
            program[key] = None
    return program

# 로컬 모델을 불러오는 함수
def load_recommendation_model(filename):
    media_path = os.path.join(settings.MEDIA_ROOT, f'{filename}.pkl')
    try:
        with open(media_path, 'rb') as file:
            recommendation_model = pickle.load(file)
        return recommendation_model
    except Exception as e:
        print(f"Error loading model from file: {e}")
        return None
    
# 로컬 데이터 읽는 함수
def read_data_from_local(file_name):
    try:
        file_path = os.path.join('static', file_name)
        data = pd.read_csv(file_path, encoding='euc-kr')
        data = data.fillna(value={'disp_rtm': "null", "series_nm":"null"})  
        print(f'Successfully read data from: {file_path}')
        sample_data = data.head(1)
        print(f"Sample data: {sample_data}")
        return data
    except Exception as e:
        print(f'Error reading data from local file: {e}')
        raise e  
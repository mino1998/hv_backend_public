import os
import random
import pandas as pd
from django.http import JsonResponse
from rest_framework.response import Response
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import logging
import json
import ast
from hv_back.utils import get_server_time
from hv_back.utils import load_recommendation_model
from hv_back.utils import convert_none_to_null_1
from hv_back.utils import convert_none_to_null
from hv_back.utils import read_data_from_local
from django.utils import timezone
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import BaselineOnly
from surprise import accuracy


# recommendation_1 : 시간대 인기차트 프로그램명이 저장된 데이터를 부르는 함수    
def get_assets_by_time(server_time):
    time_view_df = pd.read_csv('static/time_view_df.csv',encoding='euc-kr')
    server_hour = server_time.hour
    selected_data = time_view_df[time_view_df['time_range'] == server_hour]['top_asset']
    top_assets_list = selected_data.iloc[0].split(', ')
    return top_assets_list

# recommendation_1 : 실시간 인기 프로그램의 프로그램 정보만을 모아서 return하는 함수
def get_programs_by_assets(top_assets):
    asset_df = pd.read_csv('static/asset_df.csv', encoding='euc-kr')
    try:
        selected_programs = asset_df[asset_df['asset_nm'].isin(top_assets)]
        selected_programs = selected_programs.where(pd.notna(selected_programs), None)
        if selected_programs.empty:
            return pd.DataFrame()
        return selected_programs
    except Exception as e:
        logging.exception(f"Error in get_programs_by_assets: {e}")
        return pd.DataFrame()

# recommendation_2 : 장르기반 프로그램 뽑는 함수
def get_programs_by_genre(genre):
    program_data = read_data_from_local('asset_df.csv')
    genre_programs = program_data[program_data['category_l'] == genre]
    genre_programs = genre_programs.where(pd.notna(genre_programs), None)
    programs = genre_programs.to_dict('records')
    return programs


# recommendation_2 : 사용자 장르 찾는 함수
def get_most_watched_genre(subsr):
    subsr_data = read_data_from_local('subsr_max_genre.csv')
    print(f"subsr_data: {subsr_data}")
    try:
        subsr_genre = subsr_data.loc[subsr_data['subsr'].astype(str) == str(subsr), 'top_genres'].iloc[0]
        subsr_genre = eval(subsr_genre)[0] if subsr_genre else None
    except IndexError:
        subsr_genre = None
    return subsr_genre


# recommendation_2 : 예외처리를 위한 랜덤 뽑아주기
def get_random_programs(num_programs):
    program_data = read_data_from_local('asset_df.csv')
    try:
        if not program_data.empty:
            selected_programs = program_data.sample(min(num_programs, len(program_data)))
            programs = [{'asset_nm': row['asset_nm'], 'image': row['image'] if not pd.isna(row['image']) else None} for _, row in selected_programs.iterrows()]
            return programs
        else:
            return []
    except Exception as e:
        logging.exception(f"Error in get_random_programs: {e}")
        return []


# recommendation_3 : 모델을 불러와서 predict값에 따라 프로그램 순위 내림차순 데이터프레임 return
def get_user_recommendations(subsr, vod_df, asset_df, model, top_n=20):
    all_assets = asset_df['asset_nm'].unique()
    subsr_predictions = [model.predict(int(subsr), asset) for asset in all_assets]
    watched_assets = vod_df[vod_df['subsr'].astype(str) == str(subsr)]['asset_nm'].unique()
    rec_assets = [rec.iid for rec in sorted(subsr_predictions, key=lambda x: x.est, reverse=True)
                  if rec.iid not in watched_assets][:top_n]
    subsr_recommendations = pd.DataFrame({
        'subsr': [subsr] * top_n,
        'asset_nm': rec_assets
    })
    return subsr_recommendations


@method_decorator(csrf_exempt, name='dispatch')
class RecommendationView_1(View):
    def post(self, request):
        try:
            server_time = get_server_time()
            print(f'server_time, {server_time}')
            top_assets = get_assets_by_time(server_time)
            print(f'top_assets, {top_assets}')
            selected_programs = get_programs_by_assets(top_assets)
            result_data = selected_programs.apply(lambda x: x.map(convert_none_to_null_1)).to_dict('records')
            print(f'result_data, {result_data}')
            return JsonResponse({'data': result_data}, content_type='application/json')

        except Exception as e:
            logging.exception(f"Error in RecommendationView: {e}")
            return JsonResponse({'error': 'Internal Server Error'}, status=500)

@method_decorator(csrf_exempt, name='dispatch')
class RecommendationView_2(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            subsr = data.get('subsr', None)
            print(f"Received subsr: {subsr}")
            if not subsr:
                return JsonResponse({'error': 'subsr is required'}, status=400)
            most_watched_genre = get_most_watched_genre(subsr)
            print(f"genre!: {most_watched_genre}")
            if most_watched_genre is not None:
                print(f"genre not none!")
                programs = get_programs_by_genre(most_watched_genre)
            else:
                programs = get_random_programs(20)
                print(f"genre none")
            if not programs:
                return JsonResponse({'error': 'No programs available'}, status=404)
            num_programs_to_select = min(20, len(programs))
            selected_programs = programs if num_programs_to_select >= len(programs) else random.sample(programs, num_programs_to_select)
            result_data = [convert_none_to_null(program) for program in selected_programs]
            return JsonResponse({'data': result_data}, content_type='application/json')

        except Exception as e:
            logging.exception(f"Error in RecommendationView_2: {e}")
            return JsonResponse({'error': 'Internal Server Error'}, status=500)
        

@method_decorator(csrf_exempt, name='dispatch')
class RecommendationView_3(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            subsr = data.get('subsr', None)
            print(f"Received subsr: {subsr}")
            if not subsr:
                return JsonResponse({'error': 'subsr is required'}, status=400)
            
            current_time = timezone.now()
            four_months_ago = current_time - timezone.timedelta(days=120)
            four_months_ago_month = four_months_ago.strftime('%m')
            three_months_ago = current_time - timezone.timedelta(days=90)
            three_months_ago_month = three_months_ago.strftime('%m')
        
            model_filename = f'baseline_model_{four_months_ago_month}{three_months_ago_month}'
            recommendation_model = load_recommendation_model(model_filename)
            if recommendation_model is None:
                return JsonResponse({'error': 'Failed to load the recommendation model'}, status=500)
            vod_df=read_data_from_local('vod_df.csv')
            asset_df=read_data_from_local('asset_df.csv')
            programs = get_user_recommendations(subsr=subsr, vod_df=vod_df, asset_df=asset_df, model=recommendation_model)
            if programs.empty:
                return JsonResponse({'error': 'No programs available'}, status=404)
            
            recommended_programs_df = asset_df.loc[asset_df['asset_nm'].isin(programs['asset_nm'])]
            result_data = recommended_programs_df.to_dict(orient='records')
            result_data = [convert_none_to_null(program) for program in result_data]
            return JsonResponse({'data': result_data}, content_type='application/json')

        except Exception as e:
            logging.exception(f"Error in RecommendationView_2: {e}")
            return JsonResponse({'error': 'Internal Server Error'}, status=500)
        
        
@method_decorator(csrf_exempt, name='dispatch')
class RecommendationView_4(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            subsr = data.get('subsr', None)
            print(f"Received subsr: {subsr}")
            if not subsr:
                return JsonResponse({'error': 'subsr is required'}, status=400)
            
            vod_log = read_data_from_local('vod_df.csv')
            print(vod_log.head(1))
            vod_log = vod_log[vod_log['subsr'].astype(str) == str(subsr)]
            if vod_log.empty:
                return JsonResponse({'error': 'No viewing history for the specified user'}, status=404)
            
            vod_log['datetime'] = pd.to_datetime(vod_log['date'] + ' ' + vod_log['time'])
            asset_nm = vod_log.loc[vod_log['datetime'].idxmax(), 'asset_nm']
            
            cos_sim = read_data_from_local('contents_sim.csv')
            if cos_sim is not None:
                print(f"cos_sim not none!")
                programs_str = cos_sim[cos_sim['asset_nm'] == asset_nm]['similar_assets'].iloc[0]
                programs = ast.literal_eval(programs_str) if programs_str else []
                asset_df = read_data_from_local('asset_df.csv')
                asset_data = asset_df[asset_df['asset_nm'].isin(programs)]
                selected_programs = asset_data.to_dict(orient='records')
            else:
                print(f"cos_sim is none")
                selected_programs = []
            if not selected_programs:
                return JsonResponse({'error': 'No programs available'}, status=404)
            num_programs_to_select = min(10, len(selected_programs))
            result_data = [convert_none_to_null(program) for program in selected_programs[:num_programs_to_select]]
            return JsonResponse({'data': result_data}, content_type='application/json')
        except Exception as e:
            logging.exception(f"Error in RecommendationView_2: {e}")
            return JsonResponse({'error': 'Internal Server Error'}, status=500)



@method_decorator(csrf_exempt, name='dispatch')
class SearchVeiw(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            program_to_search = data.get('programName', None) 
            print(f"program_to_search:", {program_to_search})
            if not program_to_search:
                return JsonResponse({'error': 'program_to_search is missing'}, status=400)
            try:
                asset_df = read_data_from_local('asset_df.csv')
            except Exception as e:
                logging.exception(f"Error reading data from local file: {e}")
                return JsonResponse({'error': 'Failed to read data file'}, status=500)
            try:
                selected_data = asset_df[asset_df['asset_nm'].str.contains(program_to_search)]
            except KeyError:
                return JsonResponse({'error': 'Invalid filtering condition'}, status=400)
            result_data = selected_data.where(pd.notna(selected_data), None).applymap(convert_none_to_null_1).to_dict('records')
            return JsonResponse({'data': result_data})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logging.exception(f"Error in ProcessButtonClickView: {e}")
            return JsonResponse({'error': 'Internal Server Error'}, status=500)


@method_decorator(csrf_exempt, name='dispatch')  
class ProcessButtonClickView(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
            button_text = data.get('button_text')
            print(f"button_text: {button_text}")
            if not button_text:
                return JsonResponse({'error': 'Button text is missing'}, status=400)
            try:
                asset_df = read_data_from_local('asset_df.csv')
            except Exception as e:
                logging.exception(f"Error reading data from local file: {e}")
                return JsonResponse({'error': 'Failed to read data file'}, status=500)
            try:
                selected_data = asset_df[asset_df['category_h'] == button_text]
            except KeyError:
                return JsonResponse({'error': 'Invalid filtering condition'}, status=400)
            result_data = selected_data.where(pd.notna(selected_data), None).applymap(convert_none_to_null_1).to_dict('records')

            return JsonResponse({'data': result_data})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data'}, status=400)
        except Exception as e:
            logging.exception(f"Error in ProcessButtonClickView: {e}")
            return JsonResponse({'error': 'Internal Server Error'}, status=500)
        


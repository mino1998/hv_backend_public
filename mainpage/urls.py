from django.urls import path
from .views import RecommendationView_1
from .views import RecommendationView_2
from .views import RecommendationView_3
from .views import RecommendationView_4
from .views import SearchVeiw
from .views import ProcessButtonClickView

urlpatterns = [
    path('recommendation_1/', RecommendationView_1.as_view(), name='recommendation_1'),
    path('recommendation_2/', RecommendationView_2.as_view(), name='recommendation_2'),
    path('recommendation_3/', RecommendationView_3.as_view(), name='recommendation_3'),
    path('recommendation_4/', RecommendationView_4.as_view(), name='recommendation_4'),
    path('search/', SearchVeiw.as_view(), name="searchview"),
    path('process_button_click/', ProcessButtonClickView.as_view(), name='process_button_click'),
]

# React
# 카테고리 API 
# 검색 API
# 체크하는거

# HTML, CSS
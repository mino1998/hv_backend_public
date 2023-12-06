from .models import UserAuth
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken

class LoginView(APIView):
    def post(self, request):
        subsr = request.data.get('subsr')
        use_ip = request.data.get('use_ip')
        user_auth = UserAuth.objects.filter(subsr=subsr, use_ip=use_ip).first()

        if user_auth:
            refresh = RefreshToken.for_user(user_auth)
            token = {
                'refresh': str(refresh),
                'access': str(refresh.access_token),
                'subsr': subsr,
            }
            return Response(token, status=status.HTTP_200_OK)
        
        else:
            return Response({'detail': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
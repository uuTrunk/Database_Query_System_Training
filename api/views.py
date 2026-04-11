import math
from rest_framework.views import APIView
from rest_framework.response import Response
from pydantic import BaseModel
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from training.predict import predict

class PredictRequestModel(BaseModel):
    text: str

def calculate_optimal_threads(success_prob: float) -> int:
    p = max(0.0, min(float(success_prob), 0.999999))
    if p > 0.999999: return 1
    if p <= 0: return 5
    required_threads = math.ceil(math.log(0.000001) / math.log(1 - p))
    return max(1, min(required_threads, 5))

class PredictView(APIView):
    def post(self, request):
        try:
            req_data = PredictRequestModel(**request.data)
            score = predict(req_data.text)
        except Exception as e:
            score = 0.2
        
        score = max(0.0, min(score, 1.0))
        threads = calculate_optimal_threads(score)
        return Response({"score": score, "threads": threads})

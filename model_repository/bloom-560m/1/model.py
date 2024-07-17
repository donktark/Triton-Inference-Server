# model.py - Hugging Face 템플릿
import json
import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import pipeline

class TritonPythonModel:
    def initialize(self, args):
        self.generator = pipeline("text-generation", model="bigscience/bloom-560m", device = 0)

    def execute(self, requests):
		# 모델 정의
        responses = []
        # Multi-request
        for request in requests:
						# Inference 입력
            input = pb_utils.get_input_tensor_by_name(request, "input")
            input_string = input.as_numpy()[0].decode()

            # 모델 처리 부분
            pipeline_output = self.generator(input_string, do_sample=True, min_length=50)
            generated_txt = pipeline_output[0]["generated_text"]
            output = generated_txt

            # Inference 출력
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "output",
                        np.array([output.encode()]),
                    )
                ]
            )
            responses.append(inference_response)

        return responses

    def finalize(self, args):
         self.generator = None
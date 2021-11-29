FROM public.ecr.aws/lambda/python:3.7

COPY models ./models

RUN python3.7 -m pip install torch==1.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

COPY requirements.txt ./

RUN python3.7 -m pip install -r requirements.txt -t .

COPY *.py ./

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
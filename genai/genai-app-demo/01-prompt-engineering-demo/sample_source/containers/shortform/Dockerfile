FROM public.ecr.aws/lambda/python:3.12

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt

# Install FFMpeg
# RUN dnf install -y wget tar xz
# RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
# RUN mkdir ffmpeg-release-amd64-static
# RUN tar -xf ffmpeg-release-amd64-static.tar.xz --strip-components=1 -C ffmpeg-release-amd64-static
# RUN rm -f ffmpeg-release-amd64-static.tar.xz
# RUN ln -s /root/ffmpeg-release-amd64-static/ffmpeg /usr/local/bin/ffmpeg

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.lambda_handler" ]

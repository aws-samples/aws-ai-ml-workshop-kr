import boto3
import json
import os
import uuid
import time
import re

# from moviepy.editor import *


MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'
BEDROCK_REGION_NAME=os.environ.get('BEDROCK_REGION_NAME', 'us-west-2')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
MEDIA_CONVERT_ROLE = os.environ.get('MEDIA_CONVERT_ROLE')

bedrock_runtime = boto3.client(
  service_name='bedrock-runtime',
  region_name=BEDROCK_REGION_NAME
)

s3 = boto3.client('s3')
transcribe = boto3.client('transcribe')
mediaconvert = boto3.client('mediaconvert')


# def use_moviepy(tracks):
#     with open(f'/tmp/{video_name}', 'wb') as out_f:
#         s3.download_fileobj(BUCKET_NAME, f'videos/{video_name}', out_f)

#     clip = VideoFileClip(f'/tmp/{video_name}')
#     clips = []
#     for start_time, end_time in tracks:
#         print(start_time, end_time)
#         clips.append(clip.subclip(start_time, end_time))

#     shortened_video_name = f'shortened_{video_name}'
#     shortened = concatenate_videoclips(clips)
#     shortened.write_videofile(f'/tmp/{shortened_video_name}', audio_codec="aac", temp_audiofile=f'/tmp/temp_{video_name}')

#     with open(f'/tmp/{shortened_video_name}', 'rb') as f:
#         s3.upload_fileobj(f, BUCKET_NAME, f'videos/{shortened_video_name}')


def get_response(prompt, command):
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "text",
                        "text": command
                    }
                ]
            }
        ],
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "top_k": 50,
        "top_p": 0.92,
        "temperature": 0.9
    }

    # Run Bedrock API
    response = bedrock_runtime.invoke_model(
        modelId=MODEL_ID,
        contentType='application/json',
        accept='application/json',
        body=json.dumps(body)
    )

    response_body = json.loads(response.get('body').read())
    output = response_body['content'][0]['text']

    return output

def stitching_clips(video_name, tracks):
    tracks_unsorted = []
    for start_time, end_time in tracks:
        st = start_time.replace('.', ':', 1)[:-1]
        et = end_time.replace('.', ':', 1)[:-1]
        # print(st, et)
        tracks_unsorted.append(f'{st} {et}')

    input_clippings = []
    for track in sorted(tracks_unsorted):
        st, et = track.split()
        print(st, et)
        input_clippings.append(
            {
                "StartTimecode": st,
                "EndTimecode": et
            }            
        )

    shortened_video_name = f'shortened_{video_name}'
    only_name, _ = os.path.splitext(shortened_video_name)
    settings = {
        "TimecodeConfig": {
            "Source": "ZEROBASED"
        },
        "OutputGroups": [
            {
                "CustomName": "test",
                "Name": "File Group",
                "Outputs": [
                    {
                        "ContainerSettings": {
                            "Container": "MP4",
                            "Mp4Settings": {}
                        },
                        "VideoDescription": {
                            "CodecSettings": {
                                "Codec": "H_264",
                                "H264Settings": {
                                    "MaxBitrate": 4500000,
                                    "RateControlMode": "QVBR",
                                    "SceneChangeDetect": "TRANSITION_DETECTION"
                                }
                            }
                        },
                        "AudioDescriptions": [
                            {
                                "AudioSourceName": "Audio Selector 1",
                                "CodecSettings": {
                                    "Codec": "AAC",
                                    "AacSettings": {
                                    "Bitrate": 96000,
                                    "CodingMode": "CODING_MODE_2_0",
                                    "SampleRate": 48000
                                    }
                                }
                            }
                        ]
                    }
                ],
                "OutputGroupSettings": {
                    "Type": "FILE_GROUP_SETTINGS",
                    "FileGroupSettings": {
                        "Destination": f"s3://{BUCKET_NAME}/videos/{only_name}",
                        "DestinationSettings": {
                            "S3Settings": {
                            "StorageClass": "STANDARD"
                            }
                        }
                    }
                }
            }
        ],
        "FollowSource": 1,
        "Inputs": [
            {
                "InputClippings": input_clippings,
                "AudioSelectors": {
                    "Audio Selector 1": {
                        "DefaultSelection": "DEFAULT"
                    }
                },
                "VideoSelector": {},
                "TimecodeSource": "ZEROBASED",
                "FileInput": f"s3://{BUCKET_NAME}/videos/{video_name}"
            }                
        ]
    }        

    response = mediaconvert.create_job(    
        Role=MEDIA_CONVERT_ROLE,
        Settings=settings
    )

    job_id = response['Job']['Id']
    status = response['Job']['Status']
    print(job_id, status)

    while status == 'SUBMITTED' or status == 'PROGRESSING':
        time.sleep(1)
        response = mediaconvert.get_job(Id=job_id)
        status = response['Job']['Status']
        # print(status)

    print(status)
    print(response)

    return shortened_video_name

   

def lambda_handler(event, context):
    print(event)

    video_name = json.loads(event['body'])['name']
    job_name = str(uuid.uuid1())

    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        LanguageCode='ko-KR',
        MediaFormat='mp4',
        Media={
            'MediaFileUri': f's3://{BUCKET_NAME}/videos/{video_name}'
        },
        OutputBucketName=BUCKET_NAME,
        OutputKey=f'subtitles/{job_name}/output.json',
        Subtitles={
            'Formats': ['vtt']
        }
    )

    response = transcribe.get_transcription_job(
        TranscriptionJobName=job_name
    )

    status = response['TranscriptionJob']['TranscriptionJobStatus']
    while status == 'QUEUED' or status == 'IN_PROGRESS':
        print('Transcription in progress...')
        time.sleep(10)
        response = transcribe.get_transcription_job(
            TranscriptionJobName=job_name
        )
        status = response['TranscriptionJob']['TranscriptionJobStatus']

    print('Transcription is completed.')

    # 내용 요약 작업 수행
    object_key = f'subtitles/{job_name}/output.json'

    response = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)
    content = json.loads(response['Body'].read().decode('utf-8'))

    temp = []
    for transcript in content['results']['transcripts']:
        temp.append(transcript['transcript'])

    transcripts = ' '.join(temp)

    command = "위의 스트립트를 요약해서 제시해주고 주요 키워드를 목록으로 10개 추출합니다. '요약하면 다음과 같습니다'같은 문구는 제외합니다."
    summary_text = get_response(transcripts, command)
    print(summary_text)

    # 스크립트 요약 작업 수행
    object_key = f'subtitles/{job_name}/output.vtt'

    response = s3.get_object(Bucket=BUCKET_NAME, Key=object_key)
    content = response['Body'].read().decode('utf-8')

    command = """위의 script를 요약하지 않습니다.
동영상 내용중 상품을 설명하는 부분만 선별합니다.
선별된 부분은 원본의 Index, 시간, Script 형식을 유지합니다.
선별된 부분의 시간을 다 합치면 55초에서 60초가 되도록 만듭니다.
결과값은 WEBVTT형식을 유지합니다.
앞/뒤에 부가설명은 제외하고, 반드시 원본의 Index, 시간, Script 형식의 Script만 결과로 만듭니다.
'동영상 내 상품설명 부분을 50-60초 내외로 요약한 결과는 다음과 같습니다.'같은 문구는 제외합니다. 
선별된 부분의 각 스크립트의 시간은 2초를 넘어야 합니다.
선별된 부분을 연결하면 상품 설명의 기승전결이 보이도록 자연스러워야 합니다.
"""
    output = get_response(content, command)

    pattern = re.compile(r'(\d{2}:\d{2}:\d{2}.\d{3}) --> (\d{2}:\d{2}:\d{2}.\d{3})')
    tracks = pattern.findall(output)

    shortened_video_name = stitching_clips(video_name, tracks)

    response = {
        "summary": summary_text,
        "shortened": f'videos/{shortened_video_name}'
    }

    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }

import boto3
import json
import csv
import datetime
import time
import random
import os
from multiprocessing import Process

input_data = csv.DictReader(open('StreamLog'))
user_list = open('UserList', 'r')

#Play log generator
def playlog():

    flag = 0
    temp = 0

    for line in input_data:
        # Ingest data to Kinesis Firehose in 0.03 seconds
        if(temp != 0 and temp % 1000 == 0):
            flag += 1

        time.sleep(0.03)
        temp += 1

        raw_data = {}
        raw_data.update(line)

        raw_data['posnewz'] = int(raw_data['posnewz'])
        raw_data['posnewy'] = int(raw_data['posnewy'])
        raw_data['posnewx'] = int(raw_data['posnewx'])
        raw_data['pidx'] = int(raw_data['pidx'])
        raw_data['action'] = int(raw_data['action'])
        raw_data['posoldx'] = int(raw_data['posoldx'])
        raw_data['posoldy'] = int(raw_data['posoldy'])
        raw_data['posoldz'] = int(raw_data['posoldz'])
        raw_data['idx'] = int(raw_data['idx'])

        # Write json file to /tmp/playlog/
        filename = '/tmp/playlog/' + str(flag) + '_playlog.json'
        with open(filename, 'a') as logFile:
            json.dump(raw_data, logFile)
            # Kinesis Agent parsed from each file based on \n
            logFile.write('\n')
            os.chmod(filename, 0o777)
    
    print('all play log has been generated')

def dynamodb():
    client = boto3.resource('dynamodb')
    table = client.Table('userProfile')
    lines = user_list.readlines()

    while True:
        time.sleep(0.01)
        selectUser = int(random.choice(lines))

        # Get current level 
        response = table.get_item(
            Key = {'pidx': selectUser}
        )
        ulevel = response['Item']['ulevel']

        currentTime = str(datetime.datetime.now())

        # Update level
        if(ulevel < 100):
            response = table.update_item(
                Key = {'pidx': selectUser},
                UpdateExpression = "SET ulevel = :ul, utimestamp = :ut",
                ExpressionAttributeValues = {
                    ':ul' : ulevel + 1,
                    ':ut' : currentTime
                },
                ReturnValues = "UPDATED_NEW"
            )

#Execute both fuctions in parallel
proc1 = Process(target = playlog)
proc2 = Process(target = dynamodb)
proc1.start()
proc2.start()
proc1.join()
proc2.join()
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import TextRecognitionMode
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time

import UrlImageRecognizing
import LocalImageRecognizing

from os import walk


def initialize_check():
    print('initializing')

    # Add your Computer Vision subscription key to your environment variables.
    if 'COMPUTER_VISION_SUBSCRIPTION_KEY' in os.environ:
        subscription_key = os.environ['COMPUTER_VISION_SUBSCRIPTION_KEY']
    else:
        print(
            "\nSet the COMPUTER_VISION_SUBSCRIPTION_KEY environment variable."
            "\n**Restart your shell or IDE for changes to take effect.**")
        # sys.exit()
    # Add your Computer Vision endpoint to your environment variables.
    if 'COMPUTER_VISION_ENDPOINT' in os.environ:
        endpoint = os.environ['COMPUTER_VISION_ENDPOINT']
    else:
        print(
            "\nSet the COMPUTER_VISION_ENDPOINT environment variable."
            "\n**Restart your shell or IDE for changes to take effect.**")
        # sys.exit()



    # return ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))


def export_data(cv_client, local_path):
    localCV = LocalImageRecognizing.LocalCV(cv_client, local_path)

    # Filename to write
    filename = "export_data_西餐廳.txt"
    # Open the file with writing permission
    myfile = open(filename, 'a')
    # Write a line to the file
    # Close the file
    print("Exporting {}".format(localCV.get_image_path()))
    myfile.write("\n")
    myfile.write(localCV.get_image_path())
    myfile.write("\n")
    myfile.write(localCV.get_tag_image())
    myfile.write(localCV.get_categorize_image())
    myfile.write(localCV.get_detect_color())
    myfile.close()


if __name__ == '__main__':
    computervision_client = initialize_check()
    url = "https://cdn.hk01.com/di/media/images/1482995/org/9debcaf412ffe39134631798798abe2d.jpg/HY1JGrqV4eyVXssYntBCkA5JVGOPdFs0Hd6yLR3esi0?v=w1920"

    urlCV = UrlImageRecognizing.UrlCV(computervision_client, url)

    '''
    urlCV.describe_image()
    urlCV.tag_image()
    urlCV.color_image()
    '''
    # Filename to write
    filename = "export_data_西餐廳.txt"
    # Open the file with writing permission
    myfile = open(filename, 'w')

    path = "resources/restaurants/西餐廳/"
    myfile.write("AAAAAAAAA\n")

    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
        break

    for i in range(len(f)):
        try:
            export_data(computervision_client, path + f[i])
            print(path + f[i]+" Export succeed.\n")
        except:
            print("Waiting for 30 sec...")
            time.sleep(30)

    print("Process finished")


    '''
    file_path = path + f[0]

    localCV = LocalImageRecognizing.LocalCV(computervision_client, file_path)

    
    localCV.describe_image()
    localCV.detect_color()
    localCV.tag_image()
    # localCV.generate_thumbnail()
    localCV.detect_adult_or_racy()
    localCV.show_image_path()
    '''

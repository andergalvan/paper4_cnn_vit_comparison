import os
import cv2
import torch
from facenet_pytorch import MTCNN

# CPU or GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset paths
INPUT_PATH = '/home/ubuntu/Paper4_CNN_ViT_Comparison/CASIA-WebFace/casia_webface/casia_webface'
OUTPUT_PATH = '/home/ubuntu/Paper4_CNN_ViT_Comparison/CASIA-WebFace/casia_webface/casia_webface_imgs_mtcnn'

# Create an MTCNN object to crop the images
mtcnn = MTCNN(post_process=False, device=DEVICE)

def process_images(input_path, output_path):
    
    """ Crop all images in the input directory, saving them in the corresponding output directory. """
    
    # Create the output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate over each person in the input directory
    for person in os.listdir(input_path):
        person_path = os.path.join(input_path, person)
        
        if os.path.isdir(person_path):
            # Create the output directory for the person
            person_output_path = os.path.join(output_path, person)
            if not os.path.exists(person_output_path):
                os.makedirs(person_output_path)

            # Iterate over the images of the person
            for image in os.listdir(person_path):
                image_path = os.path.join(person_path, image)
                
                if os.path.isfile(image_path):
                    image_read = cv2.imread(image_path) # Read the image
                    image_rgb = cv2.cvtColor(image_read, cv2.COLOR_BGR2RGB) # Change to RGB
                    save_path_cropped = os.path.join(person_output_path, image)
                    _, _ = mtcnn(image_rgb, save_path=save_path_cropped, return_prob=True) # Crop the image

    print('All images have been cropped and saved.')

process_images(INPUT_PATH, OUTPUT_PATH)
import torch
import torch.nn.functional as F
import gradio as gr
import numpy as np
import json
import time

import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from skimage.transform import resize

import architecture.ShuffleNetV1.network as ShuffleNetV1
import architecture.ShuffleNetV2.network as ShuffleNetV2

# model to use
global model
model = None

# paths to weights
paths = {}

# shufflenet 1
# group size 3
paths['s1_g3_05'] = 'weights/ShuffleNetV1/Group3/models/0.5x.pth.tar'
paths['s1_g3_10'] = 'weights/ShuffleNetV1/Group3/models/1.0x.pth.tar'
paths['s1_g3_15'] = 'weights/ShuffleNetV1/Group3/models/1.5x.pth.tar'
paths['s1_g3_20'] = 'weights/ShuffleNetV1/Group3/models/2.0x.pth.tar'

# shufflenet 1
# group size 8
paths['s1_g8_05'] = 'weights/ShuffleNetV1/Group8/models/0.5x.pth.tar'
paths['s1_g8_10'] = 'weights/ShuffleNetV1/Group8/models/1.0x.pth.tar'
paths['s1_g8_15'] = 'weights/ShuffleNetV1/Group8/models/1.5x.pth.tar'
paths['s1_g8_20'] = 'weights/ShuffleNetV1/Group8/models/2.0x.pth.tar'

# shufflenet 2
paths['s2_05'] = 'weights/ShuffleNetV2/models/0.5x.pth.tar'
paths['s2_10'] = 'weights/ShuffleNetV2/models/1.0x.pth.tar'
paths['s2_15'] = 'weights/ShuffleNetV2/models/1.5x.pth.tar'
paths['s2_20'] = 'weights/ShuffleNetV2/models/2.0x.pth.tar'

# possible configurations
configurations = [
    's1_g3_05',
    's1_g3_10',
    's1_g3_15',
    's1_g3_20',
    's1_g8_05',
    's1_g8_10',
    's1_g8_15',
    's1_g8_20',
    's2_05',
    's2_10',
    's2_15',
    's2_20'
]

# loads a model into the "model" variable
# configuration is a string that specifies the model
def load_model(configuration):

    print('Loading model ' + configuration)

    # split the configuration into its parameters
    parameters = configuration.split('_')

    # get the path to the weights
    path = paths[configuration]

    # load the weights
    checkpoint = torch.load(path, map_location=torch.device('cpu'))

    # all keys in checkpoint have an unnecessary 'module.' prefix, so we remove it
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}

    # all keys have an unnecessary 'module.' prefix
    # so we remove it
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}

    global model

    # ShuffleNetV1
    if parameters[0] == 's1':

        model_name = 'ShuffleNetV1'
        groups = int(parameters[1][1:])
        size = parameters[2][0] + '.' + parameters[2][1:] + 'x'

        model = ShuffleNetV1.ShuffleNetV1(model_size=size, group=groups)
        
    # ShuffleNetV2
    elif parameters[0] == 's2':
        
        model_name = 'ShuffleNetV2'
        groups = 2
        size = parameters[1][0] + '.' + parameters[1][1:] + 'x'

        model = ShuffleNetV2.ShuffleNetV2(model_size=size)

    else:
        # invalid configuration
        raise Exception('Invalid configuration ' + configuration)

    # load the weights into the architecture
    model.load_state_dict(state_dict)
    model.eval()

    return_string = model_name + ' ' + size + ', ' + str(groups) + ' groups'

    return return_string


# import the class labels
path_labels = 'imagenet-simple-labels.json'
with open(path_labels) as f:
    labels = json.load(f)


# crop a 224x224 region from the center of the image
def center_crop(img):
    """Returns a center crop of an image
    
    Arguments:
        numpy.ndarray -- input image

    Returns:
        numpy.ndarray -- center cropped image
    """

    # get the dimensions of the image
    height, width, _ = img.shape

    # calculate the top left corner
    top = (height - 224) // 2
    left = (width - 224) // 2

    # calculate the bottom right corner
    bottom = top + 224
    right = left + 224

    # crop the image
    img = img[top:bottom, left:right, :]

    return img


# predict the class of an image and time it
def timed_prediction(tensor):
    """Returns the predicted class of an image
    
    Arguments:
        torch.Tensor -- input image

    Returns:
        str -- predicted class
    """

    # start the timer
    start = time.time()

    # predict the class
    with torch.no_grad():
        output = model(tensor)

    # end the timer
    end = time.time()
    elapsed_time = end-start

    # get the top 3 labels
    _, indices = torch.sort(output, descending=True)
    indices = indices[0][:3]

    label = [labels[int(idx.item())] for idx in indices]

    return label, elapsed_time


# predict the class of an image
# returns the top 3 classes and prediction time in seconds
def predict_single(img):
    
    # crop the image
    img = center_crop(img)
    input_tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float()

    # Get the prediction
    prediction, time = timed_prediction(input_tensor)

    text_prediction = prediction[0] + ", " + prediction[1] + ", " + prediction[2]
    text_time = str(round(time, 5)) + " s"

    return text_prediction, text_time

# get predictions from all models
def predict_all(img):
    
    # crop the image
    img = center_crop(img)
    input_tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float()

    predictions = {}

    for configuration in configurations:
        # load the model
        load_model(configuration)

        # Get the prediction
        prediction, time = timed_prediction(input_tensor)

        # turn it into a nice string
        text_prediction = prediction[0] + ", " + prediction[1] + ", " + prediction[2]
        text_time = str(round(time, 5)) + "s"

        predictions[configuration] = text_prediction + " [" + text_time + "]"

    return list(predictions.values())


custom_css = """
#row1 {height: 60vh !important;overflow-y: auto;}
#row2 {height: 80vh !important;overflow-y: auto;}
#label_conf {height: 10vh !important; padding: 0px !important;}
#label_conf .output-class {font-size: var(--text-lg) !important;}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# ShuffleNet Demo")

    with gr.Tab("Single Inference"):

        with gr.Row(elem_id="row0"):

            with gr.Column(scale = 5):
                dropdown = gr.Dropdown(configurations, label="Choose a Model", value="s2_10")

            with gr.Column(scale = 2):
                label_model = gr.Label("ShuffleNetV2 1.0x, 2 groups", label="Model")
            
            dropdown.change(load_model, inputs=dropdown, outputs=label_model)

        with gr.Row(elem_id="row1"):
            with gr.Column(scale = 5):
                input_image_single = gr.Image(shape=(224, 224))

            with gr.Column(scale = 2):
                label_prediction = gr.Label("-", label="Prediction")
                label_time = gr.Label("-", label="Prediction time")

        btn_predict_single = gr.Button("Predict")
        btn_predict_single.click(predict_single, inputs=input_image_single, outputs=[label_prediction, label_time])
        input_image_single.change(predict_single, inputs=input_image_single, outputs=[label_prediction, label_time])

    with gr.Tab("Full Inference"):
       
        conf_labels = {}

        with gr.Row(elem_id="row2"):
            with gr.Column(scale = 4):
                input_image_full = gr.Image(shape=(224, 224))

            with gr.Column(scale = 1):
                for configuration in configurations[:6]:
                    conf_labels[configuration] = gr.Label("-", label=configuration, elem_id = "label_conf")

            with gr.Column(scale = 1):
                for configuration in configurations[6:]:
                    conf_labels[configuration] = gr.Label("-", label=configuration, elem_id = "label_conf")

        btn_predict_full = gr.Button("Predict")
        btn_predict_full.click(predict_all, inputs=input_image_full, outputs=list(conf_labels.values()))
        input_image_full.change(predict_all, inputs=input_image_full, outputs=list(conf_labels.values()))

# load the default model
load_model('s2_10')
demo.launch(inline=False,inbrowser=True)
import torch
import gradio as gr
import json
import time
from tqdm import tqdm

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
# returns a list of top 3 predictions and prediction time in seconds for all models
# the output gets fed into the label_conf labels in the Full Inference tab
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

# test the speed of a model
def speed_test_single(img):

    # crop the image
    img = center_crop(img)
    input_tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float()

    time_sum = 0
    inferences = 100

    # a for loop that uses tqdm:
    for i in tqdm(range(inferences)):

        # Get the prediction
        _, time = timed_prediction(input_tensor)
        time_sum += time

    time_avg = time_sum / inferences

    return time_avg, time_sum

# run speed_test_single on all models
# this function uses the function speed_test_single(img)
def speed_test_full(img):
    
    times = {}

    for configuration in configurations:
        # load the model
        load_model(configuration)

        # Get the prediction
        time_avg, time_sum = speed_test_single(img)

        # turn it into a nice string
        text_time_avg = str(round(time_avg, 5)) + "s"
        text_time_sum = str(round(time_sum, 5)) + "s"

        times[configuration] = text_time_sum

    return list(times.values())



custom_css = """
#row1 {height: 60vh !important;overflow-y: auto;}
#row2 {height: 80vh !important;overflow-y: auto;}
#label_conf_inference {height: 10vh !important; padding: 0px !important;}
#label_conf_inference .output-class {font-size: var(--text-lg) !important;}
#label_conf_speed {height: 10vh !important; padding: 0px !important;}
#label_conf_speed .output-class {font-size: 25pt !important;}
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
                input_image_single = gr.Image(shape=(224, 224), tool=None)

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
                input_image_full = gr.Image(shape=(224, 224), tool=None)

            with gr.Column(scale = 1):
                for configuration in configurations[:6]:
                    conf_labels[configuration] = gr.Label("-", label=configuration, elem_id = "label_conf_inference")

            with gr.Column(scale = 1):
                for configuration in configurations[6:]:
                    conf_labels[configuration] = gr.Label("-", label=configuration, elem_id = "label_conf_inference")

        btn_predict_full = gr.Button("Predict")
        btn_predict_full.click(predict_all, inputs=input_image_full, outputs=list(conf_labels.values()))
        input_image_full.change(predict_all, inputs=input_image_full, outputs=list(conf_labels.values()))

    # here we compare the speed of the models
    with gr.Tab("Speed Test"):

        conf_labels_speed = {}

        with gr.Row(elem_id="row2"):
            with gr.Column(scale = 4):
                input_image_speed = gr.Image(shape=(224, 224), tool=None)

            with gr.Column(scale = 1):
                for configuration in configurations[:6]:
                    conf_labels_speed[configuration] = gr.Label("-", label=configuration, elem_id = "label_conf_speed")

            with gr.Column(scale = 1):
                for configuration in configurations[6:]:
                    conf_labels_speed[configuration] = gr.Label("-", label=configuration, elem_id = "label_conf_speed")

        btn_predict_speed = gr.Button("Predict")
        btn_predict_speed.click(speed_test_full, inputs=input_image_speed, outputs=list(conf_labels_speed.values()))

        gr.Markdown("### Speed Test")
        gr.Markdown("This test does 100 inferences of a 224x224 image and shows the total time for all inferences. This may take a while!")

    with gr.Tab("Scaling"):

        with gr.Row(elem_id="row2"):
            with gr.Column(scale = 1):
                input_image_scale = gr.Image(shape=(224, 224), tool=None)

            with gr.Column(scale = 1):
                output_image_scale = gr.Image(tool=None)

        btn_predict_scale = gr.Button("Scale to 224x224")
        btn_predict_scale.click(center_crop, inputs=input_image_scale, outputs=output_image_scale)
        input_image_scale.change(center_crop, inputs=input_image_scale, outputs=output_image_scale)

    with gr.Tab("Information"):
        gr.Markdown("""
        # Model Information
        Both the model architecture and the weights are taken from the [ShuffleNet-Series GitHub Repo](https://github.com/megvii-model/ShuffleNet-Series).

        ## ShuffleNet V1
        ShuffleNet V1 is a convolutional neural network designed for efficient computation, 
        particularly focusing on mobile devices with very limited computing power. The network 
        gets its name from a novel operation that it introduces: channel shuffle. This operation 
        allows for cross-channel information flow which in turn enables the building of more 
        powerful structures with a fraction of computation cost. The model achieves this by using 
        pointwise group convolutions and channel shuffle, drastically reducing computation cost 
        while maintaining accuracy. Various versions of the model are created by varying the 
        number of groups.

        [arxiv.org](https://arxiv.org/abs/1707.01083)

        ## ShuffleNet V2
        ShuffleNet V2 improves on V1 by considering direct and indirect metrics to measure the 
        network's speed and designing the architecture accordingly. The network considers aspects 
        such as memory access cost, parallelism, and channel selection for the network's speed. 
        The resulting model is even more efficient than V1 for a range of model sizes. It 
        eliminates the pointwise group convolutions and maintains the channel shuffle operation. 
        The improved performance makes it particularly suitable for real-time applications on 
        mobile devices.

        [arxiv.org](https://arxiv.org/abs/1807.11164)

        ## Dataset
        The model is trained on the ImageNet dataset, which consists of 1.2 million images with
        1000 classes. The images are of varying sizes and aspect ratios. Both models are trained on
        images that are center cropped to 224x224. Images are normalized using the mean and standard 
        deviation of the ImageNet dataset. The label names are taken from the ```imagenet-simple-labels```
        repository on GitHub.

        [image-net.org](http://www.image-net.org/) [imagenet-simple-labels](https://github.com/anishathalye/imagenet-simple-labels/tree/master)
        
        Built by David Rath (david.rath@studium.uni-hamburg.de)
        
        """)

# load the default model
load_model('s2_10')
demo.launch(inline=False,inbrowser=True)
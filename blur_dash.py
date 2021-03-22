import datetime

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import cv2
from io import BytesIO
import base64
import numpy as np
import skimage.exposure
import PIL.Image as Image

import torch
import torchvision
from torchvision import datasets, models, transforms
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Skin Lesion Photo Classifier'
server = app.server


device = torch.device('cpu')
model = torch.load('model_conv_6classes.pth', map_location=device)
labels = np.array(open("class.txt").read().splitlines())

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

app.layout = html.Div([
    html.Div([
        html.H2('Skin Lesion Photo Classifier'),
        html.Strong('This application takes an ISIC image, and tells you whether the image needs to be retaken because it is too blurry and low contrast. If the image is good enough, it will tell you whether to schedule the patient for a consultation.' ),
    ]),

    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '100px',
            'lineHeight': '100px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),

    html.Div(id='output-image-upload'),
])


def parse_contents(contents, filename, date):
    #convert uploaded image file in Pillow image file
    encoded_image = contents.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    bytes_image = BytesIO(decoded_image)
    image_pil = Image.open(bytes_image).convert('RGB')
    image = cv2.imdecode(np.frombuffer(decoded_image, np.uint8), 1)

    blur_threshold = 30
    contrast_threshold = 0.25

    blur_measure = cv2.Laplacian(image, cv2.CV_64F).var()
    if blur_measure < blur_threshold:
        blur_message = f"This image is too blurry. The blur measure for this image is {blur_measure}"
    else:
        blur_message = f"This image is clear. The blur measure for this image is {blur_measure}"


    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness_mean = hsv[...,2].mean()
    

    low_contrast = False
    threshold = 0
    while not low_contrast:
        threshold = round(threshold + 0.025, 3)
        low_contrast = skimage.exposure.is_low_contrast(image, fraction_threshold=threshold)

    print(threshold)
    contrast_state = threshold > contrast_threshold
    print(contrast_state)
    if contrast_state:
        contrast_message = f"This image has good contrast. Contrast level is {threshold}."
    else:
        contrast_message = f"This image has low contrast. Contrast level is {threshold}."

    if blur_measure > blur_threshold and threshold > contrast_threshold:
        img = preprocess(image_pil)
        img = img.unsqueeze(0)
        pred = model(img)
        #print(pred.detach().numpy())
        prediction = labels[torch.argmax(pred)]
        #output_text = html.Strong(f"Prediction is {prediction}")
        if prediction in ["akiec", "bcc", "mel"]:
            output_text = html.H4(f"Please schedule patient for a consultation.")
        elif prediction in ["bkl", "df", "nv", "vasc"]:
            output_text = html.H4(f"Patient does not need another appointment.")
        else:
            output_text = html.H4(f"Something went wrong")

        df = pd.DataFrame({'class':labels, 'probability':pred[0].detach().numpy()})
        output_table = generate_table(df.sort_values(['probability'], ascending=[False]))
    else:
        output_text = html.H4(f"Please retake your image")
        img = preprocess(image_pil)
        img = img.unsqueeze(0)
        pred = model(img)
        #print(pred.detach().numpy())
        prediction = labels[torch.argmax(pred)]

        df = pd.DataFrame({'class':labels, 'probability':pred[0].detach().numpy()})
        output_table = generate_table(df.sort_values(['probability'], ascending=[False]))


    return html.Div([
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        output_text,
        html.Br(),
        html.Strong(blur_message),
        html.Br(),
        html.Strong(contrast_message),
        html.Br(),
        html.Strong(f"The brightness of this image is {brightness_mean}"),
        html.Br(),
        html.Img(src=contents),
        html.Hr(),
        output_table
    ])


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


if __name__ == '__main__':
    #app.run_server(debug=True, port=8050)
    app.run_server(host='0.0.0.0',debug=True, port=8050)

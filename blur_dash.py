import datetime

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import cv2
from io import BytesIO
import base64
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        html.H2('Blur Detector'),
        html.Strong('This application measures the blur of an image using Laplacian Variance'),
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
    #image = Image.open(bytes_image).convert('RGB')
    image = cv2.imdecode(np.frombuffer(decoded_image, np.uint8), 1)

    blur_measure = cv2.Laplacian(image, cv2.CV_64F).var()
    if blur_measure < 100:
        blur_message = f"This image is too blurry. The blur measure for this image is {blur_measure}"
    else:
        blur_message = f"This image is clear. The blur measure for this image is {blur_measure}"


    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness_mean = hsv[...,2].mean()
    
    lab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    L,A,B=cv2.split(lab)
    # compute minimum and maximum in 5x5 region using erode and dilate
    kernel = np.ones((5,5),np.uint8)
    min = cv2.erode(L,kernel,iterations = 1)
    max = cv2.dilate(L,kernel,iterations = 1)

    # convert min and max to floats
    min = min.astype(np.float64)
    max = max.astype(np.float64)

    # compute local contrast
    contrast = (max-min)/(max+min)

    # get average across whole image
    average_contrast = 100*np.mean(contrast)
    
    return html.Div([
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Strong(blur_message),
        html.Br(),
        html.Strong(f"The contrast of this image is {average_contrast}%"),
        html.Br(),
        html.Strong(f"The brightness of this image is {brightness_mean}"),
        html.Br(),
        html.Img(src=contents),
        html.Hr(),
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

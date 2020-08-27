from flask import Flask, request, render_template, send_file
from fastai.vision import *
from flask_cors import CORS
import os
from cv2 import cv2
import numpy as np
from skimage.segmentation import find_boundaries
import warnings
from PIL import Image
from io import BytesIO
import re, time, base64
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")
torch.nn.Module.dump_patches = True

app = Flask(__name__)
CORS(app)

# Create a directory in a known location to save files to.
uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

learn = load_learner('./model_1')
learn_2 = load_learner('./model_2/')
learn_3 = load_learner('./model_3/')

@app.route("/", methods=['GET','POST'])
def home():
    if request.method == 'POST':
        return output(request.files['image'].read())
    else:
        return render_template('index.html')


def output(data):
  img = open_image(BytesIO(data))
  img.resize(torch.Size([img.shape[0], 500, 500]))
  pred_1 = learn.predict(img)[1]
  mask_1 = pred_1.numpy()[0]
  pred_2 = learn_2.predict(img)[1]
  mask_2 = pred_2.numpy()[0]
  pred_3 = learn_3.predict(img)[1]
  mask_3 = pred_3.numpy()[0]
  image = np.asarray(bytearray(data), dtype="uint8")
  img_cv = cv2.imdecode(image, cv2.IMREAD_COLOR)
  img_cv = cv2.resize(img_cv,(500, 500))
  mask_dark_circles = np.copy(mask_1)
  mask_dark_circles[mask_dark_circles==3] = 2
  contours = find_boundaries(mask_dark_circles, connectivity=1, mode='thick', background=(2))
  img_cv[contours == True]= 255
  img_cv[mask_1==3]= (255,0,0)
  img_cv[mask_1 == 4] = 150
  img_cv[mask_2 == 5] = (112,72,242)
  img_cv[mask_3== 6] = 255
  t = time.time()
  cv2.imwrite('./output/' + str(t) + '.png', img_cv, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
  filename = './output/'+str(t)+'.png'
  with open(filename, "rb") as img_file:
    my_string = base64.b64encode(img_file.read()).decode('ascii')
  #return send_file(filename, attachment_filename="output.png")
  return {"image": (my_string)}


if __name__ == "__main__":
    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('',5000), app)
    http_server.serve_forever()

                                                                                                                                                                                         87,26         Bot


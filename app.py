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

def getI420FromBase64(codec, image_path="./uploads/"):
    base64_data = re.sub('^data:image/.+;base64,', '', codec)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    t = time.time()
    # img.save(os.path.join(uploads_dir, secure_filename(str(t) + '.png')), "PNG")
    img.save(image_path + str(t) + '.png', 'PNG')
    return str(t)

app = Flask(__name__)
CORS(app)

# Create a directory in a known location to save files to.
uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

learn = load_learner(os.path.dirname(os.path.abspath(__file__)+'model_1'), 'export.pkl')
learn_2 = load_learner('./model_2/')
learn_3 = load_learner('./model_3/')

@app.route("/", methods=['GET','POST'])
def home():
    if request.method == 'POST':
        data = request.json['image']
        return output(data)
    else:
        return render_template('index.html')


def output(data):
  name = getI420FromBase64(data)
  img = open_image('./uploads/' + name + '.png')
  img.resize(torch.Size([img.shape[0], 500, 500]))
  pred_1 = learn.predict(img)[1]
  mask_1 = pred_1.numpy()[0]
  pred_2 = learn_2.predict(img)[1]
  mask_2 = pred_2.numpy()[0]
  pred_3 = learn_3.predict(img)[1]
  mask_3 = pred_3.numpy()[0]
  img_cv = cv2.imread('./uploads/'+ name + '.png')
  img_cv = cv2.resize(img_cv,(500, 500))
  mask_dark_circles = np.copy(mask_1)
  mask_dark_circles[mask_dark_circles==3] = 2
  contours = find_boundaries(mask_dark_circles, connectivity=1, mode='thick', background=(2))
  img_cv[contours == True]= 255
  img_cv[mask_1==3]= 150
  img_cv[mask_1 == 4] = 150
  img_cv[mask_2 == 5] = 0
  img_cv[mask_3== 6] = 255
  cv2.imwrite('./output/' + name + '.png', img_cv)
  filename = './output/'+name+'.png'
  with open(filename, "rb") as img_file:
    my_string = base64.b64encode(img_file.read()).decode('ascii')
  # return send_file(filename, mimetype='image/gif')
  return {"name": (my_string)}


if __name__ == "__main__":
    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('',5000), app)
    http_server.serve_forever()
                                                                                                                                                                                         87,26         Bot


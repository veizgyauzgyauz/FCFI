from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import base64
import io
from PIL import Image
import numpy as np
import copy
from fastapi.staticfiles import StaticFiles

import os
import sys
import cv2
import torch
import numpy as np

sys.path.insert(0, '.')
from isegm.inference import utils
from isegm.inference.clicker import Click, Clicker
from isegm.utils.vis import draw_with_blend_and_clicks
from isegm.inference.predictors import get_predictor


def get_predictor_and_zoomin_params():
    predictor_params = {}
    zoom_in_params = {
        'target_size': 720,
        'expansion_ratio': 3.0,
    }

    return predictor_params, zoom_in_params

class Segmenter:
    def __init__(self, resume_path, save_path, device):
        os.makedirs(save_path, exist_ok=True)
        self.resume_path = resume_path
        self.device = device
        self.pred_thr = 0.49
        self.load_model()
        self.image = None
        self.previous_mask = None
        self.previous_click_num = 0

    def load_model(self):
        model = utils.load_is_model(self.resume_path, self.device)
        predictor_params, zoomin_params = get_predictor_and_zoomin_params()
        self.predictor = get_predictor(model, 'NoBRS', self.device,
                                      prob_thresh=self.pred_thr,
                                      predictor_params=predictor_params,
                                      zoom_in_params=zoomin_params,
                                      with_flip=True,
                                      )
    
    def reset(self):
        self.previous_mask = None
        self.image = None
        self.previous_click_num = 0

    def segment(self, input_image, foreground_points, background_points, designated_click_radius=-1) -> Image:
        input_image = np.array(input_image).astype(np.uint8)
        if self.image is None:
            self.image = copy.deepcopy(input_image)
        h, w, _ = input_image.shape
        h_rs = h
        w_rs = w
        designated_click_radius = min(h, w, designated_click_radius)
        length_limit = 600
        
        if h > length_limit and h >= w:
            h_rs = length_limit
            w_rs = int(w * length_limit / h)
            input_image = cv2.resize(input_image, (w_rs, h_rs))
        elif w > length_limit and w >= h:
            h_rs = int(h * length_limit / w)
            w_rs = length_limit
            input_image = cv2.resize(input_image, (w_rs, h_rs))
        
        with torch.no_grad():
            self.predictor.set_input_image(input_image)
            clicker = Clicker()
            for foreground_point in foreground_points:
                y = foreground_point['y'] / h * h_rs
                x = foreground_point['x'] / w * w_rs
                clicker.add_click(Click(True, (y, x)))
            for background_point in background_points:
                y = background_point['y'] / h * h_rs
                x = background_point['x'] / w * w_rs
                clicker.add_click(Click(False, (y, x)))
            if self.previous_mask is None:
                self.previous_mask = np.zeros(input_image.shape[:2])
                pred_probs, perform_time, updated_feedback1, updated_feedback2 = self.predictor.get_prediction(clicker, self.previous_mask, 0.0, designated_click_radius, new_click_num=len(clicker) - self.previous_click_num)
            else:
                previous_mask_rs = cv2.resize(self.previous_mask, (w_rs, h_rs))
                pred_probs, perform_time, updated_feedback1, updated_feedback2 = self.predictor.get_prediction(clicker, previous_mask_rs, 1.0, designated_click_radius, new_click_num=len(clicker) - self.previous_click_num)
            pred_probs = cv2.resize(pred_probs, (w, h))
            self.previous_mask = pred_probs
            self.previous_click_num = len(clicker)

app = FastAPI()
app.mount("/static", StaticFiles(directory="demo/static"), name="static")
templates = Jinja2Templates(directory="demo/templates")

save_path = './results'
resume_path = './weights/model_h18s.pth'
device = torch.device(f"cuda:{0}")
segmenter_instance = Segmenter(resume_path, save_path, device)

@app.post("/submit_post_message")
async def post_message(request: Request):
    request_data = await request.json()
    original_image = request_data['image']
    base64_decoded = base64.b64decode(original_image.split(',')[1])
    image = Image.open(io.BytesIO(base64_decoded))

    background_points = request_data['backgroundPoints']
    foreground_points = request_data['foregroundPoints']
    designated_click_radius = request_data['sliderValue']

    segmenter_instance.segment(image, foreground_points, background_points, designated_click_radius)
    
    output_mask = (segmenter_instance.previous_mask > segmenter_instance.pred_thr).astype(np.uint8)
    output_mask_PIL = 255 * np.repeat(output_mask[:,:,None], 3, axis=-1)
    output_mask_PIL = Image.fromarray(output_mask_PIL)
    buf = io.BytesIO()
    output_mask_PIL.save(buf, format='JPEG')
    buf.seek(0)
    mask_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    output_image = draw_with_blend_and_clicks(
        img=segmenter_instance.image, mask=output_mask, clicks_list=None,
        mask_color=(128, 0, 0),
    )
    output_image = Image.fromarray(output_image)
    buf = io.BytesIO()
    output_image.save(buf, format='JPEG')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    response = JSONResponse({'output_mask': mask_base64, 'output_image': image_base64})
    return response


@app.post("/clear_post_message")
async def post_message(request: Request):
    segmenter_instance.reset()

@app.post("/change_color_post_message")
async def post_message(request: Request):
    if segmenter_instance.previous_mask is None:
        return JSONResponse({'output_mask': None, 'output_image': None})
  
    request_data = await request.json()
    color_r = request_data['color_r']
    color_g = request_data['color_g']
    color_b = request_data['color_b']
    color_a = request_data['color_a']
    print(color_r, color_g, color_b, color_a)

    output_mask = (segmenter_instance.previous_mask > segmenter_instance.pred_thr).astype(np.uint8)
    output_mask_PIL = Image.fromarray(output_mask, mode='L')
    buf = io.BytesIO()
    output_mask_PIL.save(buf, format='JPEG')
    buf.seek(0)
    mask_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    output_image = draw_with_blend_and_clicks(
        img=segmenter_instance.image, mask=output_mask,
        alpha=color_a,
        clicks_list=None,
        mask_color=(color_r, color_g, color_b),
    )
    output_image = Image.fromarray(output_image)
    buf = io.BytesIO()
    output_image.save(buf, format='JPEG')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    response = JSONResponse({'output_mask': mask_base64, 'output_image': image_base64})
    return response

@app.get("/")
async def main(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("template.html", {"request": request})

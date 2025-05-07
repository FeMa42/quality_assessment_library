import os
import requests
import torch
import numpy as np
from PIL import Image, ImageOps
import PIL.Image
import glob
from typing import List, Optional, Union
from transformers import AutoProcessor, AutoModelForCausalLM 
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class FlorenceWheelbaseOD:
    def __init__(self):
        torch.cuda.empty_cache()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", torch_dtype=self.torch_dtype, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)

    def estimate_dimension_differences(self, image_folder_generated: str, image_folder_original: str, normalize=True) -> dict:
        generated_vehicle_data = self.get_vehicle_dimensions_from_folder(image_folder_generated, normalize=normalize)
        reference_vehicle_data = self.get_vehicle_dimensions_from_folder(image_folder_original, normalize=normalize)
        length_difference = abs(generated_vehicle_data["depth_of_object"] - reference_vehicle_data["depth_of_object"])
        width_difference = abs(generated_vehicle_data["width_of_object"] - reference_vehicle_data["width_of_object"])
        wheelbase_difference = abs(generated_vehicle_data["wheelbase"] - reference_vehicle_data["wheelbase"])
        height_difference = abs(generated_vehicle_data["height_of_object"] - reference_vehicle_data["height_of_object"])
        final_data = {
            'length_difference': length_difference,
            'width_difference': width_difference,
            'wheelbase_difference': wheelbase_difference,
            'height_difference': height_difference
        }
        return final_data
        
    def get_wheelbase_from_folder(self, image_folder: str) -> List[float]:
        images_in_generated_folder = sorted(glob.glob(os.path.join(image_folder, "*.png")))
        pil_imgs = [PIL.Image.open(image_path) for image_path in images_in_generated_folder]
        ws, _, _, _ = calc_multiview_bbox_dim(pil_imgs, plot=False)
        _, _, image1, _ = find_sideviews(ws, pil_imgs)
        # wheelbase = calculate_wheelbase(image1, show_wheelbase= True, show_bounding_boxes=True, show_plot=True)
        wheelbase, wheel_bb_1, wheel_bb_2 = self.calculate_wheelbase_with_bb(image1)
        return wheelbase, wheel_bb_1, wheel_bb_2

    def get_vehicle_dimensions_from_folder(self, image_folder: str, normalize=False) -> dict:
        images_in_generated_folder = sorted(glob.glob(os.path.join(image_folder, "*.png")))
        if len(images_in_generated_folder) == 0:
            print(f"No images found in {image_folder}")
            return None
        pil_imgs = [PIL.Image.open(image_path) for image_path in images_in_generated_folder]
        ws, hs, xs, ys = calc_multiview_bbox_dim(pil_imgs, plot=False)
        sorted_indices_by_width = sorted(range(len(ws)), key=lambda i: ws[i])
        smallest_width_indices = sorted_indices_by_width[:2]  # Kleinste Breite
        largest_width_indices = sorted_indices_by_width[-2:]  # Größte Breiten
        max_index = largest_width_indices[0] #Side view 1

        # Radstand
        max_index, second_max_index, image1, image2 = find_sideviews(ws, pil_imgs)
        try: 
            wheelbase, wheel_bb_1, wheel_bb_2 = self.calculate_wheelbase_with_bb(image1)
        except:
            print(f"Error in calculate_wheelbase_with_bb")
            wheelbase = 0
        if normalize: 
            normalization = np.mean(hs)
        else:     
            normalization = 1

        height_of_object = np.mean(hs)
        depth_list = [ws[i] for i in largest_width_indices]
        normalized_depth_of_object = np.mean(depth_list) / normalization
        width_list = [ws[i] for i in smallest_width_indices]
        normalized_width_of_object = np.mean(width_list) / normalization
        normalized_wheelbase = wheelbase / normalization

        final_data = {'normalized': normalize,
                    'height_of_object': height_of_object,
                    'depth_of_object': normalized_depth_of_object, 
                    'width_of_object': normalized_width_of_object, 
                    'wheelbase': normalized_wheelbase}
        return final_data

    def calculate_wheelbase(self, img_pil, show_wheelbase= False, show_bounding_boxes=False, show_plot=False):
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        prompt = "<OD>"
        inputs = self.processor(text=prompt, images=img_pil, return_tensors="pt").to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task="<OD>", image_size=(img_pil.width, img_pil.height)
        )

        wheel_bbox1 = parsed_answer['<OD>']['bboxes'][1]
        wheel_bbox2 = parsed_answer['<OD>']['bboxes'][2]

        center_x1 = (wheel_bbox1[0] + wheel_bbox1[2]) / 2
        center_x2 = (wheel_bbox2[0] + wheel_bbox2[2]) / 2

        center_y1 = (wheel_bbox1[1] + wheel_bbox1[3]) / 2
        #center_y2 = (wheel_bbox2[1] + wheel_bbox2[3]) / 2

        wheelbase = abs(center_x1 - center_x2)
        if show_plot:
            fig, ax = plt.subplots()
            ax.imshow(np.array(img_pil))

            if show_wheelbase==True: # Zeichne eine grüne Linie von (center_x1, center_y1) mit der Länge des Achsabstands
                if center_x1 < center_x2:
                    ax.plot(
                        [center_x1, center_x1 + wheelbase],  # X-Werte (+Radstand um Korrektheit zu testen)
                        [center_y1, center_y1],                 # Y-Werte (konstant)
                        color="green", linewidth=2, label="Wheelbase"
                    )
                else:
                    ax.plot(
                        [center_x1, center_x1 - wheelbase],
                        [center_y1, center_y1],
                        color="green", linewidth=2, label="Wheelbase"
                    )
                ax.set_title("Radstand visualisiert")
                ax.legend()

            if show_bounding_boxes==True: #Bounding Boxes zeichnen
                rect1 = patches.Rectangle(
                    (wheel_bbox1[0], wheel_bbox1[1]),
                    wheel_bbox1[2] - wheel_bbox1[0],
                    wheel_bbox1[3] - wheel_bbox1[1],
                    linewidth=1, edgecolor='blue', facecolor='none'
                )
                rect2 = patches.Rectangle(
                    (wheel_bbox2[0], wheel_bbox2[1]),
                    wheel_bbox2[2] - wheel_bbox2[0],
                    wheel_bbox2[3] - wheel_bbox2[1],
                    linewidth=1, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect1)
                ax.add_patch(rect2)

            plt.show()

        return wheelbase


    def calculate_wheelbase_with_bb(self, img_pil):
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        prompt = "<OD>"
        inputs = self.processor(text=prompt, images=img_pil, return_tensors="pt").to(self.device, self.torch_dtype)

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task="<OD>", image_size=(img_pil.width, img_pil.height)
        )

        wheel_bbox1 = parsed_answer['<OD>']['bboxes'][1]
        wheel_bbox2 = parsed_answer['<OD>']['bboxes'][2]

        center_x1 = (wheel_bbox1[0] + wheel_bbox1[2]) / 2
        center_x2 = (wheel_bbox2[0] + wheel_bbox2[2]) / 2

        center_y1 = (wheel_bbox1[1] + wheel_bbox1[3]) / 2
        center_y2 = (wheel_bbox2[1] + wheel_bbox2[3]) / 2

        wheelbase = abs(center_x1 - center_x2)

        wheel_bb_1 = {"xy": (wheel_bbox1[0], wheel_bbox1[1]), "width": wheel_bbox1[2] - wheel_bbox1[0], "height": wheel_bbox1[3] - wheel_bbox1[1]}
        wheel_bb_2 = {"xy": (wheel_bbox2[0], wheel_bbox2[1]), "width": wheel_bbox2[2] - wheel_bbox2[0], "height": wheel_bbox2[3] - wheel_bbox2[1]}
        return wheelbase, wheel_bb_1, wheel_bb_2


def calc_multiview_bbox_dim(pil_imgs, plot=False):
    """ a function calculating the boundingbox width and heights for a given list of PIL images."""
    ws, hs, xs, ys = [], [], [], []
    for pil_img in pil_imgs:
        img_arr = np.array(pil_img)
        # compute bbox
        ret, mask = cv2.threshold(
            np.array(pil_img.split()[-1]), 0, 255, cv2.THRESH_BINARY
        )
        x, y, w, h = cv2.boundingRect(mask)
        ws.append(w)
        hs.append(h)
        xs.append(x)
        ys.append(y)

        if plot:
            fig, ax = plt.subplots()
            ax.imshow(img_arr)
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.set_title(f"A: {w*h}, W/H: {w/h}")

    return ws, hs, xs, ys

def find_sideviews(ws, pil_imgs):
    sorted_indices = sorted(range(len(ws)), key=lambda i: ws[i], reverse=True)
    max_index = sorted_indices[0]
    second_max_index = sorted_indices[1]

    return max_index, second_max_index, pil_imgs[max_index], pil_imgs[second_max_index]
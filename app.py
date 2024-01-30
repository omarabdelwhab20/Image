from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
import io

app = Flask(__name__)

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

def make_prediction(img): 
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
                                          colors=["red" if label=="person" else "green" for label in prediction["labels"]] , width=2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(io.BytesIO(file.read()))
            prediction = make_prediction(img)
            img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2, 0, 1), prediction)

            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
            plt.imshow(img_with_bbox)
            plt.xticks([], [])
            plt.yticks([], [])
            ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

            # Save the figure to a bytes object
            img_bytes = io.BytesIO()
            plt.savefig(img_bytes, format='png')
            img_bytes.seek(0)

            plt.close(fig)

            del prediction["boxes"]
            predicted_probabilities = str(prediction)
            return render_template('result.html', img_data=img_bytes.read(), predicted_probabilities=predicted_probabilities)

    return render_template('index.html')

@app.route('/predictApi', methods=["POST"])
def predict_api():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Please try again. The Image does not exist'})

        image = request.files.get('file')
        img = Image.open(io.BytesIO(image.read()))
        prediction = make_prediction(img)

        # Adjust this part based on your desired JSON response
        response_data = {
            'labels': prediction['labels'],
            'scores': prediction['scores'].tolist()
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

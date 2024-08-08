from flask import Flask, render_template, request
import joblib
import numpy as np
import logging

model = joblib.load('kmodel.lb')
std = joblib.load('stds.lb')

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

cluster_to_crops = {
    0: {
        'crops': ['pigeonpeas', 'pomegranate', 'orange', 'papaya', 'coconut'],
        'images': ['pigeonpeas.jpg', 'pomegranate.jpg', 'orange.jpg', 'papaya.jpg', 'coconut.jpg']
    },
    1: {
        'crops': ["maize", "banana", "papaya"],
        'images': ['maize.jpg', 'banana.jpg', 'papaya.jpg']
    },
    2: {
        'crops': ['maize', 'banana', 'watermelon', 'muskmelon', 'papaya', 'cotton', 'coffee'],
        'images': ['maize.jpg', 'banana.jpg', 'watermelon.jpg', 'muskmelon.jpg', 'papaya.jpg', 'cotton.jpg', 'coffee.jpg']
    },
    3: {
        'crops': ['grapes', 'apple'],
        'images': ['grapes.jpg', 'apple.jpg']
    },
    4: {
        'crops': ['rice', 'pigeonpeas', 'papaya', 'jute', 'coffee'],
        'images': ['rice.jpg', 'pigeonpeas.jpg', 'papaya.jpg', 'jute.jpg', 'coffee.jpg']
    },
    5: {
        'crops': ['maize', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'mango', 'orange', 'papaya'],
        'images': ['maize.jpg', 'pigeonpeas.jpg', 'mothbeans.jpg', 'mungbean.jpg', 'blackgram.jpg', 'lentil.jpg', 'mango.jpg', 'orange.jpg', 'papaya.jpg']
    },
    6: {
        'crops': ['maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'lentil'],
        'images': ['maize.jpg', 'chickpea.jpg', 'kidneybeans.jpg', 'pigeonpeas.jpg', 'lentil.jpg']
    },
    7: {
        'crops': ['pigeonpeas', 'mothbeans', 'lentil', 'mango'],
        'images': ['pigeonpeas.jpg', 'mothbeans.jpg', 'lentil.jpg', 'mango.jpg']
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/get_result", methods=['POST'])                                                                                                                                                                 #type:ignore
def get_result():
    if request.method == "POST":
        try:
            N = float(request.form.get('N', ''))
            P = float(request.form.get('P', ''))
            K = float(request.form.get('K', ''))
            temperature = float(request.form.get('temperature', ''))
            humidity = float(request.form.get('humidity', ''))
            ph = float(request.form.get('ph', ''))
            rainfall = float(request.form.get('rainfall', ''))

            app.logger.debug(f"Received data: N={N}, P={P}, K={K}, temperature={temperature}, humidity={humidity}, ph={ph}, rainfall={rainfall}")
            
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            input_data_transformed = std.transform(input_data)
            cluster = model.predict(input_data_transformed)[0]
            
            app.logger.debug(f"Prediction: Cluster {cluster}")
            
            crops_info = cluster_to_crops.get(cluster, {'crops': ["Unknown Cluster"], 'images': []})
            crops = crops_info['crops']
            images = crops_info['images']
            
            crop_image_pairs = [{'crop': crop, 'image': image} for crop, image in zip(crops, images)]
            
            app.logger.debug(f"Crop-image pairs: {crop_image_pairs}")
            
            return render_template('output.html', crop_image_pairs=crop_image_pairs)
        
        except Exception as e:
            app.logger.error(f"Error during prediction: {e}")
            return render_template('output.html', result="Error occurred during prediction. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)

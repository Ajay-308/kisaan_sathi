import os
import warnings
warnings.simplefilter("ignore")
import tensorflow as tf
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from flask import Flask , render_template,request
from PIL import Image

import cv2
import pickle
import numpy as np
from tensorflow.keras import backend as K
from os import listdir
K.clear_session()

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
im = ''
result = '...'
percentage = '...'
i = 0
imageName = ''
solution = ''


air_quality = pickle.load(open('Air_Quality.pkl','rb'))
fertilizer_model = pickle.load(open('Fertilizer.pkl','rb'))
crop_recommendation = pickle.load(open('croprecommendation2.pkl','rb'))
crop_trading  = pickle.load(open('crop_trading.pkl','rb'))
soil_model = load_model("soilTYPE.h5")  # Replace with your model path

soil_classes = ['Black Soil', 'Cinder Soil', 'Laterite Soil', 'Peat Soil', 'Yellow Soil']

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/croprecommendation/')
def croprecommendation():
    return render_template('crop_recommend.html')

@app.route('/soiltype/')
def soiltype():
    return render_template('SoilType.html')

@app.route('/home/')
def home():
    return render_template('index.html')

@app.route('/fertilizer/')
def fertilizer():
    return render_template('fertilizer.html')

@app.route('/croptrade/')
def croptrade():
    return render_template('Trading.html')

@app.route('/airquality/')
def airquality():
    return render_template('airquality.html')


@app.route('/cropdisease/')
def cropdisease():
    return render_template('upload.html')



@app.route('/predict_airquality',methods=['GET','POST'])
def predictairquality():
    if request.method=='GET':
        return render_template('airquality.html')
    else:
        soi = float(request.form['so2_individual_pollutant_index'])
        noi = float(request.form['no2_individual_pollutant_index'])
        rpi = float(request.form['rspm_individual_pollutant_index'])
        spmi = float(request.form['spm_individual_pollutant_index'])
        sample_data = [[soi,noi,rpi,spmi]]
        prediction = air_quality.predict(sample_data)

        return render_template('airquality.html',output = prediction)

@app.route('/predict_fertilizer',methods=['GET','POST'])
def predictfertilizer():
    if request.method=='GET':
        return render_template('fertilizer.html')
    else:
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        moisture = float(request.form['Moisture'])
        soil_type = request.form['Soil_Type']
        crop_type = request.form['Crop_Type']
        nitrogen = float(request.form['Nitrogen'])
        potassium = float(request.form['Potassium'])
        phosphorus = float(request.form['Phosphorus'])

        if soil_type == 'Black':
            soil_type = 0
        elif soil_type == 'Clayey':
            soil_type = 1
        elif soil_type == 'Loamy':
            soil_type = 2
        elif soil_type == 'Red':
            soil_type = 3
        else:
            soil_type=4

        if crop_type == 'Barley':
            crop_type=0
        if crop_type == 'Cotton':
            crop_type=1
        if crop_type == 'Ground Nuts':
            crop_type=2
        if crop_type == 'Maize':
            crop_type=3
        if crop_type == 'Millets':
            crop_type=4
        if crop_type == 'Oil seeds':
            crop_type=5
        if crop_type == 'Paddy':
            crop_type=6
        if crop_type == 'Pulses':
            crop_type=7
        if crop_type == 'Sugarcane':
            crop_type=8
        if crop_type == 'Tobacco':
            crop_type=9
        else:
            crop_type=10

        fertilizer_data = [[temp,humidity,moisture,soil_type,
                            crop_type,nitrogen,
                            potassium,phosphorus]]
        fertilizer_prediction = fertilizer_model.predict(fertilizer_data)
        if fertilizer_prediction == 0:
            fertilizer_prediction = '10-26-26'
        if fertilizer_prediction == 1:
            fertilizer_prediction = '14-35-14'
        if fertilizer_prediction == 2:
            fertilizer_prediction = '17-17-17'
        if fertilizer_prediction == 3:
            fertilizer_prediction = '20-20'
        if fertilizer_prediction == 4:
            fertilizer_prediction = '28-28'
        if fertilizer_prediction == 5:
            fertilizer_prediction='DAP'
        else:
            fertilizer_prediction = 'Urea'
        

        return render_template('fertilizer.html',fertilizer_output = fertilizer_prediction)
    
@app.route('/predict_crop',methods=['GET','POST'])
def predictcrop():
    if request.method=='GET':
        return render_template('crop_recommend.html')
    else:
        nitrogen = float(request.form['Nitrogen'])
        phosphorus = float(request.form['Phosphorus'])
        potassium = float(request.form['Potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])
        crop_data = [[nitrogen,phosphorus,potassium,temperature,humidity,ph,rainfall]]
        crop_mapping = {
            0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut',
            5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans',
            10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean',
            15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas',
            19: 'pomegranate', 20: 'rice', 21: 'watermelon'
}
        predictcrop =crop_recommendation.predict(crop_data)
        if predictcrop[0] in crop_mapping:
            predictcrop = crop_mapping[predictcrop[0]]
        else:
            predictcrop = "Unknown"
        return render_template('crop_recommend.html',crop_output = predictcrop)
    
@app.route('/predict_tradingvalue',methods=['GET','POST'])
def predicttradingvalue():
    if request.method=='GET':
        return render_template('Trading.html')
    else:
        element = float(request.form['Element'])
        item = float(request.form['Item'])
        year = float(request.form['Year'])
        Unit = float(request.form['Unit'])
        trading_data = [[element,item,year,Unit]]
        predicttradingvalue = crop_trading.predict(trading_data)
        return render_template('Trading.html',trading_output = predicttradingvalue)
@app.route("/upload", methods=["POST"])
def upload():
    global im, result, percentage , i , imageName , solution
    target = os.path.join(APP_ROOT, 'static\\')
    print(f'Target : {target}')

    if not os.path.isdir(target):
        os.mkdir(target)
    for imgg in os.listdir(target):
        try:
            imgPath = target + imgg
            os.remove(imgPath)
            print(f'Removed : {imgPath}')
        except Exception as e:
            print(e)
        
    for file in request.files.getlist("file"):
        print(f'File : {file}')
        i += 1
        imageName = str(i) + '.JPG'
        filename = file.filename
        destination = "/".join([target, imageName])
        print(f'Destination : {destination}')
        file.save(destination)
        print('analysing Image')
        try:
            image = os.listdir('static')
            im = destination
            print(f'Analysing Image : {im}')
        except Exception as e:
            print(e)
        result = "Failed to Analyse"
        percentage = "0 %"
        try:
            detect()
            solution = solutions(result)
        except Exception as e:
            print(f'Error While Loading : {e}')  
    return render_template('complete.html', name=result, accuracy=percentage , img = imageName , soln = solution)

@app.route("/upload_soil", methods=["POST"])
def upload_soil():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Ensure the UPLOAD_FOLDER directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            file.save(filepath)

            # Resize the image to the expected input shape
            image = Image.open(filepath)
            image = image.resize((220, 220))
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array /= 255.0

            predictions = soil_model.predict(image_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = soil_classes[predicted_class_index]

            return render_template('soilAnalyis.html', predicted_class=predicted_class, image_path=filename)

    return render_template('SoilType.html')

def detect():
    global im, result, percentage
    print(f'Image : {im}')
    # resolution
    ht=50
    wd=50
    classNames = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy" , "Potato___Early_blight" , "Potato___healthy" ,  "Potato___Late_blight" ,
        "Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_healthy",
                  "Tomato_Late_blight","Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot",
                  "Tomato_Spider_mites_Two_spotted_spider_mite","Tomato__Target_Spot",
                  "Tomato__Tomato_mosaic_virus","Tomato__Tomato_YellowLeaf__Curl_Virus"]
    totClass = len(classNames)
    print(classNames)
    print(totClass)
    mdl = r"LeafDisease50x50.h5"
    image = cv2.imread(im)
    orig = image.copy()
    try:
        image = cv2.resize(image, (ht, wd))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
    except Exception as e:
        print("Error Occured : ",e)
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(mdl)
    (zero, one,two, three,four,five,six,seven, eight,nine, ten , eleven, twelve , thirteen , fourteen) = model.predict(image)[0]
    prob = [zero, one,two, three,four,five,six,seven, eight,nine, ten , eleven, twelve , thirteen , fourteen]

    maxProb = max(prob)
    maxIndex = prob.index(maxProb)
    label = classNames[maxIndex]
    proba = maxProb
    result = label
    percentage = float("{0:.2f}".format(proba * 100))
    for i in range(0,totClass):
        print(f'{classNames[i]} : {prob[i]}')

Tomato_Bacterial_spot = """
Fertilizers:
1. Bonide Citrus, Fruit & Nut Orchard Spray (32 Oz)
2. Bonide Infuse Systemic Fungicide...
3. Hi-Yield Captan 50W fungicide (1...
4. Monterey Neem Oil

"""

Tomato_Early_blight = """
\n
1. Mancozeb Flowable with Zinc Fungicide Concentrate
2. Spectracide Immunox Multi-Purpose Fungicide Spray Concentrate For Gardens
3. Southern Ag – Liquid Copper Fungicide
4. Bonide 811 Copper 4E Fungicide
5. Daconil Fungicide Concentrate. 

"""
Tomato_healthy = """
\nYour Plant Is Healthier.
"""
Tomato_Late_blight = """
\n
Plant resistant cultivars when available.
Remove volunteers from the garden prior to planting and space plants far enough apart to allow for plenty of air circulation.
Water in the early morning hours, or use soaker hoses, to give plants time to dry out during the day — avoid overhead irrigation.
Destroy all tomato and potato debris after harvest.
"""
Tomato_Leaf_Mold = """
\nFungicides : 
1. Difenoconazole and Cyprodinil
2. Difenoconazole and Mandipropamid
3. Cymoxanil and Famoxadone
4. Azoxystrobin and Difenoconazole

"""
Tomato_Septoria_leaf_spot = """
\n
Use disease-free seed and dont save seeds of infected plants
Start with a clean garden by disposing all affected plants.
Water aids the spread of Septoria leaf spot. Keep it off the leaves as much as possible by watering at the base of the plant only. 
Provide room for air circulation. Leave some space between your tomato plants so there is good airflow.

"""
Tomato_Spider_mites_Two_spotted_spider_mite = """
\n
Prune leaves, stems and other infested parts of plants well past any webbing and discard in trash (and not in compost piles). Don’t be hesitant to pull entire plants to prevent the mites spreading to its neighbors.
Use the Bug Blaster to wash plants with a strong stream of water and reduce pest numbers.
Commercially available beneficial insects, such as ladybugs, lacewing and predatory mites are important natural enemies. For best results, make releases when pest levels are low to medium.
Dust on leaves, branches and fruit encourages mites. A mid-season hosing (or two!) to remove dust from trees is a worthwhile preventative.
Insecticidal soap or botanical insecticides can be used to spot treat heavily infested areas.
"""
Tomato__Target_Spot = """
1. Remove old plant debris at the end of the growing season; otherwise, the spores will travel from debris to newly planted tomatoes in the following growing fc, thus beginning the disease anew. Dispose of the debris properly and don’t place it on your compost pile unless you’re sure your compost gets hot enough to kill the spores.

2. Rotate crops and don’t plant tomatoes in areas where other disease-prone plants have been located in the past year – primarily eggplant, peppers, potatoes or, of course – tomatoes. Rutgers University Extension recommends a three-year rotation cycle to reduce soil-borne fungi.

3. Pay careful attention to air circulation, as target spot of tomato thrives in humid conditions. Grow the plants in full sunlight. Be sure the plants aren’t crowded and that each tomato has plenty of air circulation. Cage or stake tomato plants to keep the plants above the soil.

4. Water tomato plants in the morning so the leaves have time to dry. Water at the base of the plant or use a soaker hose or drip system to keep the leaves dry. Apply a mulch to keep the fruit from coming in direct contact with the soil. Limit to mulch to 3 inches or less if your plants are bothered by slugs or snails.

5. You can also apply fungal spray as a preventive measure early in the season, or as soon as the disease is noticed.

"""
Tomato__Tomato_mosaic_virus = """
\n
Fungicides will not treat this viral disease.
Avoid working in the garden during damp conditions (viruses are easily spread when plants are wet).
Frequently wash your hands and disinfect garden tools, stakes, ties, pots, greenhouse benches, etc. 
Remove and destroy all infected plants.Do not compost.
Do not save seed from infected crops.
"""
Tomato__Tomato_YellowLeaf__Curl_Virus = """
\n
Use a neonicotinoid insecticide, such as dinotefuran (Venom) imidacloprid (AdmirePro, Alias, Nuprid, Widow, and others) or thiamethoxam (Platinum), as a soil application or through the drip irrigation system at transplanting of tomatoes or peppers. 
Cover plants with floating row covers of fine mesh (Agryl or Agribon) to protect from whitefly infestations.
Practice good weed management in and around fields to the extent feasible.
Remove and destroy old crop residue and volunteers on a regional basis.
"""
def solutions(disease):
    switcher = {
        "Tomato_Bacterial_spot": Tomato_Bacterial_spot ,
        "Tomato_Early_blight": Tomato_Early_blight ,
        "Tomato_healthy": Tomato_healthy , 
        "Tomato_Late_blight" : Tomato_Late_blight,
        "Tomato_Leaf_Mold" : Tomato_Leaf_Mold,
        "Tomato_Septoria_leaf_spot" : Tomato_Septoria_leaf_spot,
        "Tomato_Spider_mites_Two_spotted_spider_mite" : Tomato_Spider_mites_Two_spotted_spider_mite,
        "Tomato__Target_Spot" : Tomato__Target_Spot,
        "Tomato__Tomato_mosaic_virus" : Tomato__Tomato_mosaic_virus,
        "Tomato__Tomato_YellowLeaf__Curl_Virus" : Tomato__Tomato_YellowLeaf__Curl_Virus,
        }
    return switcher.get(disease,"Not Found In The List")



if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=5500) 
"""
utils.py
Contains all static data for the LeafScan AI application, including
the class names recognized by the model and the detailed cause/treatment
information for each plant disease.
"""

# Class names exactly corresponding to the model output indices
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Dictionary mapping exact class names to their causes and treatments
disease_info = {
    'Apple___Apple_scab': {
        'cause': 'Fungal infection caused by Venturia inaequalis.',
        'treatment': 'Apply fungicides like captan or mancozeb during early growth stages.'
    },
    'Apple___Black_rot': {
        'cause': 'Fungal infection caused by Botryosphaeria obtusa.',
        'treatment': 'Remove infected plant parts and apply fungicides such as thiophanate-methyl.'
    },
    'Apple___Cedar_apple_rust': {
        'cause': 'Fungal infection caused by Gymnosporangium juniperi-virginianae.',
        'treatment': 'Apply fungicides containing myclobutanil or mancozeb.'
    },
    'Apple___healthy': {
        'cause': 'No disease detected.',
        'treatment': 'No action needed.'
    },
    'Blueberry___healthy': {
        'cause': 'No disease detected.',
        'treatment': 'No action needed.'
    },
    'Cherry_(including_sour)_Powdery_mildew': {
        'cause': 'Fungal disease caused by Podosphaera clandestina.',
        'treatment': 'Apply fungicides such as sulfur or myclobutanil.'
    },
    'Cherry_(including_sour)_healthy': {
        'cause': 'No disease detected.',
        'treatment': 'No action needed.'
    },
    'Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot': {
        'cause': 'Fungal infection caused by Cercospora zeae-maydis.',
        'treatment': 'Use fungicides containing azoxystrobin or pyraclostrobin and remove infected leaves.'
    },
    'Corn_(maize)Common_rust': {
        'cause': 'Fungal infection caused by Puccinia sorghi.',
        'treatment': 'Apply fungicides containing tebuconazole or propiconazole.'
    },
    'Corn_(maize)_Northern_Leaf_Blight': {
        'cause': 'Fungal infection caused by Exserohilum turcicum.',
        'treatment': 'Apply fungicides like chlorothalonil or propiconazole and ensure proper crop rotation.'
    },
    'Corn_(maize)_healthy': {
        'cause': 'No disease detected.',
        'treatment': 'No action needed.'
    },
    'Grape___Black_rot': {
        'cause': 'Fungal infection caused by Guignardia bidwellii.',
        'treatment': 'Use fungicides such as myclobutanil or sulfur. Prune and remove infected parts.'
    },
    'Grape__Esca(Black_Measles)': {
        'cause': 'Fungal disease caused by Phaeoacremonium spp. and Phaeomoniella chlamydospora.',
        'treatment': 'No effective fungicide. Prune and remove infected vines to reduce spread.'
    },
    'Grape__Leaf_blight(Isariopsis_Leaf_Spot)': {
        'cause': 'Fungal infection caused by Isariopsis leaf spot fungus.',
        'treatment': 'Apply fungicides such as mancozeb or copper-based fungicides.'
    },
    'Grape___healthy': {
        'cause': 'No disease detected.',
        'treatment': 'No action needed.'
    },
    'Orange__Haunglongbing(Citrus_greening)': {
        'cause': 'Bacterial infection caused by Candidatus Liberibacter spp.',
        'treatment': 'Currently, no effective cure. Remove and destroy infected trees. Use antibiotics for early stages.'
    },
    'Peach___Bacterial_spot': {
        'cause': 'Bacterial infection caused by Xanthomonas campestris.',
        'treatment': 'Use copper-based fungicides and ensure proper spacing for air circulation.'
    },
    'Peach___healthy': {
        'cause': 'No disease detected.',
        'treatment': 'No action needed.'
    },
    'Pepper,bell__Bacterial_spot': {
        'cause': 'Bacterial infection caused by Xanthomonas vesicatoria.',
        'treatment': 'Use copper-based fungicides and practice crop rotation.'
    },
    'Pepper,bell__healthy': {
        'cause': 'No disease detected.',
        'treatment': 'No action needed.'
    },
    'Potato___Early_blight': {
        'cause': 'Fungal infection caused by Alternaria solani.',
        'treatment': 'Apply fungicides like chlorothalonil and remove affected plant parts.'
    },
    'Potato___Late_blight': {
        'cause': 'Fungal infection caused by Phytophthora infestans.',
        'treatment': 'Use fungicides like metalaxyl or mancozeb, and remove infected foliage.'
    },
    'Potato___healthy': {
        'cause': 'No disease detected.',
        'treatment': 'No action needed.'
    },
    'Raspberry___healthy': {
        'cause': 'No disease detected.',
        'treatment': 'No action needed.'
    },
    'Soybean___healthy': {
        'cause': 'No disease detected.',
        'treatment': 'No action needed.'
    },
    'Squash___Powdery_mildew': {
        'cause': 'Fungal disease caused by Podosphaera xanthii.',
        'treatment': 'Use sulfur-based fungicides and improve air circulation around plants.'
    },
    'Strawberry___Leaf_scorch': {
        'cause': 'Fungal infection caused by Diplocarpon earlianum.',
        'treatment': 'Apply fungicides like captan or mancozeb and remove affected leaves.'
    },
    'Strawberry___healthy': {
        'cause': 'No disease detected.',
        'treatment': 'No action needed.'
    },
    'Tomato___Bacterial_spot': {
        'cause': 'Bacterial infection caused by Xanthomonas campestris.',
        'treatment': 'Use copper-based fungicides and ensure proper spacing for air circulation.'
    },
    'Tomato___Early_blight': {
        'cause': 'Fungal infection caused by Alternaria solani.',
        'treatment': 'Use fungicides containing chlorothalonil and remove infected plant parts.'
    },
    'Tomato___Late_blight': {
        'cause': 'Fungal infection caused by Phytophthora infestans.',
        'treatment': 'Use fungicides like mancozeb or metalaxyl and remove infected leaves.'
    },
    'Tomato___Leaf_Mold': {
        'cause': 'Fungal infection caused by Passalora fulva.',
        'treatment': 'Use fungicides containing chlorothalonil or copper-based fungicides.'
    },
    'Tomato___Septoria_leaf_spot': {
        'cause': 'Fungal disease caused by Septoria lycopersici.',
        'treatment': 'Use fungicides such as chlorothalonil or mancozeb, and remove affected leaves.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'cause': 'Insect pest infestation caused by two-spotted spider mites.',
        'treatment': 'Use miticides like abamectin or insecticidal soap.'
    },
    'Tomato___Target_Spot': {
        'cause': 'Fungal infection caused by Corynespora cassiicola.',
        'treatment': 'Use fungicides such as pyraclostrobin or mancozeb.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'cause': 'Viral infection caused by the Tomato yellow leaf curl virus (TYLCV).',
        'treatment': 'Remove infected plants and control the whitefly vector.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'cause': 'Viral infection caused by the Tomato mosaic virus (ToMV).',
        'treatment': 'Remove infected plants and use virus-resistant varieties.'
    },
    'Tomato___healthy': {
        'cause': 'No disease detected.',
        'treatment': 'No action needed.'
    }
}

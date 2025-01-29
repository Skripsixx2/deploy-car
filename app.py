from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

def load_model():
    global model, label_encoders
    with open('random_forest_car.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Mapping kategori ke angka sesuai dengan encoding yang digunakan saat training
    label_encoders = {
        "buying": ["vhigh", "high", "med", "low"],
        "maint": ["vhigh", "high", "med", "low"],
        "doors": ["2", "3", "4", "5more"],
        "persons": ["2", "4", "more"],
        "lug_boot": ["small", "med", "big"],
        "safety": ["low", "med", "high"]
    }

@app.route('/')
def index():
    return render_template('index.html', classPrediction="None")

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the car evaluation category based on user inputs
    and render the result to the HTML page
    '''
    # Ambil nilai input dari form
    buying = request.form.get("buying")
    maint = request.form.get("maint")
    doors = request.form.get("doors")
    persons = request.form.get("persons")
    lug_boot = request.form.get("lug_boot")
    safety = request.form.get("safety")

    # Konversi input menjadi angka berdasarkan label_encoders
    input_data = {
        "buying": label_encoders["buying"].index(buying),
        "maint": label_encoders["maint"].index(maint),
        "doors": label_encoders["doors"].index(doors),
        "persons": label_encoders["persons"].index(persons),
        "lug_boot": label_encoders["lug_boot"].index(lug_boot),
        "safety": label_encoders["safety"].index(safety),
    }

    # Buat DataFrame dari input
    input_df = pd.DataFrame([input_data])

    # Lakukan prediksi
    prediction = model.predict(input_df)[0]

    # Mapping hasil prediksi ke kategori asli (sesuai dataset)
    class_labels = ["unacc", "acc", "good", "vgood"]
    class_label = class_labels[prediction]

    return render_template(
        'index.html', 
        classPrediction=class_label, 
        buying=buying, 
        maint=maint, 
        doors=doors, 
        persons=persons, 
        lug_boot=lug_boot, 
        safety=safety
    )

if __name__ == '__main__':
    load_model()
    app.run(debug=True)

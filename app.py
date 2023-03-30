from flask import Flask,render_template,request,jsonify
import model
app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def submit():
    if request.method=='POST':
        age =  int(request.form['age'])
        bmi =  float(request.form['bmi'])
        smoker_yes =  int(request.form['smoke'])
        smoker_no =  1-smoker_yes
        children =  int(request.form['child'])
        prediction = model.insurance(age, bmi, children, smoker_no, smoker_yes)

        return render_template('results.html',result='Your Insurance is '+ str(prediction))

if __name__=="__main__":
    app.run(host="0.0.0.0")

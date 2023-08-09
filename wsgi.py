from flask import Flask, render_template
from mako.template import Template

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def hello():
      mytemplate = Template("<html><body><h1>HealthCare ChatBot</h1><p>Enter your symptoms </p><h2>Predicted Disease:</h2><p>${prediction}</p><h2>Description:</h2><p>${descriptions}</p><h2>Precautions:</h2><ul> {% for precaution in precautions %}<li> ${precaution} </li></ul></body></html>")
      return mytemplate.render(prediction="good", 
                               descriptions="Hypoglycemia is a condition in which your blood sugar (glucose) level is lower than normal. Glucose is your body's main energy source. Hypoglycemia is often related to diabetes treatment. But other drugs and a variety of conditions — many rare — can cause low blood sugar in people who don't have diabetes.", 
                               precaution="cough")

    return app

if __name__ == '__main__':
    app = create_app()
    app.run()
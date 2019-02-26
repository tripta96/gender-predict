from flask import Flask, Blueprint
from src.bl import System
from flask_restplus import Api

app = Flask(__name__)
blueprint = Blueprint('api', __name__, url_prefix='/api')

api = Api(blueprint, version='1.0', title='Gender Predictor',
            description='A gender preedicting API')
app.register_blueprint(blueprint)
system = System()

from flask import Flask, render_template, request, Blueprint
from flask_restplus import Api, Resource, reqparse, fields, marshal
from src.System import System
# import json


app = Flask(__name__)

# blueprints used for separating api from frontend
blueprint = Blueprint('api', __name__, url_prefix='/api')
api = Api(blueprint, version='1.0', title='Gender Predictor',
          description='A gender preedicting API')
app.register_blueprint(blueprint)


# sets up "database", reads csv and trains
system = System()

parser = reqparse.RequestParser()
parser.add_argument('name', required=True,
                     location='form')


model = api.model('data', {
    "name": fields.String(description="First Name", required=True),
    "gender": fields.String(description="Gender", required=True)
})


@api.route('/api')
class GenderPredict(Resource):
    @api.doc(description='Returns a prediction from first name')
    @api.expect(parser)
    def post(self):
        args = parser.parse_args()
        return {'gender': system.predict(args['name'])}

@api.route('/api/update')
class updateData(Resource):
    @api.doc(description='Retrain Model')
    def put(self):
        system.train()

    @api.doc(description='Update data on server')
    @api.expect(model)
    def post(self):
        data = request.json
        if(not isinstance(data, list)):
            dict_list = []
            dict_list.append(data)
            system.add_data(dict_list)
        else:
            system.add_data(data)


@app.route('/', methods=['POST', 'GET'])
def home():
    gender = None
    if request.method == 'POST':
        gender = system.predict(request.form['name'])
    return render_template('home.html', gender=gender)


@app.route('/404')
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

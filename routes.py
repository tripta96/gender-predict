from flask import Flask, render_template, request, Blueprint
from flask_restplus import Api, Resource, reqparse, fields
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

# data = {
#     'name':fields.String, 
#     'gender':fields.String
#     }
# data_list = {'data_list': fields.List(fields.Nested(data))}


# MAYBE MAKE THIS JSON AND HAVE MULTIPLE ITEMS GOING IN AND THEN TRAIN
parser_update = parser.copy()
parser_update.add_argument('gender', required=True, choices=('M', 'F'))


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
    @api.expect(parser_update)
    def post(self):
        args = parser_update.parse_args()
        system.add_data(args['name'], args['gender'])

    # # @api.expect(data)
    # # @api.marshal_with(data)
    # def post(self):
    #     # print(api.payload)

    #     # jdata = request.get_json()
    #     # print(jdata)
    #     args = parser_update.parse_args()
    #     # print(args['data'])
    #     # str_fixed = args['data'].replace("'","\"")
    #     # str_fixed = str_fixed.replace("u\"","\"")
    #     # print(str_fixed)
    #     # print(json.loads(str_fixed))
    #     # for name in json.loads(str_fixed['gender']):
    #     #     print(name)
    #     system.add_data(args['name'], args['gender'])


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

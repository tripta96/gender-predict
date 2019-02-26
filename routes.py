from flask import render_template, request
from server import system
from server import app, api
from flask_restplus import Resource, reqparse


parser = reqparse.RequestParser()

parser.add_argument('name', required=True,
                    type=str, location='form')


# MAYBE MAKE THIS JSON AND HAVE MULTIPLE ITEMS GOING IN AND THEN TRAIN
parser2 = reqparse.RequestParser()
parser2.add_argument('name', required=True,
                        type=str, location='form')
parser2.add_argument('gender', choices=('M', 'F'))


@api.route('/api')
class GenderPredict(Resource):
    @api.doc('get a prediction')
    @api.expect(parser)
    def post(self):
        args = parser.parse_args()
        print(args['name'])
        return {'gender': system.predict(args['name'])}

    @api.doc('add new records and retrain')
    @api.expect(parser2)
    def put(self):
        parser.add_argument('gender', choices=('M', 'F'))
        args = parser.parse_args()
        system.add_data(args['name'], args['gender'])
        system.train()


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

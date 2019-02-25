from flask import render_template, request, url_for
from server import app


@app.route('/')
def home():
    return "Hello World!"


@app.route('/404')
@app.errorhandler(404)
def page_not_found(e=None):
    return render_template('404.html'), 404

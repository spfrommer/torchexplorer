from __future__ import annotations

import os

import click
import logging

from flask import Flask, render_template
from flask import send_from_directory
from flask import request


app = Flask(__name__)


# Surpress all non-error messages
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
def secho(text, file=None, nl=None, err=None, color=None, **styles):
    pass
def echo(text, file=None, nl=None, err=None, color=None, **styles):
    pass
click.echo = echo  # type: ignore
click.secho = secho  # type: ignore


vega_spec_path = os.path.join(os.path.dirname(__file__), 'vega_dataless.json')

def get_vega_spec() -> str:
    with open(vega_spec_path) as f:
        return f.read()


@app.route('/data/<path:path>')
def send_report(path):
    return send_from_directory('data', path)


@app.route("/")
def index():
    port = int(request.headers.get('Host').split(':')[1])
    return render_template('index.html', vega_spec=get_vega_spec(), port=port)


if __name__ == "__main__":
    app.run(debug=True)


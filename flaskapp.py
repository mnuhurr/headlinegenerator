
import flask
import torch
import torch.nn.functional as F

from common import read_yaml

from models import TokenGenerator
from dataclasses import dataclass
from generate_headlines import get_tokenizer, generate_headline


device = torch.device('cpu')

@dataclass(frozen=True)
class ModelParams:
    vocab_size: int
    max_sequence_length: int
    d_model: int
    d_ff: int
    n_heads: int
    n_layers: int
    dropout: float = 0.0


def get_model(fn, params):
    model = TokenGenerator(
        vocab_size=params.vocab_size,
        max_sequence_length=params.max_sequence_length,
        d_model=params.d_model,
        d_ff=params.d_ff,
        n_heads=params.n_heads,
        n_layers=params.n_layers,
        dropout=params.dropout)

    model.load_state_dict(torch.load(fn, map_location=device))
    return model


class EndpointAction(object):
    def __init__(self, action, mimetype=None):
        self.action = action
        self.mimetype = mimetype if mimetype is not None else 'text/plain'

    def __call__(self, **kwargs):
        retval = self.action(**kwargs)
        response = flask.Response(retval, status=200, headers={}, mimetype=self.mimetype)
        return response


class HeadlineApp:
    def __init__(self, model, tokenizer, port=58080):

        self.app = flask.Flask(__name__)

        self.tokenizer = tokenizer
        self.model = model
        self.port = port

        self.n_gen = 5

        self.add_endpoint(endpoint='/', endpoint_name='front', handler=self.front, mimetype='text/html', methods=['GET', 'POST'])

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None, mimetype=None, methods=None):
        if methods is None:
            methods = ['GET']

        self.app.add_url_rule(endpoint, endpoint_name, EndpointAction(handler, mimetype=mimetype), methods=methods)

    def run(self):
        self.app.run(host='0.0.0.0', port=self.port)

    def generate(self, **kwargs):
        prompt = kwargs.get('prompt')
        headline = generate_headline(self.model, self.tokenizer, prompt=prompt)
        return headline

    def front(self, **kwargs):
        if flask.request.method == 'GET':
            return flask.render_template('index.html')

        else:
            # generate
            prompt = flask.request.form.get('prompt')

            headlines = []

            for _ in range(self.n_gen):
                hl = generate_headline(self.model, self.tokenizer, prompt=prompt)
                headlines.append(hl)

            return flask.render_template('index.html', default_prompt=prompt, results=headlines)


def main(config_fn='app-settings.yaml'):
    cfg = read_yaml(config_fn)

    max_length = cfg.get('max_sequence_length', 128)

    d_model = cfg.get('d_model', 256)
    d_ff = cfg.get('d_ff', 1024)
    n_heads = cfg.get('n_heads', 8)
    n_layers = cfg.get('n_layers', 8)
    dropout = cfg.get('dropout', 0.2)
    
    model_path = cfg.get('model_path')

    # get tokenizer
    tokenizer = get_tokenizer('data')

    # get model:
    model_params = ModelParams(tokenizer.get_vocab_size(), max_length, d_model, d_ff, n_heads, n_layers, dropout)
    model = get_model(model_path, model_params)

    # start app
    port = cfg.get('port', 58080)
    app = HeadlineApp(model, tokenizer, port=port)
    app.run()

if __name__ == '__main__':
    main()

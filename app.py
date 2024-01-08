from flask import Flask, render_template, request, redirect, url_for
from datetime import datetime
from threading import Thread

from learning_model import fetch_data_multiple, preprocess_data, train_models
from main import generate_signals, execute_trades

app = Flask(__name__)

# Placeholder for global variables (replace with a proper configuration management approach)
config = {
    'schedule_time': '09:00',
    'symbols': ["AAPL", "MSFT", "GOOGL", "AMC", "COST", "LEVI", "SKLZ", "ABNB", "GOOG", "TELL", "BABA", "GFAI", "DEA",
                "DNA", "BOIL"],
    'start_date': '2018-01-01',
    'end_date': datetime.now().strftime('%Y-%m-%d')
}

# Placeholder for models (replace with proper model management)
models = {}
feature_names = {}


def train_and_execute():
    global models, feature_names
    data = fetch_data_multiple(config['symbols'], config['start_date'], config['end_date'])
    models, feature_names = train_models(data)
    signals = generate_signals(models, feature_names, data)
    execute_trades(signals)


@app.route('/')
def index():
    return render_template('index.html', config=config)


@app.route('/configure', methods=['POST'])
def configure():
    global config
    config['schedule_time'] = request.form.get('schedule_time')
    # Update other configuration parameters as needed
    return redirect(url_for('index'))


@app.route('/run_bot', methods=['POST'])
def run_bot():
    # Run the bot in a separate thread to avoid blocking the web server
    thread = Thread(target=train_and_execute)
    thread.start()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)

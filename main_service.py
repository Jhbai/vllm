from flask import Flask, request
from main import predict

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def handle_predict():
    if not request.is_json:
        return {"error": "Request must be JSON"}, 400
    
    params = request.get_json()
    return predict(params)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

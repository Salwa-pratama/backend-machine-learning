from flask import Flask
from flask_cors import CORS

from Controllers.controller_naive import naive_route
from Controllers.controller_id3 import id3_route

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return {"message": "Wine ML API Connected!"}

app.register_blueprint(naive_route, url_prefix="/api")

app.register_blueprint(id3_route, url_prefix="/api")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

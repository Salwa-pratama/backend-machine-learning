from flask import Flask
from flask_cors import CORS

from Controllers.controller_naive import naive_route
from Controllers.controller_id3 import id3_route
from Controllers.controller_rf import random_forest_route
from Controllers.controller_nn import neural_network_route
from Controllers.controller_xgboost import xgboost_route

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return {"message": "Machine Learning API is ready"}

# Naive Bayes x Wine Dataset
app.register_blueprint(naive_route, url_prefix="/api")
# ID3 x Wine Dataset
app.register_blueprint(id3_route, url_prefix="/api")
# Random Forest x Diabetes Dataset
app.register_blueprint(random_forest_route, url_prefix="/api")
# Random Forest x Diabetes Dataset
app.register_blueprint(neural_network_route, url_prefix="/api")
# Xgboost
app.register_blueprint(xgboost_route, url_prefix="/api")


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)

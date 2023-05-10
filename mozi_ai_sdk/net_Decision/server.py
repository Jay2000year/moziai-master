
from flask import Flask, request
print('1111')
import pickle
print('12222')
from mozi_ai_sdk.net_Decision.utils.Inferring import RandomForestInfer, load_model
print('3333')
RFmodel = load_model('./utils/RFmodel_saved.joblib')

app = Flask(__name__)


@app.route('/http/query', methods=['post', 'get'])
def parse_data():
    print('开始解析数据.....')
    if not request.data:
        return 'fail'
    get_data = request.json
    input_features = dict(get_data).get('探测态势')
    print(input_features)
    # if input_features == '推演已结束！':
    #     shutdown_server()
    # shutdown()
    try:
        out, out_prob = RandomForestInfer(RFmodel, input_features)
        # ndarray转换成list
        out = out.tolist()
        out = pickle.dumps(out)
        return out
    except Exception as e:
        print(e)

    # return jsonify({"类型编号":out})


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


if __name__ == "__main__":

    app.run(host='192.168.3.118', port=5000)

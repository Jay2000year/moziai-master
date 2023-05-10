from pickle import load
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from scipy.special import softmax
from sklearn import tree
import joblib
from Integer_Line import get_line
import threat_cal



value_DF = pd.read_excel('Value_data.xlsx',header=0)

def load_model(file_path):
    loaded_RFmodel = joblib.load(file_path)
    return loaded_RFmodel


def RandomForestInfer(RFmodel, Infer_input_x):
    InferOut = RFmodel.predict(Infer_input_x)
    InferOut_proba = RFmodel.predict_proba(Infer_input_x)
    return InferOut, InferOut_proba


def With_Prior_Infer(RFmodel, Infer_input_x, Prior=1,):
    # InferOut = RFmodel.predict(Infer_input_x)
    InferOut_proba = RFmodel.predict_proba(Infer_input_x)
    Prior = np.asarray(Prior)
    PostProba = Prior_fuse(Prior, InferOut_proba)
    postOut = PostProba.argsort()[:,::-1][:,0]
    return postOut, PostProba


def Prior_fuse(Prior, Infer_proba):
    PostProba = Prior + Infer_proba
    return PostProba


if __name__ == "__main__":
    RFmodel = load_model('RFmodel_saved.joblib')
    input_features = [[1, 1, 3, 2, 5, 0, 2, 0, 0, 0]]

    ##威胁案例，8个目标坐标以及自身目标
    grid_size = 800/5
    thread_ins = threat_cal.ThreatField(grid_size=int(np.ceil(grid_size)))
    #自身目标
    self_pos = np.array([700,700])/5
    #8个目标坐标
    target_pos = np.array([[350, 200],[450, 200],[250, 250],[550, 250],[350, 350],[450, 350], [400, 600], [400, 200]])/5
    catogory = ["A","A","A","A","A","A","B","C",]

    ## 计算威胁，输出关于所有目标的威胁求和，已归一化，maxmin形式。
    thread_ins.CalThreatField(target_pos, catogory)   
    ThreatFieldArray = thread_ins.ThreatFieldArray
    tar2risk = []
    for postuple in target_pos:
        
        points = get_line(self_pos,postuple)
        Risk = 0
        for pt in points:
            Risk = Risk + ThreatFieldArray[pt]
        tar2risk.append(Risk)
    tar2risk = np.asarray(tar2risk)
    tar2risk = (tar2risk-tar2risk.min())/(tar2risk.max()-tar2risk.min())
    ### 输出威胁列表，对准对应8个目标
    print(tar2risk)


    ## 2D-array, shape ==> (n_samples, n_features)
    ## Template, input [Carrier, Cruiser, Destroyer, Replenishment oiler, fighter, Airborne Early Warning and Control, helicopter, Fuel dispenser, Amphibious assault ship, Amphibious transport dock]
    ## Template, output [S-CSG, M-CSG, ARG, SAG]
    type_names = ["S-CSG", "M-CSG", "ARG", "SAG"]

    # out, out_prob =RandomForestInfer(RFmodel, input_features)
    ## prior先验概率分布，与舰队类型匹配。
    out, out_prob = With_Prior_Infer(RFmodel=RFmodel, Infer_input_x=input_features, Prior=[0.25,0.25,0.5,0],)

    type2val = {}
    for type_name in value_DF['类型'].unique():
        sub_DF = value_DF[value_DF['类型']==type_name]

        ls_avg_val = []
        for name in sub_DF["节点名称"].unique():
            ls_avg_val.append((name,sub_DF[sub_DF["节点名称"]==name]['体系价值'].mean()))

        ls_avg_val.sort(key=lambda x:x[1],reverse=True)
        type2val[type_name]=ls_avg_val



    for out_ind in out:
        t_n = type_names[out_ind]
        sorted_value_ls = type2val[t_n]
        ## 输出价值列表
        print(sorted_value_ls)
        ### 列表输出，对应每次决策识别结果。




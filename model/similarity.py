import numpy as np
class Similarity:
    def __init__(self,features,query,how='cos'):
        similars=[]
        if how=='cos':
            for one in features:
                similars.append(self._cos_sin(one, query)) # 틀렸을 확률
        else: # dis
            similars=self._distance(features,query)
        # 가장 유사도가 큰 순으로 
        ids=np.argsort(similars)

        self.scores = [(similars[id], id) for id in ids[::-1]]

    # 코사인 유사도 함수
    def _cos_sin(self, A, B):
        return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))
    # 유클리드 거리
    def _distance(self, features, query):
        return np.linalg.norm(np.array(features) - query, axis=1)
    
    def get_score(self):
        return self.scores
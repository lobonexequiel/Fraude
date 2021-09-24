import pandas as pd
from tqdm import tqdm
from connections.fraude.elastic.model import Elastic
from elasticsearch_dsl import Search, Q, MultiSearch

class Fraude_Querys(Elastic.Consultas):

    @classmethod
    def get_data_by_id(cls, elastic_id):
        """
            Get all the info by Elastic ID
        """
        q = Q("term", _id=elastic_id)
        s = Search(using=cls.conex, index="documentos_fraude_gcs").filter(q)
        for hit in s.scan():
            return hit.filename

    @classmethod
    def verify_file_hash(cls, file_hash):
        """
            Term trabaja a nivel de cada palabra. EL FAMOSO INDICE INVERTIDO
        """
        q = Q("term", hash=file_hash)
        s = Search(using=cls.conex, index="documentos_fraude_gcs").filter(q)
        for hit in s.scan():
            return False
        return True

    @classmethod
    def get_df_instances(cls, **kwargs):
        print("Retrieving all instances...")
        s = Search(using=cls.conex, index="documentos_fraude_gcs")
        q = None
        for key, value in kwargs.items():
            q = q & Q("match", **{key: value}) if q else Q("match", **{key: value})
        if(q):
            s = s.filter(q)
        list_dic = []
        for hit in s.scan():
            meta_dic = hit.meta.to_dict()
            dic = {}
            for key in meta_dic:
                dic[f"_{key}"] = meta_dic[key]
            dic.update(hit.to_dict())
            list_dic.append(dic)
        df = pd.DataFrame(list_dic)
        return df
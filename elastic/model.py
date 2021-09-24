from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError
from elasticsearch.helpers import bulk
from elasticsearch_dsl import Search, Q, MultiSearch

import pandas as pd
import certifi

class Elastic:
    class Consultas:
        conex = None

        @classmethod
        def conectar(cls, conex):
            cls.conex = conex

    conex = None

    def __init__(self):
        self._conex_default()
        self.Consultas().conectar(self.conex)

    def _conex_default(self):
        """
        Metodo privado de la clase.
        Genera la conexion por default a la base de Elastic.
        """
        print("Connecting to Elastic...")
        self.conex = Elasticsearch(
            host="44fc4303780345dfb05cc8fb5102f5f7.us-central1.gcp.cloud.es.io",
            port=9243,
            http_auth="gaston:LegalHub2021*",
            use_ssl=True,
            verify_certs=True,
            ca_certs=certifi.where(),
            timeout=2400,
        )

    def nuevo_indice(self, indice):
        """
        Crea un nuevo indice a la base elastic vinculada.

        :arg indice: Nombre del indice a crear
        :type indice: Str
        :return: None
        """
        if self.conex is None:
            self._conex_default()
        try:
            self.conex.indices.create(index=indice)
        except RequestError:
            print("Indice {} ya existe!".format(indice))

    def delete_indice(self,indice):
        if self.conex is None:
            self._conex_default()
        try:
            self.conex.indices.delete(index=indice)
        except:
            print(f"No existe tal indice {indice}")

    def insertar_doc(self, indice, doc, tipo=None):
        """
        Inserta un nuevo documento en el indice especificado.

        :arg indice: Nombre del indice vinculado al nuevo documento
        :type indice: Str
        :arg doc: Documento a insertar.
        :type doc: Dict(key, value), compuesto por los distintos campos del documento.
        :arg tipo: Tipo del documento a insentar
        :type indice: Str, defaults to None
        :return: None
        """
        if self.conex is None:
            self._conex_default()
        if tipo is None:
            return self.conex.index(index=indice, body=doc)
        else:
            return self.conex.index(index=indice, doc_type=tipo, body=doc)

    def borrar_doc(self, indice, id_doc):
        """
        Borra el documento especificado.

        :arg indice: Nombre del indice vinculado al docmuento a borrar.
        :type indice: Str
        :arg indice: id del documento a borrar.
        :type indice: Str
        :return: None
        """
        if self.conex is None:
            self._conex_default()
        self.conex.delete(index=indice, id=id_doc)

    def operacion_multiple(self, documentos):
        """
        Funcion utilizado para carga de multiples operaciones en la base elastic search

        :arg indice: Listado de documentos y operaciones a realizar.
        :type indice: Dataframe. Además de los datos porpios de documentos se deben especificar los campos:
            "_op_type": Valores posibles delete, update. Si no se informa se ejecuta una operacion create.
            "_index": Indice de ElasticSearch donde impactara la operacion. Obligatorio.
            "_type": Tipo de documento a insertar. Optativo.
            "_id": id del documento a procesar. Optativo en la operacion create, obligatorio en operaciones delete y update.
        :return: None
        """
        if self.conex is None:
            self._conex_default()
        try:
            print("Uploading Data...")
            if(len(documentos) > 10000):
                self._conex_default()
            bulk(self.conex, self._generar_registros(documentos))
        except:
            print("No pudo cargar los documentos. Revisar conexion")

    @staticmethod
    def _generar_registros(df):
        """
        Metodo privado de la clase.
        A partir del def recibido en la funcion operacion_multiple, genera los registros a procesar.
        """
        lista = []
        for _, row in df.iterrows():
            registro = pd.DataFrame.from_dict([row.to_dict()]).dropna(axis=1)
            cabecera = registro[
                [
                    col
                    for col in registro.columns
                    if col in ["_op_type", "_index", "_id", "_type"]
                ]
            ].to_dict("records")
            registro = pd.DataFrame.from_dict([row.to_dict()]).dropna(axis=1)
            datos = registro[
                [
                    col
                    for col in registro.columns
                    if col not in ["_op_type", "_index", "_id", "_type"]
                ]
            ].to_dict("records")
            if cabecera[0]["_op_type"] == "update":
                cabecera[0].update({"doc": datos[0]})
            else:
                cabecera[0].update(datos[0])
            lista.append(*cabecera)
        return lista
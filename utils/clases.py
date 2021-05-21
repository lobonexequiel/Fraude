"""
Archivo creado para generar y guardar las clases que van a soportar
el manejo posterior de los datos de texto y posibles vectores
"""
from utils.clean import general, remove_stops


class Documento(object):

    def __init__(self, expediente, texto):
        """Clase que instancia un documento y sus características
        Se carga el documento como lista de palbras
        El expediente será el codigo de seguimiento para poder volver hacia
        el documento original

        :param texto: str, texto spliteado por espacios
        :param expediente: str, expediente o id de kibana
        :returns:

        """
        self.texto = texto
        self.expediente = expediente
        self.parrafos = []

    def clean_text(self):
        self.texto = general(self.texto)
        self.texto = remove_stops(self.texto)

    def get_sentencia(self):
        return Sentencia(self.expediente, self.texto)

    def get_nombre(self):
        return self.nombre

    def get_contextos(self):
        return self.contextos

    def set_nombres(self, nombres):
        self.nombres = nombres

    def set_contextos(self, contextos):
        self.contextos = contextos

    @staticmethod
    def read_csv(csv_name, _id='id', nombre='text', sep='|'):
        """retorna una lista de obj documentos siempre que el csv 
        tenga entre sus columnas los nombres 'id' y 'text'

        :param csv_name: str, nombre o path del csv 
        :param sep: str, separados de pandas read_csv
        :returns: list, lista de objeto documentos

        """
        import pandas as pd
        documentos = []
        try:
            df = pd.read_csv(csv_name, sep=sep)
            for i in range(len(df)):
                documentos.append(Documento(df.loc[i, _id], df.loc[i, nombre]))
        except Exception as e:
            print(e)
            return None
        return documentos


class Sentencia(Documento):
    """Clase que instacia un grama
    veremos si en cada sentencia se encuentra un dni,cuil, nombre y lo marcamos
    lo ideal es que en cada sentencia solo tengamos un atributo y no más de 1
    """

    def set_grama(self, grama_len):
        parraf_list = self.texto.split()
        l = len(parraf_list)
        self.gramas = list(
            map(
                lambda x: (parraf_list[x:x+grama_len]),
                range(l-grama_len)
            )
        )

    def get_gramas(self):
        return self.gramas

    def set_dni(self, dni):
        self.dni = dni

    def get_dni(self):
        return self.dni

    def set_nombre(self, nombre):
        self.nombre = nombre

    def get_nombre(self):
        return self.nombre

    def set_cuil(self, cuil):
        self.cuil = cuil

    def get_cuil(self):
        return self.cuil

    def set_cuit(self, cuit):
        self.cuit = cuit

    def get_cuil(self):
        return self.cuit

# class Parrafo:

#     """Clase que instacia un párrafo, sección del documento
#     hereda del mismo los atributos como el expediente
#     se carga el párrafo ya tokenizado como una lista de palabras
#     Si el parrafo tiene un título se carga el título como
#     nombre del parrafo
#     nro_parrafo es si contamos los parrafos en orden de
#     ocurrencia y enumeramos
#     """

#     def __init__(self, expediente, parrafo, nro_parrafo):
#         self.expediente = expediente
#         self.parrafo = parrafo
#         self._nro_parrafo = nro_parrafo

#     def get_parrafo(self, lista=False):
#         if lista:
#             return self.parrafo.split()
#         return self.parrafo

#     def set_nombre(self, nombre):
#         """este es el nombre del atributo a reconocer

#         :param nombre: str
#         :returns: None

#         """
#         self.nombre = nombre

#     def get_nombre():
#         return self.nombre

#     def get_nro_parrafo(self):
#         return self._nro_parrafo


# class Word(Sentencia):
#     """Clase que instacia las propiedades de una palabras
#     Cada palabra, segun el modelo que usemos, puede o contener un vector asociado
#     para tener la información lista, a cada palabra se le asocia una frecuencia
#     Dentro de la misma sentencia y dentro de todos los documentos
#     los nombres y datos personales tendran una frecuenta documental baja
#     """

#     def set_palabra(self, palabra):
#         """TODO describe function

#         :param palabra:
#         :returns:

#         """
#         self.palabra = palabra

#     def get_palabra(self):
#         return self.palabra

#     def set_vector(self, vector):
#         self.vector = vector

#     def get_vector(self):
#         return self.vector

#     def set_doc_freq(self, doc_freq):
#         self.doc_freq = doc_freq

#     def get_doc_freq(self):
#         return self.doc_freq

#     def set_term_freq(self, term_freq):
#         self.term_freq = term_freq

#     def get_doc_freq(self):
#         return self.doc_freq

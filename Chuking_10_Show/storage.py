#storage.py
import logging
from pymongo import MongoClient, errors as mongo_errors
from bson import Binary
from gridfs import GridFS

def save_metadata(record: dict, db_name: str, coll_pdf: str, uri: str) -> str:
    try:
        client = MongoClient(uri)
        col = client[db_name][coll_pdf]
        res = col.insert_one(record)
        return res.inserted_id
    except mongo_errors.PyMongoError as e:
        raise RuntimeError(f"Erro ao salvar metadados: {e}")

def save_file_binary(name: str, path: str, parent_id: str,
                     db_name: str, coll_bin: str, uri: str):
    try:
        client = MongoClient(uri)
        col = client[db_name][coll_bin]
        data = open(path, 'rb').read()
        doc = {'filename': name, 'file': Binary(data), 'parent_id': parent_id}
        col.insert_one(doc)
        logging.info(f"PDF binário '{name}' salvo em '{coll_bin}'.")
    except Exception as e:
        raise RuntimeError(f"Erro ao salvar binário: {e}")

def save_gridfs(path: str, name: str,
                db_name: str, bucket: str, uri: str):
    try:
        client = MongoClient(uri)
        fs = GridFS(client[db_name], collection=bucket)
        with open(path, 'rb') as f:
            fid = fs.put(f, filename=name)
        logging.info(f"Arquivo GridFS '{name}' file_id={fid}")
    except Exception as e:
        raise RuntimeError(f"Erro no GridFS: {e}")
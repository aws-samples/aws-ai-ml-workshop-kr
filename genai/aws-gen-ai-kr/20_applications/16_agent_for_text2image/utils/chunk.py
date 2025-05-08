from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class parant_documents():

    @classmethod
    def _create_chunk(cls, docs, chunk_size, chunk_overlap):

        '''
        docs: list of docs
        chunk_size: int
        chunk_overlap: int
        return: list of chunk_docs
        '''

        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )
        # print("doc: in create_chunk", docs )
        chunk_docs = text_splitter.split_documents(docs)

        return chunk_docs

    @classmethod
    def create_parent_chunk(cls, docs, parent_id_key, family_tree_id_key, parent_chunk_size, parent_chunk_overlap):

        parent_chunks = cls._create_chunk(docs, parent_chunk_size, parent_chunk_overlap)
        for i, doc in enumerate(parent_chunks):
            doc.metadata[family_tree_id_key] = 'parent'
            doc.metadata[parent_id_key] = None

        return parent_chunks
    
    @classmethod
    def create_child_chunk(cls, child_chunk_size, child_chunk_overlap, docs, parent_ids_value, parent_id_key, family_tree_id_key):

        sub_docs = []
        for i, doc in enumerate(docs):
            # print("doc: ", doc)
            parent_id = parent_ids_value[i]
            doc = [doc]
            _sub_docs = cls._create_chunk(doc, child_chunk_size, child_chunk_overlap)
            for _doc in _sub_docs:
                _doc.metadata[family_tree_id_key] = 'child'                    
                _doc.metadata[parent_id_key] = parent_id
            sub_docs.extend(_sub_docs)    

            # if i == 0:
            #     return sub_docs

        return sub_docs
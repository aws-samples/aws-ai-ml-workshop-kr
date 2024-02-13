from termcolor import colored
from IPython.core.display import display, HTML

from langchain.docstore.document import Document
from utils.rag import get_semantic_similar_docs, get_lexical_similar_docs, get_ensemble_results
from utils.opensearch import opensearch_utils

def search_hybrid(**kwargs):
    
    assert "query" in kwargs, "Check your query"
    assert "vector_db" in kwargs, "Check your vector_db"
    assert "index_name" in kwargs, "Check your index_name"
    assert "os_client" in kwargs, "Check your os_client"
    assert "Semantic_Search" in kwargs, "Check your Semantic_Search"    
    assert "Lexical_Search" in kwargs, "Check your Lexical_Search"        
    assert "Hybrid_Search" in kwargs, "Check your Hybrid_Search"            
    assert "minimum_should_match" in kwargs, "Check your minimum_should_match"                
    
    verbose = kwargs.get("verbose", False)
    
    print("Query: \n", kwargs["query"])
    # print("Semantic_Search: ", kwargs["Semantic_Search"])
    # print("Lexical_Search: ", kwargs["Lexical_Search"])
    # print("Hybrid_Search: ", kwargs["Hybrid_Search"])    
    
    
    if (kwargs["Semantic_Search"] == True) | (kwargs["Hybrid_Search"] == True):
        similar_docs_semantic = get_semantic_similar_docs(
            vector_db=kwargs["vector_db"],
            query=kwargs["query"],
            k=kwargs.get("k", 5),
            hybrid=True
        )
        if verbose:
            print("##############################")
            print("similar_docs_semantic")
            print("##############################")
            # print(similar_docs_semantic)
            opensearch_pretty_print_documents(similar_docs_semantic)

    if (kwargs["Lexical_Search"] == True)  | (kwargs["Hybrid_Search"] == True):
        similar_docs_keyword = get_lexical_similar_docs(
            query=kwargs["query"],
            minimum_should_match=kwargs.get("minimum_should_match", 50),
#            filter=kwargs.get("filter", []),
            filter= [],
            index_name=kwargs["index_name"],
            os_client=kwargs["os_client"],
            k=kwargs.get("k", 5),
            hybrid=True
        )
        if verbose:
            print("##############################")    
            print("similar_docs_keyword")    
            print("##############################")    
            # print(similar_docs_keyword)        
            opensearch_pretty_print_documents(similar_docs_keyword)            
        

    if kwargs["Hybrid_Search"] == True:
        similar_docs_ensemble = get_ensemble_results(
            doc_lists = [similar_docs_semantic, similar_docs_keyword],
            weights = kwargs.get("ensemble_weights", [.5, .5]),
            algorithm=kwargs.get("fusion_algorithm", "RRF"), # ["RRF", "simple_weighted"]
            c=60,
            k=kwargs.get("k", 5)
        )
        if verbose:
            print("##############################")
            print("similar_docs_ensemble")
            print("##############################")
            # print(similar_docs_ensemble)
            opensearch_pretty_print_documents(similar_docs_ensemble)                        

    
#    similar_docs_ensemble = list(map(lambda x:x[0], similar_docs_ensemble))
    
#    return similar_docs_ensemble


def opensearch_pretty_print_documents(response):
    '''
    OpenSearch 결과인 LIST 를 파싱하는 함수
    '''
    for doc, score in response:
        print(f'\nScore: {score}')
        # print(f'Document Number: {doc.metadata["row"]}')

        # Split the page content into lines
        lines = doc.page_content.split("\n")
        metadata = doc.metadata
        print(lines)
        print(metadata)        
        
        
        
        # print(doc.metadata['origin'])    

        # Extract and print each piece of information if it exists
        # for line in lines:
        #     split_line = line.split(": ")
        #     if len(split_line) > 1:
        #         print(f'{split_line[0]}: {split_line[1]}')

        # print("Metadata:")
        # print(f'Type: {doc.metadata["type"]}')
        # print(f'Source: {doc.metadata["source"]}')        

        print('-' * 50)

def put_parameter(boto3_clinet, parameter_name, parameter_value):

    # Specify the parameter name, value, and type
    parameter_type = 'SecureString'

    try:
        # Put the parameter
        response = boto3_clinet.put_parameter(
            Name=parameter_name,
            Value=parameter_value,
            Type=parameter_type,
            Overwrite=True  # Set to True if you want to overwrite an existing parameter
        )

        # Print the response
        print('Parameter stored successfully.')
        print(response)

    except Exception as e:
        print('Error storing parameter:', str(e))
    

def get_parameter(boto3_clinet, parameter_name):
    # Create a SSM Client

    try:
        # Get the parameter
        response = boto3_clinet.get_parameter(
            Name=parameter_name,
            WithDecryption=True  # Set to True if the parameter is a SecureString
        )

        # Retrieve parameter value from response
        parameter_value = response['Parameter']['Value']

        # Print the parameter value
        # print('Parameter Value:', parameter_value)
        
        return parameter_value

    except Exception as e:
        print('Error retrieving parameter:', str(e))


############################################
# JSON Loader Functions
############################################

from langchain.document_loaders import JSONLoader

# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title")
    metadata["url"] = record.get("url")
    metadata["project"] = record.get("project")    
    metadata["last_updated"] = record.get("last_updated")        

    if "source" in metadata:
        source = metadata["source"].split("/")[-1]
        metadata["source"] = source

    return metadata


def get_load_json(file_path, jq_schema=".[]"):
    
    loader = JSONLoader(
        file_path=file_path,
        #jq_schema='.sections[]',
        #jq_schema='.[]',
        jq_schema=jq_schema,
        content_key="content",
        metadata_func=metadata_func
    )

    data = loader.load()
    
    return data

def show_doc_json(data, file_path):
    file_name = file_path.split("/")[-1]    
    print("### File name: ", file_name)
    print("### of document: ", len(data))
    print("### The first doc")

    print(data[0])        
    
    
def insert_chunk_opensearch(index_name, os_client, chunk_docs, lim_emb):
    for i, doc in enumerate(chunk_docs):
        # print(doc)
        content = doc.page_content      
        content_emb = lim_emb.embed_query(content)
        metadata_last_updated = doc.metadata['last_updated']
        metadata_last_project = doc.metadata['project']        
        metadata_seq_num = doc.metadata['seq_num']                
        metadata_title = doc.metadata['title']    
        metadata_url = doc.metadata['url']                   

        
        # print(content)
        # print(metadata_last_updated)
        # print(metadata_last_project)
        # print(metadata_seq_num)
        # print(metadata_title)
        # print(metadata_url)
                
        # Example document
        doc_body = {
            "text": content,
            "vector_field": content_emb,  # Replace with your vector
            "metadata" : [
                {"last_updated": metadata_last_updated, 
                 "project": metadata_last_project, 
                 "seq_num": metadata_seq_num, 
                 "title": metadata_title, 
                 "url": metadata_url}
            ]
        }
        
        # print(doc_body)

        opensearch_utils.add_doc(os_client, index_name, doc_body, id=f"{i}")
        
        if i == 100:
            break
    
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter



def create_chunk(docs, chunk_size, chunk_overlap):
    '''
    docs: list of docs
    chunk_size: int
    chunk_overlap: int
    return: list of chunk_docs
    '''

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function = len,
    )
    # print("doc: in create_chunk", docs )
    chunk_docs = text_splitter.split_documents(docs)

    
    return chunk_docs


def create_parent_chunk(docs, parent_id_key, family_tree_id_key, parent_chunk_size, parent_chunk_overlap):
    parent_chunks = create_chunk(docs, parent_chunk_size, parent_chunk_overlap)
    for i, doc in enumerate(parent_chunks):
        doc.metadata[family_tree_id_key] = 'parent'        
        doc.metadata[parent_id_key] = None


    return parent_chunks
    
def create_child_chunk(child_chunk_size, child_chunk_overlap, docs, parent_ids_value, parent_id_key, family_tree_id_key):
    sub_docs = []
    for i, doc in enumerate(docs):
        # print("doc: ", doc)
        parent_id = parent_ids_value[i]    
        doc = [doc]
        _sub_docs = create_chunk(doc, child_chunk_size, child_chunk_overlap)
        for _doc in _sub_docs:
            _doc.metadata[family_tree_id_key] = 'child'                    
            _doc.metadata[parent_id_key] = parent_id
        sub_docs.extend(_sub_docs)    
        
        # if i == 0:
        #     return sub_docs
        
    return sub_docs
  
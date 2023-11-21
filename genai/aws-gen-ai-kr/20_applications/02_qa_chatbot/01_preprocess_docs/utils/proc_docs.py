from termcolor import colored
from IPython.core.display import display, HTML

from langchain.docstore.document import Document
from utils.rag import get_semantic_similar_docs, get_lexical_similar_docs, get_ensemble_results

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


def get_load_json(file_path):
    loader = JSONLoader(
        file_path= file_path,
#        jq_schema='.sections[]',
        jq_schema='.[]',        
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
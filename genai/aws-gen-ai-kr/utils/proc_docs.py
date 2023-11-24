from termcolor import colored
from IPython.core.display import display, HTML

from langchain.docstore.document import Document

class LayoutPDFReader_Custom:
    '''
    sections = layout_pdf_reader.doc.sections()
    i = 2
    print(sections[i])
    print("title: ", sections[i].title)
    print("tag: ", sections[i].tag)
    print("parent: ", sections[i].parent)
    print("parent title: ", sections[i].parent.title)
    print("children: ", sections[i].children)
    print("children: ", sections[i].children[0].tag)
    print("children sentences: ", sections[i].children[0].sentences)
    print("chunk: ", sections[i].chunks())
    # print("chunk title: ", sections[i].chunks()[0].title)

    # sections[2].to_context_text()
    display(HTML(sections[i].to_html(include_children=True, recurse=True)))
    '''
    def __init__(self, doc):
        self.doc = doc
        self.chunk_size = len(doc.chunks())
        self.section_size = len(doc.sections())
        self.table_size = len(doc.tables())        
 


    def show_chunk_info(self, show_size=5):
        for idx, chunk in enumerate(self.doc.chunks()):
            print(colored(f"To_context_text {idx}:\n {chunk.to_context_text()} ", "green"))
            print(colored(f"To_text {idx}:\n {chunk.to_text()} ", "red"))
            print(colored(f"Tag {idx}:\n {chunk.tag} ", "red"))

            print("\n")

            if idx == (show_size -1):
                break

    def create_document_with_chunk(self):
        '''
        chunk 와 메타를 langchain Document 오브젝트로 생성
        '''
        doc_list = []
        for idx, chunk in enumerate(self.doc.chunks()):
            doc=Document(
                page_content= chunk.to_text(),
                metadata={"tag": chunk.tag,
                          "row" : idx,
                         }
            )
            doc_list.append(doc)
            
        return doc_list
                
                
    def show_section_info(self, show_size=5):
        for idx, section in enumerate(self.doc.sections()):
            print(colored(f"section title: {idx}:\n {section.title} ", "green"))
            # use include_children=True and recurse=True to fully expand the section. 
            # include_children only returns at one sublevel of children whereas recurse goes through all the descendants
#            display(HTML(section.to_html(include_children=True, recurse=True)))
            display(HTML(section.to_html(include_children=True)))            
            # display(HTML(section.to_html(include_children=True, recurse=True)))
#            display(HTML(section.to_html()))            

            if idx == (show_size -1):
                break
                
#     def show_table_info(self, show_size=5):
#         for idx, table in enumerate(doc.tables()):
#             print(colored(f"table name: {idx}:\n {table.name} ", "green"))
#             display(HTML(table.to_html(include_children=True, recurse=True)))
#             # print(f"table name: {idx}:\n",  HTML(table.to_html()) )            
#             print(colored(f"table name: {idx}:\n {table.sentences} ", "blue"))

#             if idx == (show_size -1):
#                 break

#from utils.rag import get_semantic_similar_docs, get_lexical_similar_docs, get_ensemble_results
from utils.rag import retriever_utils

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
        similar_docs_semantic = retriever_utils.get_semantic_similar_docs(
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
        similar_docs_keyword = retriever_utils.get_lexical_similar_docs(
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
        similar_docs_ensemble = retriever_utils.get_ensemble_results(
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
        print(f'Document Number: {doc.metadata["row"]}')

        # Split the page content into lines
        lines = doc.page_content.split("\n")

        print(lines)
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

        
from IPython.display import Image, display

def show_graph_dag(graph):

    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        # This requires some extra dependencies and is optional
        pass

